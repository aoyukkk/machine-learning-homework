from __future__ import annotations

from pathlib import Path
import csv
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

try:
    from torch.amp import autocast, GradScaler
    _AMP_AUTOCAST_ARGS = {"device_type": "cuda"}
    _AMP_SCALER_ARGS = {"device": "cuda"}
except ImportError:
    from torch.cuda.amp import autocast, GradScaler  # type: ignore
    _AMP_AUTOCAST_ARGS = {}
    _AMP_SCALER_ARGS = {}

from config import CompareConfig
from data import denorm
from models.diffusion_unet import DiffusionUNet
from utils import Timer, save_image_grid, save_loss_curve, to_device


def _make_ddpm_schedules(steps: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    betas = torch.linspace(1e-4, 0.02, steps, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bar


def train_diffusion(
    cfg: CompareConfig,
    train_loader,
    val_loader,
    root: Path,
    device: torch.device,
    ctx: dict | None = None,
) -> dict[str, float | str]:
    ctx = ctx or {"distributed": False, "rank": 0, "local_rank": 0, "is_main": True}
    is_main = ctx["is_main"]
    distributed = ctx["distributed"]
    train_sampler = ctx.get("train_sampler")

    base_model = DiffusionUNet(num_classes=cfg.num_classes, base=cfg.unet_base).to(device)
    base_model.diffusion_steps = cfg.diffusion_steps
    model = base_model
    if distributed:
        model = nn.parallel.DistributedDataParallel(base_model, device_ids=[ctx["local_rank"]])
    elif cfg.use_data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(base_model)

    ema = DiffusionUNet(num_classes=cfg.num_classes, base=cfg.unet_base).to(device)
    ema.diffusion_steps = cfg.diffusion_steps
    ema.load_state_dict(base_model.state_dict())
    for p in ema.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=cfg.amp and device.type == "cuda", **_AMP_SCALER_ARGS)
    betas, alphas, alpha_bar = _make_ddpm_schedules(cfg.diffusion_steps, device)

    losses = []
    timer = Timer()
    model.train()
    for ep in range(cfg.epochs_diffusion):
        if train_sampler is not None:
            train_sampler.set_epoch(ep)

        ep_loss = 0.0
        pbar = tqdm(
            train_loader,
            desc=f"[diffusion] epoch {ep+1}/{cfg.epochs_diffusion}",
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            disable=not is_main,
        )
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(pbar):
            x0, y = to_device(batch, device)
            b = x0.size(0)
            t_idx = torch.randint(0, cfg.diffusion_steps, (b,), device=device)
            a_bar = alpha_bar[t_idx][:, None, None, None]
            eps = torch.randn_like(x0)
            x_t = torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * eps

            t = (t_idx.float() + 1.0) / cfg.diffusion_steps
            with autocast(enabled=cfg.amp and device.type == "cuda", **_AMP_AUTOCAST_ARGS):
                pred = model(x_t, t, y)
                loss = F.mse_loss(pred, eps) / cfg.grad_accum_steps

            scaler.scale(loss).backward()
            if (i + 1) % cfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    src_model = model.module if hasattr(model, "module") else model
                    for p_ema, p in zip(ema.parameters(), src_model.parameters()):
                        p_ema.data.mul_(cfg.ema_decay).add_(p.data, alpha=1.0 - cfg.ema_decay)

            ep_loss += float(loss.item()) * cfg.grad_accum_steps
            if is_main:
                pbar.set_postfix(loss=f"{ep_loss / (i+1):.4f}")

        avg_loss = ep_loss / max(len(train_loader), 1)
        losses.append(avg_loss)
        if is_main:
            print(
                f"[diffusion] epoch {ep+1}/{cfg.epochs_diffusion} done, avg_loss={avg_loss:.6f}",
                flush=True,
            )

    if is_main:
        ckpt_path = root / "model_store" / "diffusion_unet.pt"
        ema_path = root / "model_store" / "diffusion_unet_ema.pt"
        src_model = model.module if hasattr(model, "module") else model
        torch.save(src_model.state_dict(), ckpt_path)
        torch.save(ema.state_dict(), ema_path)

        save_loss_curve(losses, root / "figures" / "loss_diffusion.png")
        with (root / "outputs" / "train_history_diffusion.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss"])
            for i, v in enumerate(losses, start=1):
                writer.writerow([i, v])

    if distributed:
        torch.distributed.barrier()

    src_model = model.module if hasattr(model, "module") else model
    return {
        "model": "diffusion",
        "train_time_min": round(timer.elapsed_sec() / 60.0, 3),
        "params_m": round(sum(p.numel() for p in src_model.parameters()) / 1e6, 3),
        "checkpoint": str(root / "model_store" / "diffusion_unet_ema.pt"),
    }


@torch.no_grad()
def sample_diffusion(
    model: DiffusionUNet,
    num_samples: int,
    class_ids: torch.Tensor,
    image_size: int,
    device: torch.device,
    sample_steps: int,
) -> torch.Tensor:
    model.eval()
    full_steps = int(getattr(model, "diffusion_steps", 500))
    betas, alphas, alpha_bar = _make_ddpm_schedules(full_steps, device)

    x = torch.randn(num_samples, 3, image_size, image_size, device=device)
    for i in reversed(range(full_steps)):
        t = torch.full((num_samples,), (i + 1) / full_steps, device=device)
        eps = model(x, t, class_ids)
        a = alphas[i]
        a_bar_i = alpha_bar[i]
        b = betas[i]
        noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
        x = (x - ((1 - a) / torch.sqrt(1 - a_bar_i)) * eps) / torch.sqrt(a) + torch.sqrt(b) * noise

    return denorm(x)


def save_quick_preview(model: DiffusionUNet, cfg: CompareConfig, root: Path, device: torch.device) -> None:
    class_ids = torch.tensor([3, 5, 1] * 12, device=device)[: cfg.vis_per_model]
    imgs = sample_diffusion(model, class_ids.size(0), class_ids, cfg.image_size, device, cfg.diffusion_sample_steps)
    save_image_grid(imgs, root / "figures" / "diffusion_curated_ultra.png", nrow=6, title="Diffusion Samples")
