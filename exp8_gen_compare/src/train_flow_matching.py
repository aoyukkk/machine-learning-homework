from __future__ import annotations

from pathlib import Path
import csv
import sys

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
from models.flow_unet import FlowMatchingUNet
from utils import Timer, save_image_grid, save_loss_curve, to_device


def train_flow_matching(
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

    base_model = FlowMatchingUNet(num_classes=cfg.num_classes, base=cfg.unet_base).to(device)
    model = base_model
    if distributed:
        model = nn.parallel.DistributedDataParallel(base_model, device_ids=[ctx["local_rank"]])
    elif cfg.use_data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(base_model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=cfg.amp and device.type == "cuda", **_AMP_SCALER_ARGS)

    losses = []
    timer = Timer()
    model.train()
    for ep in range(cfg.epochs_flow):
        if train_sampler is not None:
            train_sampler.set_epoch(ep)

        ep_loss = 0.0
        pbar = tqdm(
            train_loader,
            desc=f"[flow] epoch {ep+1}/{cfg.epochs_flow}",
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            disable=not is_main,
        )
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(pbar):
            x0, y = to_device(batch, device)
            z = torch.randn_like(x0)
            t = torch.rand(x0.size(0), device=device)
            t_view = t[:, None, None, None]
            x_t = (1.0 - t_view) * x0 + t_view * z
            target_v = z - x0

            with autocast(enabled=cfg.amp and device.type == "cuda", **_AMP_AUTOCAST_ARGS):
                pred_v = model(x_t, t, y)
                loss = F.mse_loss(pred_v, target_v) / cfg.grad_accum_steps

            scaler.scale(loss).backward()
            if (i + 1) % cfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            ep_loss += float(loss.item()) * cfg.grad_accum_steps
            if is_main:
                pbar.set_postfix(loss=f"{ep_loss / (i+1):.4f}")

        avg_loss = ep_loss / max(len(train_loader), 1)
        losses.append(avg_loss)
        if is_main:
            print(
                f"[flow] epoch {ep+1}/{cfg.epochs_flow} done, avg_loss={avg_loss:.6f}",
                flush=True,
            )

    if is_main:
        ckpt_path = root / "model_store" / "flow_unet.pt"
        src_model = model.module if hasattr(model, "module") else model
        torch.save(src_model.state_dict(), ckpt_path)

        save_loss_curve(losses, root / "figures" / "loss_flow.png")
        with (root / "outputs" / "train_history_flow.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss"])
            for i, v in enumerate(losses, start=1):
                writer.writerow([i, v])

    if distributed:
        torch.distributed.barrier()

    src_model = model.module if hasattr(model, "module") else model
    return {
        "model": "flow_matching",
        "train_time_min": round(timer.elapsed_sec() / 60.0, 3),
        "params_m": round(sum(p.numel() for p in src_model.parameters()) / 1e6, 3),
        "checkpoint": str(root / "model_store" / "flow_unet.pt"),
    }


@torch.no_grad()
def sample_flow_matching(
    model: FlowMatchingUNet,
    num_samples: int,
    class_ids: torch.Tensor,
    image_size: int,
    device: torch.device,
    sample_steps: int,
) -> torch.Tensor:
    model.eval()
    sample_steps = max(int(sample_steps), 1)
    x = torch.randn(num_samples, 3, image_size, image_size, device=device)
    dt = 1.0 / sample_steps
    for i in range(sample_steps):
        # Reverse-time ODE integration: t=1 (noise) -> t=0 (data).
        t_val = 1.0 - (i + 0.5) / sample_steps
        t = torch.full((num_samples,), t_val, device=device)
        v = model(x, t, class_ids)
        x = x - dt * v
    return denorm(x)


def save_quick_preview(model: FlowMatchingUNet, cfg: CompareConfig, root: Path, device: torch.device) -> None:
    class_ids = torch.tensor([3, 5, 1] * 12, device=device)[: cfg.vis_per_model]
    imgs = sample_flow_matching(model, class_ids.size(0), class_ids, cfg.image_size, device, cfg.flow_sample_steps)
    save_image_grid(imgs, root / "figures" / "flow_curated_ultra.png", nrow=6, title="Flow Matching Samples")
