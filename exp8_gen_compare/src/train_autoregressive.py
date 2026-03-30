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
from models.pixelcnn import ClassConditionalPixelCNN
from utils import Timer, save_image_grid, save_loss_curve, to_device


def _quantize(x: torch.Tensor, bins: int) -> torch.Tensor:
    x = (x * 0.5 + 0.5).nan_to_num(0.0).clamp(0.0, 1.0)
    return torch.clamp((x * (bins - 1)).round().long(), 0, bins - 1)


def _dequantize(x_q: torch.Tensor, bins: int) -> torch.Tensor:
    x = x_q.float() / max(bins - 1, 1)
    return x.nan_to_num(0.0).clamp(0.0, 1.0)


def train_autoregressive(
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

    base_model = ClassConditionalPixelCNN(
        num_classes=cfg.num_classes,
        num_bins=cfg.ar_num_bins,
        ch=cfg.ar_channels,
        depth=cfg.ar_depth,
    ).to(device)
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
    for ep in range(cfg.epochs_autoregressive):
        if train_sampler is not None:
            train_sampler.set_epoch(ep)

        ep_loss = 0.0
        pbar = tqdm(
            train_loader,
            desc=f"[autoregressive] epoch {ep+1}/{cfg.epochs_autoregressive}",
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            disable=not is_main,
        )
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(pbar):
            x, y = to_device(batch, device)
            x_q = _quantize(x, cfg.ar_num_bins)
            with autocast(enabled=cfg.amp and device.type == "cuda", **_AMP_AUTOCAST_ARGS):
                logits = model(x_q, y)
                loss = (
                    F.cross_entropy(logits[:, 0], x_q[:, 0])
                    + F.cross_entropy(logits[:, 1], x_q[:, 1])
                    + F.cross_entropy(logits[:, 2], x_q[:, 2])
                ) / 3.0
                loss = loss / cfg.grad_accum_steps

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
                f"[autoregressive] epoch {ep+1}/{cfg.epochs_autoregressive} done, avg_loss={avg_loss:.6f}",
                flush=True,
            )

    if is_main:
        ckpt_path = root / "model_store" / "autoregressive_pixelcnn.pt"
        src_model = model.module if hasattr(model, "module") else model
        torch.save(src_model.state_dict(), ckpt_path)

        save_loss_curve(losses, root / "figures" / "loss_autoregressive.png")
        with (root / "outputs" / "train_history_autoregressive.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss"])
            for i, v in enumerate(losses, start=1):
                writer.writerow([i, v])

    if distributed:
        torch.distributed.barrier()

    src_model = model.module if hasattr(model, "module") else model
    return {
        "model": "autoregressive",
        "train_time_min": round(timer.elapsed_sec() / 60.0, 3),
        "params_m": round(sum(p.numel() for p in src_model.parameters()) / 1e6, 3),
        "checkpoint": str(root / "model_store" / "autoregressive_pixelcnn.pt"),
    }


@torch.no_grad()
def sample_autoregressive(
    model: ClassConditionalPixelCNN,
    num_samples: int,
    class_ids: torch.Tensor,
    image_size: int,
    device: torch.device,
    refine_steps: int = 8,
) -> torch.Tensor:
    model.eval()
    num_bins = model.num_bins if hasattr(model, "num_bins") else 16
    # unwrap DDP if needed
    src_model = model.module if hasattr(model, "module") else model
    # optimize generation by doing 3 channels at once to save 3x forward passes
    x_q = torch.zeros((num_samples, 3, image_size, image_size), device=device, dtype=torch.long)
    for i in range(image_size):
        for j in range(image_size):
            logits = src_model(x_q, class_ids)[:, :, :, i, j] # (B, 3, bins)
            for c in range(3):
                probs = torch.softmax(logits[:, c], dim=1)
                x_q[:, c, i, j] = torch.multinomial(probs, num_samples=1).squeeze(1)
    return _dequantize(x_q, num_bins)

def save_quick_preview(model: ClassConditionalPixelCNN, cfg: CompareConfig, root: Path, device: torch.device) -> None:
    class_ids = torch.tensor([3, 5, 1] * 12, device=device)[: cfg.vis_per_model]
    imgs = sample_autoregressive(model, class_ids.size(0), class_ids, cfg.image_size, device, cfg.ar_refine_steps)
    save_image_grid(imgs, root / "figures" / "autoregressive_curated_ultra.png", nrow=6, title="Autoregressive Samples")
