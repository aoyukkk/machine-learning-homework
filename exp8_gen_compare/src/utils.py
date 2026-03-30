from __future__ import annotations

from pathlib import Path
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = batch
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def save_image_grid(images: torch.Tensor, path: Path, nrow: int = 6, title: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = images.detach().cpu().nan_to_num(0.0).clamp(0.0, 1.0)
    grid = make_grid(images, nrow=nrow)
    np_grid = np.transpose(grid.numpy(), (1, 2, 0))
    plt.figure(figsize=(8.5, 8.5))
    plt.imshow(np_grid)
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_loss_curve(losses: list[float], path: Path, ylabel: str = "loss") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.2, 4.5))
    x = np.arange(1, len(losses) + 1)
    plt.plot(x, losses, linewidth=2.1)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


class Timer:
    def __init__(self) -> None:
        self.start = time.perf_counter()

    def elapsed_sec(self) -> float:
        return time.perf_counter() - self.start
