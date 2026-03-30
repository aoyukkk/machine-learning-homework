from __future__ import annotations

import torch
import torch.nn as nn

from .common_unet import TinyClassUNet


class DiffusionUNet(nn.Module):
    def __init__(self, num_classes: int, base: int = 64):
        super().__init__()
        self.net = TinyClassUNet(in_ch=3, out_ch=3, num_classes=num_classes, base=base)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(x_t, t, y)
