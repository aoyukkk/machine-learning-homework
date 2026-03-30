from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type: str, in_channels: int, out_channels: int, kernel_size: int, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        assert mask_type in ("A", "B")
        self.register_buffer("mask", torch.ones_like(self.weight))

        k_h, k_w = self.weight.shape[-2:]
        yc, xc = k_h // 2, k_w // 2
        self.mask[:, :, yc + 1 :, :] = 0
        self.mask[:, :, yc, xc + (1 if mask_type == "B" else 0) :] = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight * self.mask
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            MaskedConv2d("B", ch, ch, kernel_size=3, padding=1),
            nn.ReLU(),
            MaskedConv2d("B", ch, ch, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ClassConditionalPixelCNN(nn.Module):
    def __init__(self, num_classes: int, num_bins: int = 16, ch: int = 128, depth: int = 7):
        super().__init__()
        self.num_bins = num_bins
        self.in_proj = MaskedConv2d("A", 3, ch, kernel_size=7, padding=3)
        self.blocks = nn.Sequential(*[ResidualBlock(ch) for _ in range(depth)])
        self.cls_emb = nn.Embedding(num_classes, ch)
        self.out = nn.Sequential(
            nn.ReLU(),
            MaskedConv2d("B", ch, ch, kernel_size=1),
            nn.ReLU(),
            MaskedConv2d("B", ch, 3 * num_bins, kernel_size=1),
        )

    def forward(self, x_q: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x_q in [0, num_bins-1], shape [B,3,H,W]
        x = x_q.float() / max(self.num_bins - 1, 1)
        h = self.in_proj(x)
        h = self.blocks(h)
        h = h + self.cls_emb(y)[:, :, None, None]
        logits = self.out(h)
        b, _, hh, ww = logits.shape
        logits = logits.view(b, 3, self.num_bins, hh, ww)
        return logits

    def sample(self, y: torch.Tensor, shape: tuple[int, int], device: torch.device) -> torch.Tensor:
        b = y.size(0)
        h, w = shape
        x_q = torch.zeros((b, 3, h, w), device=device, dtype=torch.long)
        for i in range(h):
            for j in range(w):
                for c in range(3):
                    logits = self.forward(x_q, y)[:, c, :, i, j]
                    probs = torch.softmax(logits, dim=1)
                    x_q[:, c, i, j] = torch.multinomial(probs, num_samples=1).squeeze(1)
        return x_q
