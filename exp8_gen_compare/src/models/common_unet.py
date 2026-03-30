from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, out_ch)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        h = h + self.emb(emb)[:, :, None, None]
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.q = nn.Conv2d(ch, ch, 1)
        self.k = nn.Conv2d(ch, ch, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        y = self.norm(x)
        q = self.q(y).view(b, c, h * w).transpose(1, 2)
        k = self.k(y).view(b, c, h * w)
        attn = torch.softmax((q @ k) / (c ** 0.5), dim=-1)
        v = self.v(y).view(b, c, h * w).transpose(1, 2)
        out = (attn @ v).transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj(out)


class TinyClassUNet(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_classes: int, base: int = 64):
        super().__init__()
        emb_dim = base * 4
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(base),
            nn.Linear(base, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.cls_emb = nn.Embedding(num_classes, emb_dim)

        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)

        self.down1 = ResBlock(base, base, emb_dim)
        self.down2 = ResBlock(base, base * 2, emb_dim)
        self.pool1 = nn.Conv2d(base * 2, base * 2, 4, stride=2, padding=1)

        self.down3 = ResBlock(base * 2, base * 2, emb_dim)
        self.down4 = ResBlock(base * 2, base * 4, emb_dim)
        self.pool2 = nn.Conv2d(base * 4, base * 4, 4, stride=2, padding=1)

        self.mid1 = ResBlock(base * 4, base * 4, emb_dim)
        self.mid_attn = SelfAttention2d(base * 4)
        self.mid2 = ResBlock(base * 4, base * 4, emb_dim)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1)
        self.up_res2 = ResBlock(base * 2 + base * 4, base * 2, emb_dim)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.up_res1 = ResBlock(base + base * 2, base, emb_dim)

        self.out = nn.Sequential(
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.Conv2d(base, out_ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        emb = self.time_emb(t) + self.cls_emb(y)

        x0 = self.in_conv(x)
        d1 = self.down1(x0, emb)
        d2 = self.down2(d1, emb)
        p1 = self.pool1(d2)

        d3 = self.down3(p1, emb)
        d4 = self.down4(d3, emb)
        p2 = self.pool2(d4)

        m = self.mid1(p2, emb)
        m = self.mid_attn(m)
        m = self.mid2(m, emb)

        u2 = self.up2(m)
        u2 = self.up_res2(torch.cat([u2, d4], dim=1), emb)

        u1 = self.up1(u2)
        u1 = self.up_res1(torch.cat([u1, d2], dim=1), emb)

        return self.out(u1)
