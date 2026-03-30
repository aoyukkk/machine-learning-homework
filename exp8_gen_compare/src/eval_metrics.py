from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
import torch
import torch.nn as nn
from scipy import linalg
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.transforms import Resize


class InceptionFeatureExtractor(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
        model.fc = nn.Identity()
        model.eval()
        self.model = model
        self.resize = Resize((299, 299), antialias=True)
        self.to(device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resize(x)
        x = x.clamp(0.0, 1.0)
        x = (x - self.mean) / self.std
        feat = self.model(x)
        if isinstance(feat, tuple):
            feat = feat[0]
        return feat


@torch.no_grad()
def collect_features(dataloader, extractor: InceptionFeatureExtractor, device: torch.device, limit: int | None = None) -> np.ndarray:
    feats = []
    seen = 0
    for x, _ in dataloader:
        x = x.to(device)
        x = (x * 0.5 + 0.5).clamp(0.0, 1.0)
        f = extractor(x).cpu().numpy()
        feats.append(f)
        seen += x.size(0)
        if limit is not None and seen >= limit:
            break
    feat = np.concatenate(feats, axis=0)
    if limit is not None:
        feat = feat[:limit]
    return feat


def calc_fid(feat1: np.ndarray, feat2: np.ndarray) -> float:
    mu1, sigma1 = np.mean(feat1, axis=0), np.cov(feat1, rowvar=False)
    mu2, sigma2 = np.mean(feat2, axis=0), np.cov(feat2, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def write_metrics_table(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
