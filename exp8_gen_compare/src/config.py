from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass
class CompareConfig:
    seed: int = 42
    image_size: int = 32
    num_classes: int = 10
    batch_size: int = 64
    grad_accum_steps: int = 1
    num_workers: int = 8
    log_interval_steps: int = 50
    epochs_diffusion: int = 60
    epochs_flow: int = 60
    epochs_autoregressive: int = 35
    lr: float = 2e-4
    weight_decay: float = 1e-4
    ema_decay: float = 0.999
    amp: bool = True
    diffusion_steps: int = 500
    diffusion_sample_steps: int = 220
    flow_sample_steps: int = 180
    fid_num_gen: int = 5000
    fid_batch_size: int = 64
    ar_num_bins: int = 16
    ar_refine_steps: int = 20
    ar_channels: int = 192
    ar_depth: int = 10
    unet_base: int = 128
    use_data_parallel: bool = True
    max_train_samples: int | None = None
    max_test_samples: int | None = None
    vis_per_model: int = 36
    vis_target_labels: tuple[int, int, int] = (3, 5, 1)  # cat, dog, automobile

    @staticmethod
    def from_profile(profile: str) -> "CompareConfig":
        cfg = CompareConfig()

        if profile == "multi_gpu_8x24g":
            # ---- 8×RTX 4090 服务器档位 ----
            # 每卡 batch=64, 8 卡有效 batch=512
            cfg.batch_size = 64
            cfg.grad_accum_steps = 1
            cfg.num_workers = 8
            cfg.fid_num_gen = 10000
            cfg.fid_batch_size = 128
            cfg.diffusion_sample_steps = 500
            cfg.flow_sample_steps = 400
            cfg.unet_base = 192
            cfg.ar_channels = 256
            cfg.ar_depth = 14
            cfg.ar_refine_steps = 30
            cfg.epochs_diffusion = 200
            cfg.epochs_flow = 200
            cfg.epochs_autoregressive = 80
            cfg.lr = 5e-4  # 线性 scaling: 2e-4 * (512/64) ≈ 5e-4 (capped)
            cfg.vis_per_model = 36

        elif profile == "gpu_hq_24g":
            cfg.batch_size = 96
            cfg.grad_accum_steps = 1
            cfg.fid_num_gen = 6000
            cfg.diffusion_sample_steps = 250
            cfg.flow_sample_steps = 200
            cfg.unet_base = 160
            cfg.ar_channels = 224
            cfg.ar_depth = 12
            cfg.epochs_diffusion = 70
            cfg.epochs_flow = 70
            cfg.epochs_autoregressive = 40

        elif profile == "gpu_hq_12g":
            cfg.batch_size = 32
            cfg.grad_accum_steps = 2
            cfg.fid_num_gen = 1200
            cfg.diffusion_sample_steps = 140
            cfg.flow_sample_steps = 120
            cfg.unet_base = 128
            cfg.ar_channels = 192
            cfg.ar_depth = 10
            cfg.epochs_diffusion = 12
            cfg.epochs_flow = 12
            cfg.epochs_autoregressive = 10
            cfg.max_train_samples = 30000
            cfg.max_test_samples = 3000

        elif profile == "cpu_fallback":
            cfg.batch_size = 8
            cfg.grad_accum_steps = 4
            cfg.num_workers = 2
            cfg.epochs_diffusion = 6
            cfg.epochs_flow = 6
            cfg.epochs_autoregressive = 3
            cfg.fid_num_gen = 300
            cfg.diffusion_sample_steps = 30
            cfg.flow_sample_steps = 25
            cfg.vis_per_model = 18
            cfg.use_data_parallel = False
            cfg.max_train_samples = 5000
            cfg.max_test_samples = 1000

        return cfg

    def to_dict(self) -> dict:
        """转为可 JSON 序列化的 dict（tuple → list）。"""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    def dump_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
