from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import torch


@dataclass
class SystemInfo:
    has_cuda: bool
    num_gpus: int
    free_mem_mb: int
    profile: str
    nvidia_smi_text: str


def _query_nvidia_smi() -> str:
    try:
        result = subprocess.run(["nvidia-smi"], check=True, capture_output=True, text=True)
        return result.stdout
    except Exception as exc:
        return f"nvidia-smi unavailable: {exc}\n"


def choose_profile(smi_text: str) -> str:
    if not torch.cuda.is_available():
        return "cpu_fallback"

    num_gpus = torch.cuda.device_count()

    # Parse all "used / total" memory lines and pick the best free memory estimate.
    mem_matches = re.findall(r"(\d+)MiB\s*/\s*(\d+)MiB", smi_text)
    best_free = 0
    for used_str, total_str in mem_matches:
        used = int(used_str)
        total = int(total_str)
        free = max(total - used, 0)
        best_free = max(best_free, free)

    # 多卡 (>= 4 张, 每张 >= 18 GB) → 多卡 profile
    if num_gpus >= 4 and best_free >= 18000:
        return "multi_gpu_8x24g"
    if best_free >= 18000:
        return "gpu_hq_24g"
    if best_free >= 9000:
        return "gpu_hq_12g"
    return "cpu_fallback"


def probe_system(outputs_dir: Path) -> SystemInfo:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    smi_text = _query_nvidia_smi()
    profile = choose_profile(smi_text)

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    free_mem_mb = 0
    mem_matches = re.findall(r"(\d+)MiB\s*/\s*(\d+)MiB", smi_text)
    for used_str, total_str in mem_matches:
        used = int(used_str)
        total = int(total_str)
        free_mem_mb = max(free_mem_mb, max(total - used, 0))

    (outputs_dir / "nvidia_smi.txt").write_text(smi_text, encoding="utf-8")
    (outputs_dir / "system_profile.txt").write_text(
        f"profile={profile}\nfree_mem_mb={free_mem_mb}\nnum_gpus={num_gpus}\nhas_cuda={torch.cuda.is_available()}\n",
        encoding="utf-8",
    )

    return SystemInfo(
        has_cuda=torch.cuda.is_available(),
        num_gpus=num_gpus,
        free_mem_mb=free_mem_mb,
        profile=profile,
        nvidia_smi_text=smi_text,
    )
