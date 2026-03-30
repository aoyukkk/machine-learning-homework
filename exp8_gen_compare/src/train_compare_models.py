from __future__ import annotations

import argparse
import os
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from config import CompareConfig
from data import make_dataloaders
from eval_metrics import InceptionFeatureExtractor, calc_fid, collect_features, write_metrics_table
from models.diffusion_unet import DiffusionUNet
from models.flow_unet import FlowMatchingUNet
from models.pixelcnn import ClassConditionalPixelCNN
from system_probe import probe_system
from train_autoregressive import sample_autoregressive, train_autoregressive
from train_diffusion import sample_diffusion, train_diffusion
from train_flow_matching import sample_flow_matching, train_flow_matching
from utils import save_image_grid, set_seed


def _ensure_dirs(root: Path) -> None:
    for name in ["outputs", "figures", "configs", "model_store"]:
        (root / name).mkdir(parents=True, exist_ok=True)


def _class_ids_for_sampling(num_samples: int, device: torch.device) -> torch.Tensor:
    labels = torch.tensor([3, 5, 1], device=device)
    reps = int(np.ceil(num_samples / 3))
    return labels.repeat(reps)[:num_samples]


def _save_target_panels(samples_by_model: dict[str, torch.Tensor], path: Path, per_class: int = 6) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 12.5))
    class_names = ["cat", "dog", "automobile"]

    for col, (model_name, imgs) in enumerate(samples_by_model.items()):
        imgs = imgs.nan_to_num(0.0).clamp(0.0, 1.0)
        groups = [imgs[0::3], imgs[1::3], imgs[2::3]]
        for row in range(3):
            subset = groups[row][:per_class]
            row_img = torch.cat([subset[i] for i in range(subset.size(0))], dim=2)
            axes[row, col].imshow(np.transpose(row_img.cpu().numpy(), (1, 2, 0)))
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(model_name, fontsize=14)
            if col == 0:
                axes[row, col].set_ylabel(class_names[row], fontsize=13)

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=240)
    plt.close()


def _save_readability_zoom(samples_by_model: dict[str, torch.Tensor], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for idx, (name, imgs) in enumerate(samples_by_model.items()):
        img = imgs[0].nan_to_num(0.0).clamp(0.0, 1.0)
        c, h, w = img.shape
        y0, y1 = h // 4, 3 * h // 4
        x0, x1 = w // 4, 3 * w // 4
        crop = img[:, y0:y1, x0:x1]
        zoom = torch.nn.functional.interpolate(
            crop.unsqueeze(0),
            scale_factor=4,
            mode="bicubic",
            align_corners=False,
        ).squeeze(0).nan_to_num(0.0).clamp(0.0, 1.0)
        axes[idx].imshow(np.transpose(zoom.cpu().numpy(), (1, 2, 0)))
        axes[idx].set_title(f"{name} 4x zoom")
        axes[idx].axis("off")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=240)
    plt.close()


def _save_comparison_grid(samples_by_model: dict[str, torch.Tensor], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for idx, (name, imgs) in enumerate(samples_by_model.items()):
        grid = imgs[:25].nan_to_num(0.0).clamp(0.0, 1.0)
        rows = []
        for i in range(5):
            rows.append(torch.cat([grid[i * 5 + j] for j in range(5)], dim=2))
        mosaic = torch.cat(rows, dim=1)
        axes[idx].imshow(np.transpose(mosaic.cpu().numpy(), (1, 2, 0)))
        axes[idx].set_title(name, fontsize=13)
        axes[idx].axis("off")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=240)
    plt.close()


def _save_fid_barplot(rows: list[dict[str, float | str]], path: Path) -> None:
    names = [str(r["model"]) for r in rows]
    vals = [float(r["fid"]) for r in rows]
    plt.figure(figsize=(7.2, 4.8))
    colors = ["#4E79A7", "#F28E2B", "#59A14F"]
    bars = plt.bar(names, vals, color=colors[:len(names)])
    plt.ylabel("FID (lower is better)")
    plt.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


@torch.no_grad()
def _sample_in_batches(
    model,
    sampler,
    class_ids: torch.Tensor,
    batch_size: int,
    image_size: int,
    device: torch.device,
    sample_steps: int | None = None,
    desc: str = "sampling",
) -> torch.Tensor:
    chunks = []
    n_batches = (class_ids.size(0) + batch_size - 1) // batch_size
    for i in tqdm(range(0, class_ids.size(0), batch_size), total=n_batches, desc=desc, dynamic_ncols=True):
        cid = class_ids[i : i + batch_size]
        if sample_steps is None:
            out = sampler(model, cid.size(0), cid, image_size, device)
        else:
            out = sampler(model, cid.size(0), cid, image_size, device, sample_steps)
        chunks.append(out.cpu())
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def _features_from_generated(
    images: torch.Tensor,
    extractor: InceptionFeatureExtractor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    feats = []
    for i in range(0, images.size(0), batch_size):
        x = images[i : i + batch_size].to(device)
        feats.append(extractor(x).cpu().numpy())
    return np.concatenate(feats, axis=0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare three generative model families.")
    parser.add_argument("--quick-test", action="store_true", help="Run a short smoke-test schedule.")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and only evaluate from checkpoints.")
    parser.add_argument("--profile", type=str, default=None,
                        help="Manual profile override: multi_gpu_8x24g | gpu_hq_24g | gpu_hq_12g | cpu_fallback")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs to use, e.g. '0,2,5'. "
                             "If not specified, uses all visible GPUs.")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Max number of GPUs to use (picks the first N available).")
    return parser.parse_args()


def _select_gpus(gpu_str: str | None, num_gpus: int | None) -> None:
    """在 CUDA 初始化前通过 CUDA_VISIBLE_DEVICES 限定可用 GPU。

    重要：此函数必须在任何 torch.cuda 调用之前执行。
    仅在非 torchrun 启动（即 WORLD_SIZE 未被设置）时生效。
    """
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        # torchrun 已经管理了 GPU 分配，不再覆盖
        return

    if gpu_str is not None:
        # 用户明确指定了 GPU ID，直接设置
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
        print(f"[exp8] 用户指定 GPU: {gpu_str}  (CUDA_VISIBLE_DEVICES={gpu_str})", flush=True)
    elif num_gpus is not None:
        # 用户只指定了数量，选取前 N 个
        # 注意：此时还没初始化 CUDA，用 nvidia-smi 探测
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True,
            )
            all_ids = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
            selected = all_ids[:num_gpus]
            val = ",".join(selected)
            os.environ["CUDA_VISIBLE_DEVICES"] = val
            print(f"[exp8] 限制使用前 {num_gpus} 张 GPU: {selected}  (CUDA_VISIBLE_DEVICES={val})", flush=True)
        except Exception as e:
            print(f"[exp8] ⚠ nvidia-smi 探测失败，使用所有 GPU: {e}", flush=True)


def _init_distributed() -> dict:
    """检测并初始化 DDP 分布式训练环境。"""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1

    if distributed:
        import torch.distributed as dist
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        rank = dist.get_rank()
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return {
        "distributed": distributed,
        "local_rank": local_rank,
        "rank": rank,
        "world_size": world_size,
        "is_main": rank == 0,
        "device": device,
    }


def main() -> None:
    args = _parse_args()

    # 强制启用 tqdm 进度条
    os.environ["FORCE_TQDM"] = "1"

    # GPU 选择（仅在非 torchrun 模式生效）
    _select_gpus(args.gpus, args.num_gpus)

    # DDP 初始化
    ctx = _init_distributed()
    distributed = ctx["distributed"]
    is_main = ctx["is_main"]
    device = ctx["device"]

    root = Path(__file__).resolve().parent.parent
    _ensure_dirs(root)

    # 选择配置档位
    if args.profile:
        cfg = CompareConfig.from_profile(args.profile)
        if is_main:
            print(f"[exp8] 使用手动指定 profile: {args.profile}", flush=True)
    else:
        sys_info = probe_system(root / "outputs")
        cfg = CompareConfig.from_profile(sys_info.profile)
        if is_main:
            print(f"[exp8] 自动检测 profile: {sys_info.profile} (GPU数: {sys_info.num_gpus}, 可用显存: {sys_info.free_mem_mb}MB)", flush=True)

    if args.quick_test:
        cfg.epochs_diffusion = 3
        cfg.epochs_flow = 3
        cfg.epochs_autoregressive = 3
        cfg.fid_num_gen = 60
        cfg.fid_batch_size = 20
        cfg.batch_size = min(cfg.batch_size, 16)
        cfg.unet_base = min(cfg.unet_base, 64)
        cfg.ar_channels = min(cfg.ar_channels, 128)
        cfg.ar_depth = min(cfg.ar_depth, 7)
        cfg.ar_refine_steps = 12
        cfg.max_train_samples = 12000
        cfg.max_test_samples = 1000
        cfg.diffusion_sample_steps = 80
        cfg.flow_sample_steps = 80
        cfg.vis_per_model = 18
        if is_main:
            print("[exp8] ⚡ Quick-test 模式: 仅用于冒烟验证，不建议直接用于最终报告", flush=True)

    if is_main:
        cfg.dump_json(root / "configs" / "exp8_config.json")

    set_seed(cfg.seed)

    # 数据加载
    train_loader, val_loader, test_loader, split, train_sampler = make_dataloaders(
        root_dir=root,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        outputs_dir=root / "outputs",
        max_train_samples=cfg.max_train_samples,
        max_test_samples=cfg.max_test_samples,
        distributed=distributed,
        rank=ctx["rank"],
        world_size=ctx["world_size"],
    )
    ctx["train_sampler"] = train_sampler

    if is_main:
        print(f"[exp8] 数据加载完成: train={len(train_loader.dataset)} samples, batch_size={cfg.batch_size}", flush=True)
        if distributed:
            print(f"[exp8] 分布式训练: {ctx['world_size']} GPUs, 有效 batch_size={cfg.batch_size * ctx['world_size']}", flush=True)

    train_stats = []
    total_stages = 7
    stage_bar = tqdm(
        total=total_stages,
        desc="[exp8] 整体进度",
        dynamic_ncols=True,
        leave=True,
        disable=not is_main,
        bar_format="{desc}: {bar} {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )

    if not args.skip_train:
        if is_main:
            print(f"\n{'='*60}", flush=True)
            print(f"[exp8] Stage 1/{total_stages}: 训练 Diffusion 模型", flush=True)
            print(f"{'='*60}", flush=True)
        train_stats.append(train_diffusion(cfg, train_loader, val_loader, root, device, ctx))
        stage_bar.update(1)

        if is_main:
            print(f"\n{'='*60}", flush=True)
            print(f"[exp8] Stage 2/{total_stages}: 训练 Flow Matching 模型", flush=True)
            print(f"{'='*60}", flush=True)
        train_stats.append(train_flow_matching(cfg, train_loader, val_loader, root, device, ctx))
        stage_bar.update(1)

        if is_main:
            print(f"\n{'='*60}", flush=True)
            print(f"[exp8] Stage 3/{total_stages}: 训练 Autoregressive 模型", flush=True)
            print(f"{'='*60}", flush=True)
        train_stats.append(train_autoregressive(cfg, train_loader, val_loader, root, device, ctx))
        stage_bar.update(1)
    else:
        train_stats = [
            {"model": "diffusion", "train_time_min": 0.0, "params_m": 0.0, "checkpoint": str(root / "model_store" / "diffusion_unet_ema.pt")},
            {"model": "flow_matching", "train_time_min": 0.0, "params_m": 0.0, "checkpoint": str(root / "model_store" / "flow_unet.pt")},
            {"model": "autoregressive", "train_time_min": 0.0, "params_m": 0.0, "checkpoint": str(root / "model_store" / "autoregressive_pixelcnn.pt")},
        ]
        stage_bar.update(3)

    # ---- 以下仅在 rank 0 上执行（评估 + 可视化 + 报告） ----
    if not is_main:
        stage_bar.update(total_stages - 3)
        stage_bar.close()
        if distributed:
            torch.distributed.destroy_process_group()
        return

    print(f"\n{'='*60}", flush=True)
    print(f"[exp8] Stage 4/{total_stages}: 加载 checkpoint 并构建模型", flush=True)
    print(f"{'='*60}", flush=True)

    eval_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _safe_load(path: Path):
        try:
            return torch.load(path, map_location=eval_device, weights_only=True)
        except TypeError:
            return torch.load(path, map_location=eval_device)

    diff = DiffusionUNet(cfg.num_classes, base=cfg.unet_base).to(eval_device)
    diff.diffusion_steps = cfg.diffusion_steps
    diff.load_state_dict(_safe_load(root / "model_store" / "diffusion_unet_ema.pt"))
    flow = FlowMatchingUNet(cfg.num_classes, base=cfg.unet_base).to(eval_device)
    flow.load_state_dict(_safe_load(root / "model_store" / "flow_unet.pt"))
    ar = ClassConditionalPixelCNN(
        cfg.num_classes, cfg.ar_num_bins, ch=cfg.ar_channels, depth=cfg.ar_depth,
    ).to(eval_device)
    ar.load_state_dict(_safe_load(root / "model_store" / "autoregressive_pixelcnn.pt"))
    stage_bar.update(1)

    num_gen = cfg.fid_num_gen
    class_ids = _class_ids_for_sampling(num_gen, eval_device)

    print(f"\n{'='*60}", flush=True)
    print(f"[exp8] Stage 5/{total_stages}: 从三个模型采样 {num_gen} 张图片", flush=True)
    print(f"{'='*60}", flush=True)
    gen_diff = _sample_in_batches(
        diff, sample_diffusion, class_ids,
        batch_size=min(cfg.fid_batch_size, 32), image_size=cfg.image_size,
        device=eval_device, sample_steps=cfg.diffusion_sample_steps,
        desc="  采样 Diffusion",
    )
    gen_flow = _sample_in_batches(
        flow, sample_flow_matching, class_ids,
        batch_size=min(cfg.fid_batch_size, 32), image_size=cfg.image_size,
        device=eval_device, sample_steps=cfg.flow_sample_steps,
        desc="  采样 Flow Matching",
    )
    gen_ar = _sample_in_batches(
        ar, sample_autoregressive, class_ids,
        batch_size=min(cfg.fid_batch_size, 16), image_size=cfg.image_size,
        device=eval_device, sample_steps=cfg.ar_refine_steps,
        desc="  采样 Autoregressive",
    )
    stage_bar.update(1)

    # 保存可视化
    save_image_grid(gen_diff[: cfg.vis_per_model], root / "figures" / "diffusion_curated_ultra.png", nrow=6)
    save_image_grid(gen_flow[: cfg.vis_per_model], root / "figures" / "flow_curated_ultra.png", nrow=6)
    save_image_grid(gen_ar[: cfg.vis_per_model], root / "figures" / "autoregressive_curated_ultra.png", nrow=6)
    samples_by_model = {
        "Diffusion": gen_diff,
        "Flow Matching": gen_flow,
        "Autoregressive": gen_ar,
    }
    _save_comparison_grid(samples_by_model, root / "figures" / "ultra_curated_comparison.png")
    _save_target_panels(samples_by_model, root / "figures" / "class_panels_cat_dog_car_hq.png")
    _save_readability_zoom(samples_by_model, root / "figures" / "readability_zoom_panels.png")

    print(f"\n{'='*60}", flush=True)
    print(f"[exp8] Stage 6/{total_stages}: FID 特征提取与指标计算", flush=True)
    print(f"{'='*60}", flush=True)
    extractor = InceptionFeatureExtractor(eval_device)
    real_feats = collect_features(test_loader, extractor, eval_device, limit=num_gen)
    diff_feats = _features_from_generated(gen_diff, extractor, eval_device, batch_size=cfg.fid_batch_size)
    flow_feats = _features_from_generated(gen_flow, extractor, eval_device, batch_size=cfg.fid_batch_size)
    ar_feats = _features_from_generated(gen_ar, extractor, eval_device, batch_size=cfg.fid_batch_size)

    metrics_rows = []
    for stat in train_stats:
        name = str(stat["model"])
        if name == "diffusion":
            fid = calc_fid(real_feats, diff_feats)
        elif name == "flow_matching":
            fid = calc_fid(real_feats, flow_feats)
        else:
            fid = calc_fid(real_feats, ar_feats)
        metrics_rows.append(
            {
                "model": name,
                "fid": round(fid, 4),
                "train_time_min": stat["train_time_min"],
                "params_m": stat["params_m"],
            }
        )

    write_metrics_table(root / "outputs" / "comparison_metrics.csv", metrics_rows)

    ranking = sorted(metrics_rows, key=lambda x: float(x["fid"]))
    with (root / "outputs" / "ranking.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "model", "fid", "train_time_min", "params_m"])
        writer.writeheader()
        for i, row in enumerate(ranking, start=1):
            writer.writerow({"rank": i, **row})

    _save_fid_barplot(metrics_rows, root / "figures" / "fid_barplot.png")
    stage_bar.update(1)

    # 摘要
    cfg_dict = cfg.to_dict()
    with (root / "outputs" / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "profile": args.profile or "auto",
                "distributed": distributed,
                "world_size": ctx["world_size"],
                "device": str(device),
                "gpus_requested": args.gpus,
                "config": cfg_dict,
                "metrics": metrics_rows,
                "quick_test": bool(args.quick_test),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n{'='*60}", flush=True)
    print(f"[exp8] Stage 7/{total_stages}: 生成报告", flush=True)
    print(f"{'='*60}", flush=True)

    # 调用报告生成
    try:
        from generate_report import generate_latex_report
        generate_latex_report(root)
        print("[exp8] ✅ LaTeX 报告已生成: report8.tex", flush=True)
    except Exception as e:
        print(f"[exp8] ⚠ 报告生成失败: {e}", flush=True)

    stage_bar.update(1)
    stage_bar.close()

    print(f"\n{'='*60}", flush=True)
    print("✅ Exp8 对比实验完成！结果保存在 outputs/ 和 figures/ 目录。", flush=True)
    print(f"{'='*60}", flush=True)
    print("\n📊 FID 排名:", flush=True)
    for i, row in enumerate(ranking, start=1):
        emoji = "🥇" if i == 1 else ("🥈" if i == 2 else "🥉")
        print(f"  {emoji} #{i} {row['model']}: FID={row['fid']:.4f}", flush=True)

    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
