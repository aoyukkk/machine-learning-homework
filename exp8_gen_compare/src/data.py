from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms


def _img_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


class LabelMappedDataset(Dataset):
    def __init__(self, base_dataset: Dataset, label_map: dict[int, int] | None = None):
        self.base_dataset = base_dataset
        self.label_map = label_map or {}

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        x, y = self.base_dataset[idx]
        y = int(y)
        y = self.label_map.get(y, y)
        return x, y


def _build_merged_dataset(data_root: Path, image_size: int, train_transform: bool) -> Dataset:
    tfm = _img_transforms(image_size, train=train_transform)

    # Unified label order follows CIFAR-10:
    # 0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer,
    # 5 dog, 6 frog, 7 horse, 8 ship, 9 truck.
    # STL-10 class-7 monkey → 6 (semantically close animal proxy).
    stl_to_unified = {0: 0, 1: 2, 2: 1, 3: 3, 4: 4, 5: 5, 6: 7, 7: 6, 8: 8, 9: 9}

    stl_train = datasets.STL10(root=str(data_root), split="train", download=True, transform=tfm)
    stl_test = datasets.STL10(root=str(data_root), split="test", download=True, transform=tfm)
    cifar_train = datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=tfm)
    cifar_test = datasets.CIFAR10(root=str(data_root), train=False, download=True, transform=tfm)

    merged = ConcatDataset(
        [
            LabelMappedDataset(cifar_train),
            LabelMappedDataset(cifar_test),
            LabelMappedDataset(stl_train, stl_to_unified),
            LabelMappedDataset(stl_test, stl_to_unified),
        ]
    )
    return merged


def make_dataloaders(
    root_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    outputs_dir: Path,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, list[int]], DistributedSampler | None]:
    """构建训练/验证/测试 DataLoader，支持 DDP 分布式采样。"""
    data_root = root_dir / "model_store"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    train_all = _build_merged_dataset(data_root, image_size, train_transform=True)
    eval_all = _build_merged_dataset(data_root, image_size, train_transform=False)

    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(train_all))
    rng.shuffle(all_indices)

    n_total = len(all_indices)
    n_train = int(0.9 * n_total)
    n_val = int(0.05 * n_total)

    train_indices = all_indices[:n_train].tolist()
    val_indices = all_indices[n_train : n_train + n_val].tolist()
    test_indices = all_indices[n_train + n_val :].tolist()

    if max_train_samples is not None:
        train_indices = train_indices[:max_train_samples]
    if max_test_samples is not None:
        test_indices = test_indices[:max_test_samples]

    split = {
        "dataset": "merged_cifar10_stl10",
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }
    if rank == 0:
        (outputs_dir / "split_indices.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    train_subset = Subset(train_all, train_indices)
    val_subset = Subset(eval_all, val_indices)
    test_subset = Subset(eval_all, test_indices)

    generator = torch.Generator().manual_seed(seed)

    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_subset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed,
        )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        generator=generator if train_sampler is None else None,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader, split, train_sampler


def denorm(x: torch.Tensor) -> torch.Tensor:
    return (x * 0.5 + 0.5).nan_to_num(0.0).clamp(0.0, 1.0)


def filter_classes(x: torch.Tensor, y: torch.Tensor, labels: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.zeros_like(y, dtype=torch.bool)
    for lb in labels:
        mask = mask | (y == lb)
    return x[mask], y[mask]
