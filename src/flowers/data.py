"""Oxford 102 Flowers dataset: loading, official splits, and augmentation pipelines."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class DataConfig:
    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 2
    seed: int = 42


def load_official_splits(raw_dir: str | Path) -> Dict[str, np.ndarray]:
    """Return the official train/val/test index arrays (0-based)."""
    mat = sio.loadmat(str(Path(raw_dir) / "setid.mat"))
    return {
        "train": np.asarray(mat["trnid"][0], dtype=np.int64) - 1,
        "val": np.asarray(mat["valid"][0], dtype=np.int64) - 1,
        "test": np.asarray(mat["tstid"][0], dtype=np.int64) - 1,
    }


def load_labels(raw_dir: str | Path) -> np.ndarray:
    """Return integer class labels 0..101 for all 8189 images."""
    mat = sio.loadmat(str(Path(raw_dir) / "imagelabels.mat"))
    return (np.asarray(mat["labels"][0], dtype=np.int64) - 1).copy()


def image_path(raw_dir: str | Path, index: int) -> Path:
    """Image files are named image_00001.jpg ... image_08189.jpg (1-based)."""
    return Path(raw_dir) / "jpg" / f"image_{index + 1:05d}.jpg"


def train_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def eval_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class FlowersDataset(Dataset):
    """Oxford 102 Flowers subset defined by explicit indices."""

    def __init__(self, raw_dir: str | Path, indices: np.ndarray, transform=None):
        self.raw_dir = Path(raw_dir)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = load_labels(self.raw_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        idx = int(self.indices[i])
        img = Image.open(image_path(self.raw_dir, idx)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx])


def build_dataloaders(
    raw_dir: str | Path, cfg: DataConfig, augment: bool = True
) -> Dict[str, DataLoader]:
    """Build train/val/test loaders using the official splits."""
    splits = load_official_splits(raw_dir)
    train_tf = train_transforms(cfg.image_size) if augment else eval_transforms(cfg.image_size)
    eval_tf = eval_transforms(cfg.image_size)
    gen = torch.Generator().manual_seed(cfg.seed)

    def _loader(name: str, shuffle: bool, tf) -> DataLoader:
        ds = FlowersDataset(raw_dir, splits[name], transform=tf)
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=False,
            generator=gen if shuffle else None,
            drop_last=False,
        )

    return {
        "train": _loader("train", True, train_tf),
        "val": _loader("val", False, eval_tf),
        "test": _loader("test", False, eval_tf),
    }
