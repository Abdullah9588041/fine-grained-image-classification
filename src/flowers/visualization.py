"""Figure helpers: training curves, confusion subsets, Grad-CAM grids."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_training_curves(history: Dict[str, list], path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ep = np.arange(1, len(history["train_loss"]) + 1)
    axes[0].plot(ep, history["train_loss"], label="train loss")
    axes[0].plot(ep, history["val_loss"], label="val loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(ep, np.array(history["val_acc"]) * 100, color="green")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("val accuracy (%)")
    axes[1].grid(alpha=0.3)
    fig.suptitle("Training curves")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_subset(
    cm: np.ndarray,
    class_ids: Sequence[int],
    path: str | Path,
    title: str = "Confusion matrix (subset)",
) -> None:
    sub = cm[np.ix_(list(class_ids), list(class_ids))]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sub, cmap="Blues")
    ax.set_xticks(range(len(class_ids)))
    ax.set_yticks(range(len(class_ids)))
    ax.set_xticklabels([f"c{c}" for c in class_ids], rotation=45, ha="right")
    ax.set_yticklabels([f"c{c}" for c in class_ids])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_gradcam_grid(
    panels: List[tuple[Image.Image, Image.Image, str]], path: str | Path, cols: int = 3
) -> None:
    """panels: list of (original, overlay, caption)."""
    rows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols * 2, figsize=(4 * cols * 2 / 2, 3 * rows))
    axes = np.atleast_2d(axes)
    for i, (orig, over, cap) in enumerate(panels):
        r, c = divmod(i, cols)
        axes[r, 2 * c].imshow(orig)
        axes[r, 2 * c].set_title(f"{cap} (orig)")
        axes[r, 2 * c].axis("off")
        axes[r, 2 * c + 1].imshow(over)
        axes[r, 2 * c + 1].set_title("Grad-CAM")
        axes[r, 2 * c + 1].axis("off")
    for ax in axes.ravel()[len(panels) * 2 :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_augmentation_samples(
    images: List[Image.Image], path: str | Path, title: str = "Augmentation samples"
) -> None:
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    for ax, img in zip(np.atleast_1d(axes), images):
        ax.imshow(img)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
