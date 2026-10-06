"""Evaluation: top-1/top-5 accuracy, per-class metrics, confusion analysis."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def predict_all(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        out = model(x.to(device))
        ys.append(y.numpy())
        ps.append(out.argmax(1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


@torch.no_grad()
def evaluate_test(
    model: nn.Module, loader: DataLoader, device: torch.device, k: int = 5
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Single test pass returning (y_true, y_pred, top-k accuracy)."""
    model.eval()
    ys, ps, topk_correct, n = [], [], 0, 0
    for x, y in loader:
        out = model(x.to(device))
        topk = out.topk(k, dim=1).indices.cpu()
        yc = y.numpy()
        topk_correct += sum(int(yc[i]) in topk[i].tolist() for i in range(len(yc)))
        ys.append(yc)
        ps.append(out.argmax(1).cpu().numpy())
        n += len(yc)
    return np.concatenate(ys), np.concatenate(ps), topk_correct / n


@torch.no_grad()
def topk_accuracy(
    model: nn.Module, loader: DataLoader, device: torch.device, k: int = 5
) -> float:
    _, _, acc = evaluate_test(model, loader, device, k=k)
    return acc


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class_prf(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> Dict[str, np.ndarray]:
    cm = confusion_matrix(y_true, y_pred, num_classes)
    tp = np.diag(cm).astype(float)
    precision = tp / np.maximum(cm.sum(axis=0), 1)
    recall = tp / np.maximum(cm.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1, "support": cm.sum(axis=1)}


def top_confused_pairs(
    cm: np.ndarray, top_n: int = 10
) -> List[Tuple[int, int, int]]:
    """(true, predicted, count) for the most frequent off-diagonal confusions."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = []
    flat = np.argsort(off.ravel())[::-1][: top_n * 3]
    seen = set()
    for idx in flat:
        t, p = divmod(int(idx), cm.shape[1])
        if off[t, p] == 0 or (t, p) in seen:
            continue
        seen.add((t, p))
        pairs.append((t, p, int(off[t, p])))
        if len(pairs) >= top_n:
            break
    return pairs


def summarize(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 102
) -> Dict:
    cm = confusion_matrix(y_true, y_pred, num_classes)
    prf = per_class_prf(y_true, y_pred, num_classes)
    acc = float((y_true == y_pred).mean())
    return {
        "top1_accuracy": acc,
        "macro_f1": float(prf["f1"].mean()),
        "worst_classes": [
            (int(i), float(prf["f1"][i]), int(prf["support"][i]))
            for i in np.argsort(prf["f1"])[:10]
        ],
        "best_classes": [
            (int(i), float(prf["f1"][i]), int(prf["support"][i]))
            for i in np.argsort(prf["f1"])[::-1][:10]
        ],
        "top_confused_pairs": [
            {"true": t, "pred": p, "count": c} for t, p, c in top_confused_pairs(cm)
        ],
    }
