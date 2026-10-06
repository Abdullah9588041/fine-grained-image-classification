"""End-to-end analysis: train, evaluate, ablate, explain, and record real metrics.

Usage:
    python scripts/run_analysis.py --epochs 12 --ablation-epochs 3
    python scripts/run_analysis.py --skip-training   # reuse checkpoints/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from flowers.data import DataConfig, build_dataloaders, load_official_splits  # noqa: E402
from flowers.evaluate import (  # noqa: E402
    confusion_matrix,
    evaluate_test,
    summarize,
)
from flowers.explain import GradCAM, denormalize, overlay_heatmap  # noqa: E402
from flowers.models import build_model, count_parameters, load_checkpoint, target_layer  # noqa: E402
from flowers.train import TrainConfig, evaluate_loss_acc, fit, seed_everything  # noqa: E402
from flowers.visualization import (  # noqa: E402
    plot_confusion_subset,
    plot_training_curves,
    save_gradcam_grid,
)

RAW = ROOT / "data" / "raw"
CHECKPOINTS = ROOT / "checkpoints"
FIGURES = ROOT / "results" / "figures"


def train_run(name: str, cfg: DataConfig, tcfg: TrainConfig, device, augment: bool,
              freeze: bool, epochs: int, full_test_eval: bool = True) -> dict:
    loaders = build_dataloaders(RAW, cfg, augment=augment)
    model = build_model(num_classes=102, freeze_backbone=freeze)
    total, trainable = count_parameters(model)
    print(f"[{name}] params total={total:,} trainable={trainable:,}", flush=True)
    tcfg.epochs = epochs
    t0 = time.time()
    history = fit(model, loaders, tcfg, device, CHECKPOINTS, run_name=name)
    wall = time.time() - t0
    ckpt = CHECKPOINTS / f"{name}_best.pt"
    load_checkpoint(model, str(ckpt), device)
    stats = evaluate_loss_acc(model, loaders["val"], torch.nn.CrossEntropyLoss(), device)
    if full_test_eval:
        # single test pass -> top-1/top-5/predictions (test set is large; one pass only)
        y_true, y_pred, top5 = evaluate_test(model, loaders["test"], device, k=5)
        summary = summarize(y_true, y_pred)
        summary["top5_accuracy"] = top5
        cm_list = confusion_matrix(y_true, y_pred, 102).tolist()
    else:
        # ablations: validation metrics only (documented runtime trade-off)
        summary = {"top1_accuracy": None, "note": "test set not evaluated for ablations"}
        cm_list = None
    return {
        "name": name,
        "augment": augment,
        "freeze_backbone": freeze,
        "epochs_run": len(history["train_loss"]),
        "wall_seconds": round(wall, 1),
        "val_loss": round(stats["loss"], 4),
        "val_acc": round(stats["acc"], 4),
        "history": history,
        "test_summary": summary,
        "confusion_matrix": cm_list,
        "trainable_params": trainable,
    }


def gradcam_panels(model, loaders, device, n_correct=3, n_wrong=3):
    """Collect (original, overlay, caption) panels for correct + misclassified."""
    from torch.utils.data import DataLoader
    from flowers.data import FlowersDataset, eval_transforms, load_labels

    model.eval()
    cam = GradCAM(model, target_layer(model))
    panels = []
    try:
        test_ds = loaders["test"].dataset
        # test_ds is FlowersDataset wrapped? build_dataloaders returns DataLoader of FlowersDataset directly
        correct, wrong = [], []
        with torch.no_grad():
            for i in range(len(test_ds)):
                if len(correct) >= n_correct and len(wrong) >= n_wrong:
                    break
                x, y = test_ds[i]
                pred = model(x.unsqueeze(0).to(device)).argmax(1).item()
                (correct if pred == y else wrong).append((i, x, y, pred))
        for i, x, y, pred in correct[:n_correct] + wrong[:n_wrong]:
            heat = cam(x.unsqueeze(0).to(device), class_idx=pred)
            orig = denormalize(x)
            over = overlay_heatmap(orig, heat)
            tag = "correct" if pred == y else "wrong"
            panels.append((orig, over, f"true={y} pred={pred} [{tag}]"))
    finally:
        cam.remove()
    return panels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--ablation-epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-training", action="store_true")
    ap.add_argument("--skip-ablations", action="store_true")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_threads = torch.get_num_threads()
    print(f"device={device} threads={n_threads}", flush=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    cfg = DataConfig(batch_size=args.batch_size, num_workers=2, seed=args.seed)
    tcfg = TrainConfig(seed=args.seed, amp=torch.cuda.is_available())
    metrics: dict = {
        "device": str(device),
        "torch_threads": n_threads,
        "seed": args.seed,
        "dataset": "Oxford 102 Flowers (8189 images, 102 classes; official splits 1020/1020/6149)",
    }

    t0 = time.time()
    if not args.skip_training:
        main_res = train_run("main", cfg, tcfg, device, augment=True, freeze=False,
                             epochs=args.epochs)
    else:
        main_res = None
        # load existing main checkpoint for evaluation/explainability
        model = build_model(num_classes=102)
        load_checkpoint(model, str(CHECKPOINTS / "main_best.pt"), device)
        loaders = build_dataloaders(RAW, cfg, augment=True)
        stats = evaluate_loss_acc(model, loaders["val"], torch.nn.CrossEntropyLoss(), device)
        y_true, y_pred, top5 = evaluate_test(model, loaders["test"], device, k=5)
        summary = summarize(y_true, y_pred)
        summary["top5_accuracy"] = top5
        main_res = {
            "name": "main", "augment": True, "freeze_backbone": False,
            "val_loss": round(stats["loss"], 4), "val_acc": round(stats["acc"], 4),
            "test_summary": summary,
            "confusion_matrix": confusion_matrix(y_true, y_pred, 102).tolist(),
        }
    metrics["main"] = {k: v for k, v in main_res.items() if k != "confusion_matrix"}
    cm = np.array(main_res["confusion_matrix"])

    # figures for main run
    if main_res.get("history"):
        plot_training_curves(main_res["history"], FIGURES / "training_curves.png")
    worst_ids = [c for c, _, _ in main_res["test_summary"]["worst_classes"][:8]]
    plot_confusion_subset(cm, worst_ids, FIGURES / "confusion_worst8.png",
                          title="Confusion matrix — 8 hardest classes")

    # ablations
    if not args.skip_ablations and not args.skip_training:
        ab_noaug = train_run("abl_no_augment", cfg, tcfg, device, augment=False,
                             freeze=False, epochs=args.ablation_epochs,
                             full_test_eval=False)
        ab_frozen = train_run("abl_frozen", cfg, tcfg, device, augment=True,
                              freeze=True, epochs=args.ablation_epochs,
                              full_test_eval=False)
        metrics["ablations"] = {
            r["name"]: {
                "epochs": r["epochs_run"],
                "val_acc": r["val_acc"],
                "val_loss": r["val_loss"],
                "note": "validation metrics only — test set skipped for ablations (runtime)",
                "wall_seconds": r["wall_seconds"],
            }
            for r in (ab_noaug, ab_frozen)
        }
        if ab_noaug.get("history"):
            plot_training_curves(ab_noaug["history"], FIGURES / "training_curves_noaug.png")
        if ab_frozen.get("history"):
            plot_training_curves(ab_frozen["history"], FIGURES / "training_curves_frozen.png")

    # Grad-CAM panels from the main model
    model = build_model(num_classes=102)
    load_checkpoint(model, str(CHECKPOINTS / "main_best.pt"), device)
    loaders = build_dataloaders(RAW, cfg, augment=True)
    panels = gradcam_panels(model, loaders, device)
    save_gradcam_grid(panels, FIGURES / "gradcam_panels.png")

    metrics["total_wall_seconds"] = round(time.time() - t0, 1)
    with open(ROOT / "results" / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    # also store the full confusion matrix separately (large)
    np.save(ROOT / "results" / "confusion_matrix.npy", cm)
    print("metrics written to results/metrics.json", flush=True)
    print(json.dumps({k: v for k, v in metrics.items()
                      if k not in ("main",)}, indent=2)[:2000], flush=True)


if __name__ == "__main__":
    main()
