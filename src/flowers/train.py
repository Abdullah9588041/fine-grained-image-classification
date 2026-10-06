"""Training loop: AdamW, cosine schedule, early stopping, checkpointing, CSV logging."""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .models import save_checkpoint


@dataclass
class TrainConfig:
    epochs: int = 12
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 4
    seed: int = 42
    amp: bool = False  # enabled automatically when CUDA is available
    log_every: int = 20


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)  # perf; seeds still fixed


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    log_every: int = 20,
) -> float:
    model.train()
    running = 0.0
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        running += loss.item() * x.size(0)
        if log_every and (step + 1) % log_every == 0:
            print(f"    step {step + 1}/{len(loader)} loss={loss.item():.4f}", flush=True)
    return running / len(loader.dataset)


@torch.no_grad()
def evaluate_loss_acc(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Dict[str, float]:
    model.eval()
    total_loss, correct1, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        total_loss += criterion(out, y).item() * x.size(0)
        correct1 += (out.argmax(1) == y).sum().item()
        n += x.size(0)
    return {"loss": total_loss / n, "acc": correct1 / n}


def fit(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    cfg: TrainConfig,
    device: torch.device,
    workdir: str | Path,
    run_name: str = "main",
) -> Dict[str, List[float]]:
    """Full training run. Returns history; saves best checkpoint + CSV log."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }
    best_val = float("inf")
    bad_epochs = 0
    ckpt_path = workdir / f"{run_name}_best.pt"
    log_path = workdir / f"{run_name}_log.csv"

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc", "lr"])
        for epoch in range(cfg.epochs):
            tr_loss = train_one_epoch(
                model, loaders["train"], criterion, optimizer, device, scaler, cfg.log_every
            )
            stats = evaluate_loss_acc(model, loaders["val"], criterion, device)
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(stats["loss"])
            history["val_acc"].append(stats["acc"])
            history["lr"].append(lr)
            writer.writerow([epoch + 1, f"{tr_loss:.6f}", f"{stats['loss']:.6f}",
                             f"{stats['acc']:.6f}", f"{lr:.8f}"])
            print(
                f"epoch {epoch + 1}/{cfg.epochs} "
                f"train_loss={tr_loss:.4f} val_loss={stats['loss']:.4f} "
                f"val_acc={stats['acc']:.4f} lr={lr:.2e}",
                flush=True,
            )
            if stats["loss"] < best_val - 1e-6:
                best_val = stats["loss"]
                bad_epochs = 0
                save_checkpoint(model, str(ckpt_path), {"epoch": epoch, "val_loss": best_val})
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    print(f"early stopping at epoch {epoch + 1}", flush=True)
                    break
    return history
