"""Model construction: pretrained ResNet-50 transfer learning for 102 flowers."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def build_model(
    num_classes: int = 102,
    backbone: str = "resnet50",
    freeze_backbone: bool = False,
    dropout: float = 0.2,
) -> nn.Module:
    """Build a transfer-learning classifier.

    Why ResNet-50: a well-understood, strong baseline for fine-grained
    classification; pretrained ImageNet features transfer well to flower
    morphology (petal texture, shape), and residual connections keep
    gradients stable during full fine-tuning.
    """
    if backbone != "resnet50":
        raise ValueError(f"Unsupported backbone: {backbone}")
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def target_layer(model: nn.Module) -> nn.Module:
    """Last convolutional block of ResNet-50 — the standard Grad-CAM target."""
    return model.layer4[-1].conv3


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_checkpoint(model: nn.Module, path: str, extra: dict | None = None) -> None:
    payload = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(model: nn.Module, path: str, device: torch.device) -> dict:
    payload = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(payload["state_dict"])
    return payload
