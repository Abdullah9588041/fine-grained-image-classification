"""Grad-CAM implemented from scratch (Selvaraju et al., 2017).

For a target class c and convolutional feature maps A^k, the class-discriminative
localization map is L = ReLU(sum_k alpha_k^c A^k) with
alpha_k^c = (1/Z) sum_{i,j} d y^c / d A^k_{ij}  (global-average-pooled gradients).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._hooks = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """Return a normalized heatmap in [0, 1] at feature-map resolution."""
        self.model.zero_grad(set_to_none=True)
        out = self.model(x)
        if class_idx is None:
            class_idx = int(out.argmax(1)[0])
        score = out[0, class_idx]
        score.backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "Grad-CAM backward hook did not fire: the target layer is outside the "
                "autograd graph (e.g. the backbone is frozen). Use an unfrozen model."
            )
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # alpha_k^c
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.cpu().numpy()

    def remove(self):
        for h in self._hooks:
            h.remove()


def overlay_heatmap(
    image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45
) -> Image.Image:
    """Blend a JET-colored heatmap over the original image."""
    import matplotlib.cm as cm

    h, w = image.size[1], image.size[0]
    colored = (cm.jet(heatmap)[:, :, :3] * 255).astype(np.uint8)
    heat_img = Image.fromarray(colored).resize((w, h), Image.BILINEAR)
    base = np.asarray(image.convert("RGB")).astype(np.float32)
    blended = (1 - alpha) * base + alpha * np.asarray(heat_img).astype(np.float32)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def denormalize(tensor: torch.Tensor) -> Image.Image:
    """Undo ImageNet normalization for display."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    return Image.fromarray((img.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8))
