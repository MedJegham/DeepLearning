"""Grad-CAM simplifié pour interpréter les prédictions (CNN ou ResNet18 transfer)."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _gradcam_on_maps(
    activations: torch.Tensor,
    gradients: torch.Tensor,
    size: Tuple[int, int],
) -> np.ndarray:
    # activations, gradients: (1, C, H, W)
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = cam.squeeze().detach().cpu().numpy()
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    cam_t = torch.tensor(cam).unsqueeze(0).unsqueeze(0)
    cam_up = F.interpolate(cam_t, size=size, mode="bilinear", align_corners=False)
    return cam_up.squeeze().numpy()


def gradcam_cnn(model: nn.Module, x: torch.Tensor, target_class: int) -> np.ndarray:
    """x: (1, 3, H, W), modèle = CNN du projet (attribut block2)."""
    acts, grads = [], []

    def fwd_hook(_m, _inp, out):
        acts.append(out.detach())

    def full_bwd_hook(_m, _gi, go):
        if go[0] is not None:
            grads.append(go[0].detach())

    h1 = model.block2.register_forward_hook(fwd_hook)
    h2 = model.block2.register_full_backward_hook(full_bwd_hook)
    model.zero_grad()
    logits = model(x)
    score = logits[0, target_class]
    score.backward()
    h1.remove()
    h2.remove()
    _, _, h, w = x.shape
    return _gradcam_on_maps(acts[0], grads[0], (h, w))


def gradcam_resnet18(model: nn.Module, x: torch.Tensor, target_class: int) -> np.ndarray:
    """model = ResNet18Transfer (attribut .backbone, dernière couche conv = layer4)."""
    acts, grads = [], []
    layer = model.backbone.layer4[-1]

    def fwd_hook(_m, _inp, out):
        acts.append(out.detach())

    def full_bwd_hook(_m, _gi, go):
        if go[0] is not None:
            grads.append(go[0].detach())

    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(full_bwd_hook)
    model.zero_grad()
    logits = model(x)
    score = logits[0, target_class]
    score.backward()
    h1.remove()
    h2.remove()
    _, _, h, w = x.shape
    return _gradcam_on_maps(acts[0], grads[0], (h, w))


def overlay_heatmap(rgb_uint8: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """rgb_uint8 (H,W,3), heatmap (H,W) dans [0,1]."""
    import matplotlib

    cmap = matplotlib.colormaps["jet"]
    hm_color = (cmap(heatmap)[..., :3] * 255).astype(np.uint8)
    return np.clip((1 - alpha) * rgb_uint8 + alpha * hm_color, 0, 255).astype(np.uint8)
