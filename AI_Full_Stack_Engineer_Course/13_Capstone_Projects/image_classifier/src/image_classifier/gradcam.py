"""Grad-CAM (Selvaraju et al., 2017): which regions raised the score of a class?"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class GradCAM:
    """Hooks capture a conv layer's activations A and the gradient of the class logit w.r.t. A.

    weights α_k = spatial mean of ∂y_c/∂A_k;  CAM = ReLU(Σ_k α_k · A_k), upsampled to the image and scaled to [0, 1].
    Use as a context manager so the hooks are always removed.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model, self.activations, self.gradients = model, None, None
        self.handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()
        output.register_hook(lambda grad: setattr(self, "gradients", grad.detach()))

    def __call__(self, images: torch.Tensor, class_idx: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(images)
            if class_idx is None:
                class_idx = logits.argmax(dim=1)
            logits.gather(1, class_idx.view(-1, 1)).sum().backward()  # each image's own class score
        alphas = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((alphas * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        return cam.detach(), logits.detach(), class_idx

    def remove(self) -> None:
        self.handle.remove()

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(self, *exc) -> None:
        self.remove()
