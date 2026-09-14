"""Models take float RGB in [0, 1] at 64×64. Normalization (and resizing) live INSIDE the model, so the exported
file needs no separate preprocessing code that could drift out of sync."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Normalize(nn.Module):
    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class SmallCNN(nn.Module):
    """Baseline trained from scratch: 4 conv blocks (64 → 32 → 16 → 8 → 4 px), global average pooling, linear head."""

    def __init__(self, n_classes: int = 10, width: int = 32, dropout: float = 0.2):
        super().__init__()

        def block(c_in: int, c_out: int) -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True), nn.MaxPool2d(2))

        self.normalize = Normalize()
        self.features = nn.Sequential(block(3, width), block(width, 2 * width), block(2 * width, 4 * width), block(4 * width, 4 * width))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(4 * width, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(self.normalize(x)))


class TransferResNet(nn.Module):
    """ImageNet-pretrained ResNet-18 with a new 10-class head. Tiles are upsampled 64 → `input_size` inside the model,
    because ImageNet filters expect objects at larger pixel scales and 64 px would shrink layer4 to a 2×2 grid."""

    def __init__(self, n_classes: int = 10, input_size: int = 96, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18  # training-only dependency

        self.input_size = input_size
        self.normalize = Normalize()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.backbone.fc.in_features, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        return self.backbone(self.normalize(x))

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.backbone.fc.parameters())

    def body_parameters(self) -> list[nn.Parameter]:
        return [p for name, p in self.backbone.named_parameters() if not name.startswith("fc.")]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
