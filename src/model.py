"""Définitions des architectures Deep Learning utilisées dans le projet.

Modèles fournis :
  * ``CNN`` : réseau convolutif maison (inspiré de
    https://www.nature.com/articles/s41598-025-87171-9)
  * ``SmallResNet`` : ResNet allégée entraînée from scratch (28×28)
  * ``ResNet18Transfer`` : ResNet18 **pré-entraîné ImageNet** + tête 5 classes (transfer learning)
  * ``MLPBaseline`` : baseline pleinement connectée (sans prior spatial)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ResNet18Transfer(nn.Module):
    """ResNet18 pré-entraîné ImageNet, tête remplacée pour ``num_classes``."""

    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        try:
            w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=w)
        except AttributeError:
            # torchvision < 0.13
            self.backbone = models.resnet18(pretrained=pretrained)
        inf = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(inf, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class CNN(nn.Module):
    """CNN compact pour images 28x28x3, 5 classes."""

    def __init__(self, in_channels: int = 3, num_classes: int = 5,
                 dimension_of_image: int = 28, dropout: float = 0.1):
        super().__init__()
        pool_kernel_size = 2
        nb_of_pooling_step = 2

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2),
        )

        dim_after_convs = dimension_of_image // (nb_of_pooling_step * pool_kernel_size)
        flattened_features = 128 * dim_after_convs * dim_after_convs

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flattened_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=dropout)
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout(x)
        return self.fc_out(x)


class _BasicResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3,
                                stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class SmallResNet(nn.Module):
    """ResNet réduit (~ResNet-10) adapté aux images 28x28x3."""

    def __init__(self, in_channels: int = 3, num_classes: int = 5,
                 dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(_BasicResBlock(32, 32, 1), _BasicResBlock(32, 32, 1))
        self.layer2 = nn.Sequential(_BasicResBlock(32, 64, 2), _BasicResBlock(64, 64, 1))
        self.layer3 = nn.Sequential(_BasicResBlock(64, 128, 2), _BasicResBlock(128, 128, 1))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


class MLPBaseline(nn.Module):
    """Baseline MLP : pas de prior spatial, sert de référence."""

    def __init__(self, in_channels: int = 3, num_classes: int = 5,
                 dimension_of_image: int = 28, dropout: float = 0.3):
        super().__init__()
        n_in = in_channels * dimension_of_image * dimension_of_image
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_in, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


MODEL_REGISTRY = {
    "cnn": CNN,
    "resnet": SmallResNet,
    "mlp": MLPBaseline,
    "resnet18_tl": ResNet18Transfer,
}


def build_model(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "resnet18_tl":
        nc = int(kwargs.get("num_classes", 5))
        pre = kwargs.get("pretrained", True)
        if isinstance(pre, str):
            pre = pre.lower() in ("1", "true", "yes")
        return ResNet18Transfer(num_classes=nc, pretrained=bool(pre))
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Modèle inconnu: {name}. Disponibles: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
