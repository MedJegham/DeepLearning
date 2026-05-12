"""Chargement et augmentation des données rétiniennes."""
from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)

AUGMENTATION_PRESETS = [
    T.RandomRotation(degrees=8),
    T.RandomCrop(28, padding=3),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomAffine(degrees=0, shear=10, translate=(0.08, 0.08), scale=(0.95, 1.05)),
    T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
    T.RandomAutocontrast(p=0.5),
    T.RandomAdjustSharpness(sharpness_factor=1.4, p=0.3),
    T.RandomEqualize(p=0.3),
]


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_train_data(data_dir: Path = Path("data")) -> Tuple[np.ndarray, np.ndarray]:
    data = load_pickle(data_dir / "train_data.pkl")
    return data["images"], data["labels"].flatten()


def load_test_data(data_dir: Path = Path("data")) -> np.ndarray:
    data = load_pickle(data_dir / "test_data.pkl")
    return data["images"]


def augment_dataset(images: np.ndarray, labels: np.ndarray, n_augmentations: int = 10,
                    seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Augmente artificiellement le dataset pour limiter le surapprentissage."""
    random.seed(seed)
    np.random.seed(seed)

    aug_images, aug_labels = [], []
    for img, label in zip(images, labels):
        aug_images.append(img)
        aug_labels.append(label)
        pil = Image.fromarray(img, mode="RGB")
        for _ in range(n_augmentations - 1):
            num_tr = len(AUGMENTATION_PRESETS) // 2
            transforms = random.sample(AUGMENTATION_PRESETS, num_tr)
            tr = T.Compose(transforms)
            aug_images.append(np.array(tr(pil)))
            aug_labels.append(label)
    return np.array(aug_images), np.array(aug_labels)


# Normalisation ImageNet (tenseurs déjà dans [0, 1], NCHW)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_normalize(X: torch.Tensor) -> torch.Tensor:
    """Applique mean/std ImageNet sur un batch (N, 3, H, W) en [0, 1]."""
    mean = X.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = X.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (X - mean) / std


def to_tensor_dataset(X: np.ndarray, y: np.ndarray | None = None):
    """Conversion (N, H, W, C) -> (N, C, H, W) normalisée dans [0, 1]."""
    Xt = torch.tensor(X, dtype=torch.float32) / 255.0
    Xt = Xt.permute(0, 3, 1, 2)
    if y is None:
        return Xt
    yt = torch.tensor(y, dtype=torch.long)
    return Xt, yt


def preprocess_single_image(image: np.ndarray, imagenet_norm: bool = False) -> torch.Tensor:
    """Prépare une image RGB 28x28 pour l'inférence (ImageNet norm si transfer learning)."""
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[:2] != (28, 28):
        pil = Image.fromarray(image.astype(np.uint8))
        pil = pil.resize((28, 28), Image.BILINEAR)
        image = np.array(pil)
    tensor = torch.tensor(image, dtype=torch.float32) / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    if imagenet_norm:
        tensor = imagenet_normalize(tensor)
    return tensor
