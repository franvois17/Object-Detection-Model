"""Albumentations-based augmentation pipelines for training and inference."""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet channel statistics
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(size: int = 224) -> A.Compose:
    """Return a heavy augmentation pipeline for training.

    Includes random spatial and colour transforms that help the model
    generalise to new viewpoints and lighting conditions.
    """
    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(size, size),
                scale=(0.7, 1.0),
                ratio=(0.9, 1.1),
            ),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.05,
                p=0.8,
            ),
            A.GaussNoise(p=0.3),
            A.RandomRotate90(p=0.3),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms(size: int = 224) -> A.Compose:
    """Return a deterministic pipeline for validation.

    Matches the standard EfficientNet preprocessing: resize to a
    slightly larger resolution then centre-crop to *size*.
    """
    resize_to = int(size * (232 / 224))  # keep same ratio as torchvision
    return A.Compose(
        [
            A.Resize(height=resize_to, width=resize_to),
            A.CenterCrop(height=size, width=size),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_inference_transforms(size: int = 224) -> A.Compose:
    """Return the inference transform pipeline (identical to validation)."""
    return get_val_transforms(size)
