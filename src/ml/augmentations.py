"""Augmentation pipelines including random background replacement."""

from __future__ import annotations

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet channel statistics
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class RandomBackground(A.ImageOnlyTransform):
    """Replace near-white background pixels with a random solid color.

    This forces the model to learn the object shape rather than
    memorizing the background.  Works best when background removal
    (GrabCut) has already set the background to white.
    """

    def __init__(self, threshold: int = 230, always_apply: bool = False, p: float = 0.5):
        super().__init__(p=p)
        self.threshold = threshold

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        # Detect near-white pixels (background from GrabCut)
        gray = img.mean(axis=2) if img.ndim == 3 else img
        bg_mask = gray > self.threshold

        if bg_mask.sum() < 100:
            # Very few background pixels, skip
            return img

        # Random solid color
        rng = np.random.default_rng()
        color = rng.integers(0, 256, size=3, dtype=np.uint8)

        result = img.copy()
        result[bg_mask] = color
        return result

    def get_transform_init_args_names(self):
        return ("threshold",)


def get_train_transforms(size: int = 224) -> A.Compose:
    """Heavy augmentation pipeline for training.

    Includes random background replacement, spatial and color transforms.
    """
    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(size, size),
                scale=(0.65, 1.0),
                ratio=(0.85, 1.15),
            ),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=0, value=(255, 255, 255), p=0.5),
            RandomBackground(threshold=230, p=0.6),
            A.ColorJitter(
                brightness=0.35,
                contrast=0.35,
                saturation=0.35,
                hue=0.08,
                p=0.8,
            ),
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            ], p=0.3),
            A.RandomRotate90(p=0.3),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms(size: int = 224) -> A.Compose:
    """Deterministic pipeline for validation."""
    resize_to = int(size * (232 / 224))
    return A.Compose(
        [
            A.Resize(height=resize_to, width=resize_to),
            A.CenterCrop(height=size, width=size),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_inference_transforms(size: int = 224) -> A.Compose:
    """Inference transform pipeline (identical to validation)."""
    return get_val_transforms(size)
