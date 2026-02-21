"""PyTorch Dataset for product-image classification training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.database import DatabaseManager, get_all_products, get_product_images


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ProductDataset(Dataset):
    """Simple image dataset backed by a flat list of paths + labels."""

    def __init__(
        self,
        image_paths: list[str],
        labels: list[int],
        transform: Any | None = None,
    ) -> None:
        assert len(image_paths) == len(labels), "paths and labels must have equal length"
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path = self.image_paths[index]
        image = Image.open(path).convert("RGB")
        label = self.labels[index]

        if self.transform is not None:
            import numpy as np
            import albumentations as A
            if isinstance(self.transform, A.Compose):
                # Albumentations expects numpy HWC array
                img_np = np.array(image)
                transformed = self.transform(image=img_np)
                image = transformed["image"]
            else:
                image = self.transform(image)

        return image, label


# ---------------------------------------------------------------------------
# Label map
# ---------------------------------------------------------------------------

def build_label_map(db_manager: DatabaseManager) -> dict[int, int]:
    """Create a mapping from ``product_id`` to a contiguous label index.

    Products are sorted by id so the mapping is deterministic.
    """
    with db_manager as session:
        products = get_all_products(session)
    mapping: dict[int, int] = {}
    for idx, product in enumerate(sorted(products, key=lambda p: p.id)):
        mapping[product.id] = idx
    return mapping


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_datasets(
    db_manager: DatabaseManager,
    train_transform: Any | None = None,
    val_transform: Any | None = None,
    val_split: float = 0.2,
) -> tuple[ProductDataset, ProductDataset, dict[int, int]]:
    """Build train / validation datasets from the database.

    Parameters
    ----------
    db_manager:
        An initialised :class:`DatabaseManager`.
    train_transform:
        Torchvision transform pipeline applied to training images.
    val_transform:
        Torchvision transform pipeline applied to validation images.
    val_split:
        Fraction of images *per product* to hold out for validation.

    Returns
    -------
    (train_dataset, val_dataset, label_map)
        ``label_map`` maps ``product_id -> contiguous int label``.
    """
    import random

    label_map = build_label_map(db_manager)

    train_paths: list[str] = []
    train_labels: list[int] = []
    val_paths: list[str] = []
    val_labels: list[int] = []

    with db_manager as session:
        for product_id, label_idx in label_map.items():
            images = get_product_images(session, product_id)
            paths = [img.file_path for img in images if Path(img.file_path).exists()]

            if not paths:
                continue

            # Deterministic shuffle per product
            rng = random.Random(product_id)
            rng.shuffle(paths)

            n_val = max(1, int(len(paths) * val_split)) if len(paths) > 1 else 0
            val_set = paths[:n_val]
            train_set = paths[n_val:]

            train_paths.extend(train_set)
            train_labels.extend([label_idx] * len(train_set))
            val_paths.extend(val_set)
            val_labels.extend([label_idx] * len(val_set))

    train_dataset = ProductDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = ProductDataset(val_paths, val_labels, transform=val_transform)

    return train_dataset, val_dataset, label_map
