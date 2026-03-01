"""Training worker -- incremental (FAISS-only) and full (fine-tune + FAISS)."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Signal

from src.core.config import CONFIG, Config
from src.core.device import DEVICE
from src.ml.backbone import FeatureExtractor
from src.ml.embeddings import EmbeddingIndex
from src.workers.base_worker import BaseWorker

logger = logging.getLogger(__name__)


class TrainingWorker(BaseWorker):
    """Build or update the product-recognition models.

    Two modes are supported:

    * **incremental** -- extract embeddings with the frozen backbone and
      rebuild the FAISS index.  Fast, no GPU training required.
    * **full** -- fine-tune the backbone classifier, then rebuild the
      FAISS index with the improved features.

    Signals (in addition to those inherited from :class:`BaseWorker`)
    -----------------------------------------------------------------
    epoch_done : Signal(int, float, float)
        Emitted at the end of each training epoch.  Arguments are
        ``(epoch_number, train_loss, val_accuracy)``.
    """

    epoch_done = Signal(int, float, float)

    def __init__(
        self,
        mode: str = "incremental",
        config: Config | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        if mode not in ("incremental", "full"):
            raise ValueError(f"Unknown training mode: {mode!r}")
        self._mode = mode
        self._config = config or CONFIG

    # ------------------------------------------------------------------
    # Worker implementation
    # ------------------------------------------------------------------

    def do_work(self) -> dict:
        if self._mode == "incremental":
            return self._do_incremental()
        return self._do_full()

    # ------------------------------------------------------------------
    # Incremental mode
    # ------------------------------------------------------------------

    def _do_incremental(self) -> dict:
        """Extract embeddings and rebuild the FAISS index."""
        config = self._config

        # 1. Load backbone
        self.progress.emit(0, "Loading feature extractor...")
        backbone = FeatureExtractor(device=DEVICE)

        if self.is_cancelled:
            return {}

        # 2. Discover product images
        self.progress.emit(10, "Scanning product images...")
        images_dir = config.images_dir
        product_dirs = sorted(
            [d for d in images_dir.iterdir() if d.is_dir()]
        ) if images_dir.exists() else []

        if not product_dirs:
            self.progress.emit(100, "No product images found.")
            return {"num_products": 0, "num_embeddings": 0}

        if self.is_cancelled:
            return {}

        # 3. Extract embeddings for every product
        self.progress.emit(20, "Extracting embeddings...")
        index = EmbeddingIndex(dim=config.embedding_dim)
        total_products = len(product_dirs)
        total_embeddings = 0

        from PIL import Image

        for i, product_dir in enumerate(product_dirs):
            if self.is_cancelled:
                return {}

            try:
                product_id = int(product_dir.name)
            except ValueError:
                logger.warning("Skipping non-numeric directory: %s", product_dir)
                continue

            image_files = sorted(
                p for p in product_dir.iterdir()
                if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
            )
            if not image_files:
                continue

            images = []
            for img_path in image_files:
                try:
                    images.append(Image.open(img_path).convert("RGB"))
                except Exception:
                    logger.warning("Cannot open image: %s", img_path)
                    continue

            if images:
                embeddings = backbone.extract_batch(images)
                labels = [product_id] * embeddings.shape[0]
                index.add(embeddings, labels)
                total_embeddings += embeddings.shape[0]

            pct = 20 + int(70 * (i + 1) / total_products)
            self.progress.emit(
                pct,
                f"Embeddings: product {i + 1}/{total_products}",
            )

        if self.is_cancelled:
            return {}

        # 4. Save FAISS index
        self.progress.emit(95, "Saving FAISS index...")
        index_path = config.embeddings_dir / "product_index.faiss"
        config.embeddings_dir.mkdir(parents=True, exist_ok=True)
        index.save(index_path)

        self.progress.emit(100, "Incremental training complete.")
        logger.info(
            "Incremental build: %d products, %d embeddings",
            total_products,
            total_embeddings,
        )
        return {
            "mode": "incremental",
            "num_products": total_products,
            "num_embeddings": total_embeddings,
        }

    # ------------------------------------------------------------------
    # Full training mode
    # ------------------------------------------------------------------

    def _do_full(self) -> dict:
        """Fine-tune the classifier then rebuild the FAISS index."""
        config = self._config

        # 1. Build datasets from product images
        self.progress.emit(0, "Building training datasets...")
        dataset_info = self._build_datasets()
        if not dataset_info:
            self.progress.emit(100, "No training data available.")
            return {"epochs": 0, "best_accuracy": 0.0, "model_path": ""}

        train_loader, val_loader, num_classes, label_map = dataset_info

        if self.is_cancelled:
            return {}

        # 2. Create backbone and classifier
        self.progress.emit(5, "Creating classifier model...")
        import torch
        import torch.nn as nn
        import torch.optim as optim

        backbone = FeatureExtractor(device=DEVICE)
        backbone.unfreeze_last_n_blocks(config.fine_tune_blocks)

        classifier_head = nn.Linear(config.embedding_dim, num_classes).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        params = list(backbone.model.parameters()) + list(classifier_head.parameters())
        optimizer = optim.AdamW(
            params, lr=config.train_lr, weight_decay=config.train_weight_decay
        )

        if self.is_cancelled:
            return {}

        # 3. Training loop
        best_accuracy = 0.0
        total_epochs = config.train_epochs
        self.progress.emit(10, "Starting training...")

        for epoch in range(total_epochs):
            if self.is_cancelled:
                return {}

            # -- Train --
            backbone.model.train()
            classifier_head.train()
            running_loss = 0.0
            num_batches = 0

            for images, labels in train_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()
                features = backbone.model(images)
                logits = classifier_head(features)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                num_batches += 1

            train_loss = running_loss / max(num_batches, 1)

            # -- Validate --
            backbone.model.eval()
            classifier_head.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(DEVICE)
                    labels = labels.to(DEVICE)
                    features = backbone.model(images)
                    logits = classifier_head(features)
                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            val_accuracy = correct / max(total, 1)
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy

            self.epoch_done.emit(epoch + 1, train_loss, val_accuracy)

            pct = 10 + int(70 * (epoch + 1) / total_epochs)
            self.progress.emit(
                pct,
                f"Epoch {epoch + 1}/{total_epochs} -- "
                f"loss: {train_loss:.4f}, acc: {val_accuracy:.4f}",
            )

        if self.is_cancelled:
            return {}

        # 4. Save classifier
        self.progress.emit(85, "Saving classifier model...")
        config.classifier_dir.mkdir(parents=True, exist_ok=True)
        model_path = config.classifier_dir / "classifier.pth"
        torch.save(
            {
                "backbone_state": backbone.model.state_dict(),
                "head_state": classifier_head.state_dict(),
                "num_classes": num_classes,
                "label_map": label_map,
            },
            str(model_path),
        )

        if self.is_cancelled:
            return {}

        # 5. Rebuild FAISS index with pretrained backbone (not fine-tuned)
        #    so KNN inference stays consistent with the frozen extractor
        self.progress.emit(90, "Rebuilding FAISS index...")
        fresh_backbone = FeatureExtractor(device=DEVICE)
        self._rebuild_index(fresh_backbone)

        self.progress.emit(100, "Full training complete.")
        logger.info(
            "Full training done: %d epochs, best accuracy %.4f",
            total_epochs,
            best_accuracy,
        )
        return {
            "mode": "full",
            "epochs": total_epochs,
            "best_accuracy": best_accuracy,
            "model_path": str(model_path),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_datasets(self):
        """Scan the images directory and build train/val DataLoaders.

        Uses albumentations augmentations (including random background
        replacement) for training data.

        Returns
        -------
        tuple | None
            ``(train_loader, val_loader, num_classes, label_map)`` or
            ``None`` when no usable data is found.
        """
        import random
        from torch.utils.data import DataLoader
        from src.data.dataset import ProductDataset
        from src.ml.augmentations import get_train_transforms, get_val_transforms

        config = self._config
        images_dir = config.images_dir
        if not images_dir.exists():
            return None

        product_dirs = sorted(
            d for d in images_dir.iterdir() if d.is_dir()
        )
        if not product_dirs:
            return None

        label_map: dict[int, int] = {}
        train_paths: list[str] = []
        train_labels: list[int] = []
        val_paths: list[str] = []
        val_labels: list[int] = []

        for product_dir in product_dirs:
            try:
                product_id = int(product_dir.name)
            except ValueError:
                continue

            image_files = sorted(
                str(p) for p in product_dir.iterdir()
                if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
            )
            if not image_files:
                continue

            if product_id not in label_map:
                label_map[product_id] = len(label_map)
            class_idx = label_map[product_id]

            # Split per product (deterministic)
            rng = random.Random(product_id)
            shuffled = list(image_files)
            rng.shuffle(shuffled)
            n_val = max(1, int(len(shuffled) * 0.2)) if len(shuffled) > 1 else 0
            val_paths.extend(shuffled[:n_val])
            val_labels.extend([class_idx] * n_val)
            train_paths.extend(shuffled[n_val:])
            train_labels.extend([class_idx] * (len(shuffled) - n_val))

        if not train_paths:
            return None

        num_classes = len(label_map)

        train_ds = ProductDataset(
            train_paths, train_labels, transform=get_train_transforms(config.image_size)
        )
        val_ds = ProductDataset(
            val_paths, val_labels, transform=get_val_transforms(config.image_size)
        )

        train_loader = DataLoader(
            train_ds, batch_size=config.train_batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=config.train_batch_size, shuffle=False
        )
        return train_loader, val_loader, num_classes, label_map

    def _rebuild_index(self, backbone: FeatureExtractor) -> None:
        """Rebuild the FAISS index using the given backbone."""
        from PIL import Image

        config = self._config
        images_dir = config.images_dir
        index = EmbeddingIndex(dim=config.embedding_dim)

        if not images_dir.exists():
            return

        product_dirs = sorted(
            d for d in images_dir.iterdir() if d.is_dir()
        )

        for product_dir in product_dirs:
            if self.is_cancelled:
                return

            try:
                product_id = int(product_dir.name)
            except ValueError:
                continue

            image_files = sorted(
                p for p in product_dir.iterdir()
                if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
            )
            if not image_files:
                continue

            images = []
            for img_path in image_files:
                try:
                    images.append(Image.open(img_path).convert("RGB"))
                except Exception:
                    continue

            if images:
                embeddings = backbone.extract_batch(images)
                labels = [product_id] * embeddings.shape[0]
                index.add(embeddings, labels)

        index_path = config.embeddings_dir / "product_index.faiss"
        config.embeddings_dir.mkdir(parents=True, exist_ok=True)
        index.save(index_path)
        logger.info("FAISS index rebuilt at %s (%d vectors)", index_path, len(index))
