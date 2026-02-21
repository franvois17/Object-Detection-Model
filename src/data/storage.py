"""File I/O and path management for product images and models."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from src.core.config import CONFIG, Config


class StorageManager:
    """Centralised helper for filesystem paths and image persistence."""

    def __init__(self, config: Config | None = None) -> None:
        self._config = config or CONFIG

    # -- product image directories ------------------------------------------

    def get_product_dir(self, product_id: int) -> Path:
        """Return ``data/images/{product_id}/``, creating it if needed."""
        path = self._config.images_dir / str(product_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_original_dir(self, product_id: int) -> Path:
        """Return ``data/images/{product_id}/original/``, creating it if needed."""
        path = self.get_product_dir(product_id) / "original"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # -- saving crops -------------------------------------------------------

    def save_crop(
        self,
        image: np.ndarray,
        product_id: int,
        index: int,
    ) -> Path:
        """Save a cropped BGR image as JPEG (quality 95) and return its path.

        The file is written to ``data/images/{product_id}/{index:05d}.jpg``.
        """
        product_dir = self.get_product_dir(product_id)
        filename = f"{index:05d}.jpg"
        dest = product_dir / filename
        cv2.imwrite(
            str(dest),
            image,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        return dest

    def save_crops(
        self,
        images: list[np.ndarray],
        product_id: int,
    ) -> list[Path]:
        """Save a batch of cropped images, returning their paths."""
        # Determine starting index from existing files
        product_dir = self.get_product_dir(product_id)
        existing = list(product_dir.glob("*.jpg"))
        start_index = len(existing)
        paths: list[Path] = []
        for i, img in enumerate(images):
            path = self.save_crop(img, product_id, start_index + i)
            paths.append(path)
        return paths

    # -- deletion -----------------------------------------------------------

    def delete_product_files(self, product_id: int) -> None:
        """Remove the entire product directory (images and originals)."""
        product_dir = self._config.images_dir / str(product_id)
        if product_dir.exists():
            shutil.rmtree(product_dir)

    # -- listing ------------------------------------------------------------

    def get_product_image_paths(self, product_id: int) -> list[Path]:
        """Return sorted list of image paths inside the product directory."""
        product_dir = self._config.images_dir / str(product_id)
        if not product_dir.exists():
            return []
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        paths = [
            p for p in sorted(product_dir.iterdir())
            if p.suffix.lower() in extensions and p.is_file()
        ]
        return paths

    # -- database export / import -------------------------------------------

    def export_database(self, dest_path: Path) -> None:
        """Copy the SQLite database file to *dest_path*."""
        src = self._config.db_path
        if not src.exists():
            raise FileNotFoundError(f"Database file not found: {src}")
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_path)

    def import_database(self, src_path: Path) -> None:
        """Replace the current database file with *src_path*."""
        src_path = Path(src_path)
        if not src_path.exists():
            raise FileNotFoundError(f"Source database not found: {src_path}")
        dest = self._config.db_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest)

    # -- model paths --------------------------------------------------------

    def get_model_path(self, model_type: str, filename: str) -> Path:
        """Return the full path for a model file in the appropriate directory.

        ``model_type`` should be ``'classifier'``, ``'embeddings'`` (knn), or
        ``'pretrained'``.
        """
        type_to_dir = {
            "classifier": self._config.classifier_dir,
            "knn": self._config.embeddings_dir,
            "embeddings": self._config.embeddings_dir,
            "pretrained": self._config.pretrained_dir,
        }
        base_dir = type_to_dir.get(model_type)
        if base_dir is None:
            raise ValueError(
                f"Unknown model_type {model_type!r}. "
                f"Expected one of {list(type_to_dir)}"
            )
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir / filename
