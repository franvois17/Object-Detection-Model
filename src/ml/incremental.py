"""Incremental learning: add or remove products without full retraining."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from src.ml.backbone import FeatureExtractor
    from src.ml.embeddings import EmbeddingIndex

logger = logging.getLogger(__name__)


class IncrementalLearner:
    """Manage the FAISS embedding index incrementally.

    Adding a new product only requires extracting its embeddings and
    inserting them into the index — typically 5-15 seconds depending on
    the number of reference images.

    Parameters
    ----------
    backbone : FeatureExtractor
        Shared feature extractor used to compute embeddings.
    embedding_index : EmbeddingIndex
        FAISS index where product embeddings are stored.
    """

    def __init__(
        self,
        backbone: "FeatureExtractor",
        embedding_index: "EmbeddingIndex",
    ) -> None:
        self.backbone = backbone
        self.index = embedding_index

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_product(self, images: list[Image.Image], product_id: int) -> int:
        """Extract embeddings from *images* and add them to the index.

        Parameters
        ----------
        images : list[PIL.Image.Image]
            Reference images for the product.
        product_id : int
            Unique product identifier.

        Returns
        -------
        int
            Number of embeddings added.
        """
        if not images:
            return 0

        embeddings = self.backbone.extract_batch(images)
        labels = [product_id] * len(images)
        self.index.add(embeddings, labels)

        logger.info(
            "Added %d embeddings for product_id=%d (index size: %d)",
            len(images),
            product_id,
            len(self.index),
        )
        return len(images)

    def remove_product(self, product_id: int) -> None:
        """Remove all embeddings associated with *product_id*.

        Since ``IndexFlatIP`` does not support selective removal, the
        index is rebuilt from scratch without the target product's
        vectors.
        """
        # Collect surviving embeddings and labels
        surviving_embeddings: list[np.ndarray] = []
        surviving_labels: list[int] = []

        # FAISS IndexFlatIP stores raw vectors; we can reconstruct them.
        total = len(self.index)
        if total == 0:
            return

        all_vectors = np.zeros((total, self.index.dim), dtype=np.float32)
        for i in range(total):
            self.index._index.reconstruct(i, all_vectors[i])

        for i in range(total):
            if self.index._labels[i] != product_id:
                surviving_embeddings.append(all_vectors[i])
                surviving_labels.append(self.index._labels[i])

        removed = total - len(surviving_labels)
        self.index.clear()

        if surviving_embeddings:
            # Vectors are already normalised inside the index; add them
            # directly (re-normalisation is idempotent for unit vectors).
            stacked = np.stack(surviving_embeddings)
            self.index.add(stacked, surviving_labels)

        logger.info(
            "Removed %d embeddings for product_id=%d (index size: %d)",
            removed,
            product_id,
            len(self.index),
        )

    def rebuild_index(
        self, all_images: dict[int, list[Image.Image]]
    ) -> None:
        """Rebuild the entire index from scratch.

        Parameters
        ----------
        all_images : dict[int, list[PIL.Image.Image]]
            Mapping from ``product_id`` to a list of reference images.
        """
        self.index.clear()
        total_added = 0

        for product_id, images in all_images.items():
            if not images:
                continue
            embeddings = self.backbone.extract_batch(images)
            labels = [product_id] * len(images)
            self.index.add(embeddings, labels)
            total_added += len(images)

        logger.info(
            "Rebuilt index with %d embeddings for %d products",
            total_added,
            len(all_images),
        )
