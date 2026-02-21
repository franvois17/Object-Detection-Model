"""Unified inference interface for product detection."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from src.core.config import CONFIG, Config
from src.core.device import DEVICE
from src.ml.backbone import FeatureExtractor
from src.ml.embeddings import EmbeddingIndex

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Prediction dataclass
# ------------------------------------------------------------------

@dataclass
class Prediction:
    """Result of a single inference pass."""

    product_id: int | None
    product_name: str
    confidence: float
    method: str  # 'knn' or 'classifier'


# ------------------------------------------------------------------
# Inference engine
# ------------------------------------------------------------------

class InferenceEngine:
    """High-level inference interface that unifies KNN and classifier modes.

    Parameters
    ----------
    config : Config | None
        Application configuration. Defaults to the global ``CONFIG``.
    label_map : dict[int, str] | None
        Optional mapping from class/product id to human-readable name.
    """

    def __init__(
        self,
        config: Config | None = None,
        label_map: dict[int, str] | None = None,
    ) -> None:
        self.config = config or CONFIG
        self.device = DEVICE
        self.label_map: dict[int, str] = label_map or {}

        self.backbone: FeatureExtractor | None = None
        self.embedding_index: EmbeddingIndex | None = None
        self.classifier: object | None = None  # avoid eager import

        self._mode: str = "knn"  # default mode

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_models(self) -> None:
        """Load backbone, FAISS index, and (optionally) classifier from disk."""
        # Backbone is always needed
        self.backbone = FeatureExtractor(device=self.device)
        logger.info("Loaded backbone (EfficientNet-V2-S)")

        # FAISS index
        index_path = self.config.embeddings_dir / "product_index.faiss"
        self.embedding_index = EmbeddingIndex(dim=self.config.embedding_dim)
        if index_path.exists():
            self.embedding_index.load(index_path)
            logger.info(
                "Loaded FAISS index with %d vectors", len(self.embedding_index)
            )
        else:
            logger.warning(
                "No FAISS index found at %s — starting with empty index",
                index_path,
            )

        # Classifier (optional)
        classifier_path = self.config.classifier_dir / "classifier.pth"
        if classifier_path.exists():
            try:
                from src.ml.classifier import Classifier

                # We need to know num_classes; read from checkpoint
                import torch

                checkpoint = torch.load(
                    classifier_path, map_location=self.device, weights_only=False
                )
                num_classes = checkpoint["num_classes"]
                label_map = checkpoint.get("label_map", {})
                # Reverse label_map (product_id -> class_idx) to build name map
                if label_map:
                    self.label_map.update(
                        {pid: f"product_{pid}" for pid in label_map}
                    )
                self.classifier = Classifier.load(
                    classifier_path, num_classes=num_classes, device=self.device
                )
                logger.info(
                    "Loaded classifier (%d classes)", num_classes
                )
            except Exception:
                logger.warning(
                    "Could not load classifier from %s", classifier_path, exc_info=True
                )
        else:
            logger.info("No classifier checkpoint found — KNN-only mode")

    # ------------------------------------------------------------------
    # Mode selection
    # ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        """Switch inference mode.

        Parameters
        ----------
        mode : str
            ``'knn'`` for embedding similarity or ``'classifier'`` for the
            trained classification head.

        Raises
        ------
        ValueError
            If *mode* is not ``'knn'`` or ``'classifier'``.
        RuntimeError
            If ``'classifier'`` is requested but no classifier is loaded.
        """
        if mode not in ("knn", "classifier"):
            raise ValueError(f"Unknown mode '{mode}'. Use 'knn' or 'classifier'.")
        if mode == "classifier" and self.classifier is None:
            raise RuntimeError("Classifier mode requested but no classifier is loaded.")
        self._mode = mode
        logger.info("Inference mode set to '%s'", mode)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _resolve_name(self, product_id: int | None) -> str:
        if product_id is None:
            return "Desconocido"
        return self.label_map.get(product_id, f"product_{product_id}")

    def _predict_knn(self, image: Image.Image) -> Prediction:
        assert self.backbone is not None
        assert self.embedding_index is not None

        embedding = self.backbone.extract(image)
        results = self.embedding_index.search(embedding, k=self.config.knn_k)

        if not results:
            return Prediction(
                product_id=None,
                product_name="Desconocido",
                confidence=0.0,
                method="knn",
            )

        # Majority vote weighted by similarity
        votes: Counter[int] = Counter()
        score_sums: dict[int, float] = {}
        for label, score in results:
            votes[label] += 1
            score_sums[label] = score_sums.get(label, 0.0) + score

        best_label = votes.most_common(1)[0][0]
        confidence = score_sums[best_label] / votes[best_label]

        if confidence < self.config.confidence_threshold:
            return Prediction(
                product_id=best_label,
                product_name="Desconocido",
                confidence=confidence,
                method="knn",
            )

        return Prediction(
            product_id=best_label,
            product_name=self._resolve_name(best_label),
            confidence=confidence,
            method="knn",
        )

    def _predict_classifier(self, image: Image.Image) -> Prediction:
        from src.ml.classifier import Classifier

        assert isinstance(self.classifier, Classifier)
        class_id, confidence = self.classifier.predict(image)

        if confidence < self.config.confidence_threshold:
            return Prediction(
                product_id=class_id,
                product_name="Desconocido",
                confidence=confidence,
                method="classifier",
            )

        return Prediction(
            product_id=class_id,
            product_name=self._resolve_name(class_id),
            confidence=confidence,
            method="classifier",
        )

    def predict(self, image: np.ndarray | Image.Image) -> Prediction:
        """Run inference on a single image.

        Parameters
        ----------
        image : np.ndarray | PIL.Image.Image
            Input image (BGR numpy array or RGB PIL image).

        Returns
        -------
        Prediction
        """
        if isinstance(image, np.ndarray):
            # Convert BGR (OpenCV) to RGB PIL
            image = Image.fromarray(image[..., ::-1] if image.ndim == 3 else image)

        if self._mode == "knn":
            return self._predict_knn(image)
        return self._predict_classifier(image)

    def predict_batch(
        self, images: list[np.ndarray | Image.Image]
    ) -> list[Prediction]:
        """Run inference on a batch of images.

        Parameters
        ----------
        images : list[np.ndarray | PIL.Image.Image]
            Input images.

        Returns
        -------
        list[Prediction]
        """
        return [self.predict(img) for img in images]

    # ------------------------------------------------------------------
    # Temporal smoothing
    # ------------------------------------------------------------------

    def temporal_smooth(
        self,
        predictions: list[Prediction],
        window: int = 5,
    ) -> Prediction:
        """Apply majority-vote smoothing over a sliding window.

        Parameters
        ----------
        predictions : list[Prediction]
            Recent predictions (most recent last).
        window : int
            Number of predictions to consider.

        Returns
        -------
        Prediction
            The smoothed prediction.
        """
        recent = predictions[-window:]
        if not recent:
            return Prediction(
                product_id=None,
                product_name="Desconocido",
                confidence=0.0,
                method=self._mode,
            )

        # Majority vote on product_id
        votes: Counter[int | None] = Counter()
        confidence_sums: dict[int | None, float] = {}
        for pred in recent:
            votes[pred.product_id] += 1
            confidence_sums[pred.product_id] = (
                confidence_sums.get(pred.product_id, 0.0) + pred.confidence
            )

        best_id = votes.most_common(1)[0][0]
        avg_confidence = confidence_sums[best_id] / votes[best_id]

        name = self._resolve_name(best_id)
        if avg_confidence < self.config.confidence_threshold:
            name = "Desconocido"

        return Prediction(
            product_id=best_id,
            product_name=name,
            confidence=avg_confidence,
            method=self._mode,
        )
