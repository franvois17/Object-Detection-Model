"""Classification head + full classifier wrapping the EfficientNet backbone."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.core.config import CONFIG
from src.core.device import DEVICE
from src.ml.backbone import FeatureExtractor


class ClassificationHead(nn.Module):
    """Simple linear classification head with optional dropout.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the input feature vector (default 1280).
    num_classes : int
        Number of output classes.
    dropout : float
        Dropout probability applied before the linear layer.
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 1280,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.head(x)


class Classifier:
    """Full product classifier: backbone (feature extractor) + classification head.

    Parameters
    ----------
    num_classes : int
        Number of product classes.
    device : torch.device | None
        Compute device. Falls back to ``DEVICE`` from ``src.core.device``.
    dropout : float
        Dropout probability for the classification head.
    """

    def __init__(
        self,
        num_classes: int,
        device: torch.device | None = None,
        dropout: float = 0.2,
    ) -> None:
        self.device = device or DEVICE
        self.num_classes = num_classes

        self.backbone = FeatureExtractor(device=self.device)
        self.head = ClassificationHead(
            num_classes=num_classes,
            embedding_dim=CONFIG.embedding_dim,
            dropout=dropout,
        ).to(self.device)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_logits(self, image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        """Return raw logits for a single image."""
        tensor = self.backbone._preprocess(image)
        features = self.backbone.model(tensor)
        return self.head(features)

    def _get_logits_batch(
        self, images: list[Union[Image.Image, torch.Tensor]]
    ) -> torch.Tensor:
        """Return raw logits for a batch of images."""
        transform = FeatureExtractor.get_transform()
        tensors: list[torch.Tensor] = []
        for img in images:
            if isinstance(img, Image.Image):
                img = img.convert("RGB")
                tensors.append(transform(img))
            else:
                t = img
                if t.ndim == 4:
                    t = t.squeeze(0)
                tensors.append(t)
        batch = torch.stack(tensors).to(self.device)
        features = self.backbone.model(batch)
        return self.head(features)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, image: Union[Image.Image, torch.Tensor]) -> tuple[int, float]:
        """Predict the class of a single image.

        Returns
        -------
        tuple[int, float]
            ``(class_id, confidence)`` where confidence is the softmax
            probability for the winning class.
        """
        self.backbone.model.eval()
        self.head.eval()
        logits = self._get_logits(image)
        probs = torch.softmax(logits, dim=1)
        confidence, class_id = probs.max(dim=1)
        return int(class_id.item()), float(confidence.item())

    @torch.no_grad()
    def predict_batch(
        self, images: list[Union[Image.Image, torch.Tensor]]
    ) -> list[tuple[int, float]]:
        """Predict the class for each image in a batch.

        Returns
        -------
        list[tuple[int, float]]
            One ``(class_id, confidence)`` pair per image.
        """
        self.backbone.model.eval()
        self.head.eval()
        logits = self._get_logits_batch(images)
        probs = torch.softmax(logits, dim=1)
        confidences, class_ids = probs.max(dim=1)
        return [
            (int(cid.item()), float(conf.item()))
            for cid, conf in zip(class_ids, confidences)
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save backbone and classification head state dicts."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "num_classes": self.num_classes,
                "backbone_state": self.backbone.model.state_dict(),
                "head_state": self.head.state_dict(),
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        num_classes: int,
        device: torch.device | None = None,
    ) -> "Classifier":
        """Load a saved classifier from disk.

        Parameters
        ----------
        path : str | Path
            Path to the saved ``.pt`` checkpoint.
        num_classes : int
            Number of classes the head was trained with.
        device : torch.device | None
            Target device. Defaults to ``DEVICE``.

        Returns
        -------
        Classifier
        """
        device = device or DEVICE
        classifier = cls(num_classes=num_classes, device=device)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        classifier.backbone.model.load_state_dict(checkpoint["backbone_state"])
        classifier.head.load_state_dict(checkpoint["head_state"])
        classifier.backbone.model.eval()
        classifier.head.eval()
        return classifier

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_backbone(self) -> FeatureExtractor:
        """Return the underlying feature extractor (for shared use)."""
        return self.backbone
