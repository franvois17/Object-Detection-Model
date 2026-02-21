"""EfficientNet-V2-S feature extractor for product embeddings."""

from __future__ import annotations

from typing import Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from src.core.config import CONFIG
from src.core.device import DEVICE


class FeatureExtractor:
    """Wraps EfficientNet-V2-S as a frozen feature extractor.

    The classification head is removed; the model outputs a 1280-dim
    embedding from the global average-pooling layer.
    """

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or DEVICE
        self._build_model()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_model(self) -> None:
        weights = models.EfficientNet_V2_S_Weights.DEFAULT
        full_model = models.efficientnet_v2_s(weights=weights)

        # Keep everything up to and including avgpool; drop the classifier.
        self.model = nn.Sequential(
            full_model.features,
            full_model.avgpool,
            nn.Flatten(1),
        )
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    @staticmethod
    def get_transform() -> transforms.Compose:
        """Return the standard inference transform pipeline."""
        return transforms.Compose(
            [
                transforms.Resize(232),
                transforms.CenterCrop(CONFIG.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _preprocess(self, image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        """Convert a single PIL image or tensor to a batched tensor."""
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
            tensor = self.get_transform()(image)
        else:
            tensor = image
        # Ensure 4-D: (1, C, H, W)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def extract(self, image: Union[Image.Image, torch.Tensor]) -> np.ndarray:
        """Return a 1280-dim embedding for a single image.

        Parameters
        ----------
        image : PIL.Image.Image | torch.Tensor
            A single RGB image (PIL) or a pre-processed tensor.

        Returns
        -------
        np.ndarray
            1-D array of shape ``(1280,)``.
        """
        tensor = self._preprocess(image)
        embedding = self.model(tensor)
        return embedding.cpu().numpy().flatten()

    @torch.no_grad()
    def extract_batch(self, images: list[Union[Image.Image, torch.Tensor]]) -> np.ndarray:
        """Return embeddings for a list of images.

        Parameters
        ----------
        images : list[PIL.Image.Image | torch.Tensor]
            A list of RGB images or pre-processed tensors.

        Returns
        -------
        np.ndarray
            Array of shape ``(N, 1280)``.
        """
        transform = self.get_transform()
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
        embeddings = self.model(batch)
        return embeddings.cpu().numpy()

    # ------------------------------------------------------------------
    # Fine-tuning helpers
    # ------------------------------------------------------------------

    def freeze(self) -> None:
        """Freeze all backbone parameters (no gradient computation)."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int = 3) -> None:
        """Unfreeze the last *n* feature blocks for fine-tuning.

        EfficientNet-V2-S ``features`` is an ``nn.Sequential`` of blocks.
        This method freezes everything first, then unfreezes the last *n*
        blocks inside ``features``.
        """
        self.freeze()
        features = self.model[0]  # nn.Sequential of feature blocks
        total_blocks = len(features)
        for idx in range(max(0, total_blocks - n), total_blocks):
            for param in features[idx].parameters():
                param.requires_grad = True
