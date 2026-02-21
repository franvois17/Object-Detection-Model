"""SAM2 segmentation wrapper (optional -- degrades gracefully)."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from src.core.config import CONFIG
from src.core.device import DEVICE

logger = logging.getLogger(__name__)

# Try to import SAM2.  If not installed the class will still be
# instantiable but ``available`` will return False.
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    _SAM2_AVAILABLE = True
except ImportError:
    _SAM2_AVAILABLE = False
    logger.debug("SAM2 is not installed -- segmentation will be unavailable.")


class ObjectSegmenter:
    """Segment objects from their background using SAM2.

    If the ``sam2`` package is not installed the instance remains usable
    but every method that requires the model will return ``None`` or a
    pass-through result and the :pyattr:`available` property will be
    ``False``.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        self._device = device or str(DEVICE)
        self._predictor: SAM2ImagePredictor | None = None  # type: ignore[name-defined]

        if not _SAM2_AVAILABLE:
            logger.warning(
                "SAM2 package not found.  ObjectSegmenter will be "
                "disabled.  Install sam2 to enable segmentation."
            )
            return

        if model_path is None:
            model_path = CONFIG.pretrained_dir / CONFIG.sam2_model
            if not Path(model_path).exists():
                logger.warning(
                    "SAM2 weights not found at %s -- segmenter disabled.",
                    model_path,
                )
                return

        try:
            sam2_model = build_sam2(
                config_file=None,
                ckpt_path=str(model_path),
                device=self._device,
            )
            self._predictor = SAM2ImagePredictor(sam2_model)
            logger.info(
                "SAM2 loaded from %s on device %s", model_path, self._device
            )
        except Exception:
            logger.exception("Failed to load SAM2 model -- segmenter disabled.")
            self._predictor = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """Return ``True`` if the SAM2 model is loaded and ready."""
        return self._predictor is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment(
        self, image: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> np.ndarray | None:
        """Produce a binary mask for the object inside *bbox*.

        Parameters
        ----------
        image:
            BGR numpy array (OpenCV format).
        bbox:
            Bounding box ``(x1, y1, x2, y2)`` in pixel coordinates.

        Returns
        -------
        np.ndarray | None
            A binary mask with the same height/width as *image* (dtype
            ``uint8``, values 0 or 255), or ``None`` if SAM2 is
            unavailable.
        """
        if not self.available:
            return None

        try:
            # SAM2 expects RGB input.
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self._predictor.set_image(rgb)

            input_box = np.array(bbox, dtype=np.float32)
            masks, scores, _ = self._predictor.predict(
                box=input_box[None, :],
                multimask_output=False,
            )

            # Take the mask with the highest score.
            best_idx = int(np.argmax(scores))
            mask = masks[best_idx].astype(np.uint8) * 255
            return mask
        except Exception:
            logger.exception("SAM2 segmentation failed for bbox %s", bbox)
            return None

    @staticmethod
    def apply_mask(
        image: np.ndarray,
        mask: np.ndarray,
        bg_color: tuple[int, int, int] = (0, 0, 0),
    ) -> np.ndarray:
        """Apply a binary mask to *image*, replacing the background.

        Parameters
        ----------
        image:
            BGR numpy array.
        mask:
            Binary mask (0/255) with the same spatial dimensions.
        bg_color:
            Background colour (B, G, R) to fill masked-out pixels.

        Returns
        -------
        np.ndarray
            The masked image with the background replaced.
        """
        mask_bool = mask > 127

        background = np.full_like(image, bg_color, dtype=np.uint8)
        result = np.where(mask_bool[..., None], image, background)
        return result
