"""Crop detected objects from frames."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.core.config import CONFIG

logger = logging.getLogger(__name__)


class ObjectCropper:
    """Crop, pad, resize, and optionally segment detected objects from frames."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def crop_detection(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
        padding: float | None = None,
    ) -> np.ndarray:
        """Crop the bounding-box region from *image* with padding.

        Parameters
        ----------
        image:
            BGR numpy array.
        bbox:
            ``(x1, y1, x2, y2)`` in pixel coordinates.
        padding:
            Fraction of the bbox dimensions to add as padding on each
            side (e.g. 0.10 = 10 %).  Defaults to ``CONFIG.crop_padding``.

        Returns
        -------
        np.ndarray
            The cropped region (clamped to image bounds).
        """
        if padding is None:
            padding = CONFIG.crop_padding

        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox

        bw = x2 - x1
        bh = y2 - y1
        pad_x = int(bw * padding)
        pad_y = int(bh * padding)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        return image[y1:y2, x1:x2].copy()

    def crop_and_resize(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
        size: int | None = None,
        padding: float | None = None,
    ) -> np.ndarray:
        """Crop the detection region and resize to a square.

        Parameters
        ----------
        image:
            BGR numpy array.
        bbox:
            ``(x1, y1, x2, y2)`` in pixel coordinates.
        size:
            Target side length in pixels.  Defaults to ``CONFIG.crop_size``.
        padding:
            Padding fraction.  Defaults to ``CONFIG.crop_padding``.

        Returns
        -------
        np.ndarray
            Square image of shape ``(size, size, 3)``.
        """
        if size is None:
            size = CONFIG.crop_size

        crop = self.crop_detection(image, bbox, padding=padding)
        resized = cv2.resize(
            crop, (size, size), interpolation=cv2.INTER_LANCZOS4
        )
        return resized

    def process_frame(
        self,
        image: np.ndarray,
        detections,
        segmenter=None,
        size: int | None = None,
        padding: float | None = None,
    ) -> list[np.ndarray]:
        """Process all detections in a single frame.

        For each detection the region is cropped (with padding) and
        resized.  If a *segmenter* is provided and available, the
        background is removed before resizing.

        Parameters
        ----------
        image:
            BGR numpy array of the full frame.
        detections:
            Iterable of objects with a ``bbox`` attribute (e.g.
            :class:`~src.video.object_detector.Detection`).
        segmenter:
            Optional :class:`~src.video.object_segmenter.ObjectSegmenter`
            instance.  If ``None`` or not available, background removal
            is skipped.
        size:
            Target side length for the output crops.  Defaults to
            ``CONFIG.crop_size``.
        padding:
            Padding fraction.  Defaults to ``CONFIG.crop_padding``.

        Returns
        -------
        list[np.ndarray]
            One square crop per detection, in the same order.
        """
        if size is None:
            size = CONFIG.crop_size
        if padding is None:
            padding = CONFIG.crop_padding

        crops: list[np.ndarray] = []

        for det in detections:
            bbox = det.bbox

            if segmenter is not None and segmenter.available:
                mask = segmenter.segment(image, bbox)
                if mask is not None:
                    masked_image = segmenter.apply_mask(image, mask)
                else:
                    masked_image = image
            else:
                masked_image = image

            crop = self.crop_detection(masked_image, bbox, padding=padding)
            resized = cv2.resize(
                crop, (size, size), interpolation=cv2.INTER_LANCZOS4
            )
            crops.append(resized)

        logger.debug(
            "Processed %d detections from frame (size=%d, padding=%.2f)",
            len(crops),
            size,
            padding,
        )
        return crops
