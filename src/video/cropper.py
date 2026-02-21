"""Crop detected objects from frames with optional background removal."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.core.config import CONFIG

logger = logging.getLogger(__name__)


class ObjectCropper:
    """Crop, pad, resize, and optionally remove background from detected objects."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def crop_detection(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
        padding: float | None = None,
    ) -> np.ndarray:
        """Crop the bounding-box region from *image* with padding."""
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
        """Crop the detection region and resize to a square."""
        if size is None:
            size = CONFIG.crop_size

        crop = self.crop_detection(image, bbox, padding=padding)
        resized = cv2.resize(
            crop, (size, size), interpolation=cv2.INTER_LANCZOS4
        )
        return resized

    @staticmethod
    def remove_background(
        image: np.ndarray,
        bbox: tuple[int, int, int, int] | None = None,
        bg_color: tuple[int, int, int] = (255, 255, 255),
        iterations: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Remove background using GrabCut.

        Parameters
        ----------
        image:
            BGR numpy array.
        bbox:
            Optional ``(x1, y1, x2, y2)`` hint for GrabCut. If None,
            uses a centered rectangle covering 80% of the image.
        bg_color:
            BGR color for the background replacement.
        iterations:
            Number of GrabCut iterations (more = better but slower).

        Returns
        -------
        (result_image, mask)
            The image with background replaced and the binary mask (0/255).
        """
        h, w = image.shape[:2]

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            # Clamp and ensure minimum size
            x1, y1 = max(1, x1), max(1, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            rect = (x1, y1, x2 - x1, y2 - y1)
        else:
            # Use centered 80% rectangle
            mx, my = int(w * 0.1), int(h * 0.1)
            rect = (mx, my, w - 2 * mx, h - 2 * my)

        # Ensure rect has valid dimensions
        if rect[2] < 10 or rect[3] < 10:
            # Image too small for GrabCut, return as-is
            mask = np.ones((h, w), dtype=np.uint8) * 255
            return image.copy(), mask

        gc_mask = np.zeros((h, w), dtype=np.uint8)
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)

        try:
            cv2.grabCut(
                image, gc_mask, rect,
                bgd_model, fgd_model,
                iterations,
                cv2.GC_INIT_WITH_RECT,
            )
        except cv2.error:
            logger.warning("GrabCut failed, returning original image")
            mask = np.ones((h, w), dtype=np.uint8) * 255
            return image.copy(), mask

        # Foreground = definite fg (1) or probable fg (3)
        fg_mask = ((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)).astype(np.uint8)

        # Clean up mask with morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Apply mask
        binary_mask = fg_mask * 255
        bg = np.full_like(image, bg_color, dtype=np.uint8)
        result = np.where(fg_mask[:, :, None] == 1, image, bg)

        return result, binary_mask

    def crop_and_remove_bg(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
        size: int | None = None,
        padding: float | None = None,
        bg_color: tuple[int, int, int] = (255, 255, 255),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Crop, remove background, and resize.

        Returns
        -------
        (crop_image, mask)
            Both resized to (size, size).
        """
        if size is None:
            size = CONFIG.crop_size

        crop = self.crop_detection(image, bbox, padding=padding)

        # Run GrabCut on the crop (no bbox hint = use center rectangle)
        clean, mask = self.remove_background(crop, bg_color=bg_color)

        resized = cv2.resize(clean, (size, size), interpolation=cv2.INTER_LANCZOS4)
        mask_resized = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)

        return resized, mask_resized

    def process_frame(
        self,
        image: np.ndarray,
        detections,
        segmenter=None,
        size: int | None = None,
        padding: float | None = None,
        remove_bg: bool = False,
    ) -> list[np.ndarray]:
        """Process all detections in a single frame.

        Parameters
        ----------
        image:
            BGR numpy array of the full frame.
        detections:
            Iterable of objects with a ``bbox`` attribute.
        segmenter:
            Optional SAM2 segmenter (used if available and remove_bg is False).
        size:
            Target side length for output crops.
        padding:
            Padding fraction.
        remove_bg:
            If True, use GrabCut to remove background from each crop.
        """
        if size is None:
            size = CONFIG.crop_size
        if padding is None:
            padding = CONFIG.crop_padding

        crops: list[np.ndarray] = []

        for det in detections:
            bbox = det.bbox

            if remove_bg:
                clean, _ = self.crop_and_remove_bg(
                    image, bbox, size=size, padding=padding
                )
                crops.append(clean)
            elif segmenter is not None and segmenter.available:
                mask = segmenter.segment(image, bbox)
                if mask is not None:
                    masked_image = segmenter.apply_mask(image, mask)
                else:
                    masked_image = image
                crop = self.crop_detection(masked_image, bbox, padding=padding)
                resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LANCZOS4)
                crops.append(resized)
            else:
                crop = self.crop_detection(image, bbox, padding=padding)
                resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LANCZOS4)
                crops.append(resized)

        logger.debug(
            "Processed %d detections from frame (size=%d, padding=%.2f, remove_bg=%s)",
            len(crops), size, padding, remove_bg,
        )
        return crops
