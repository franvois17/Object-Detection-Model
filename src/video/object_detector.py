"""YOLOv8 object detection wrapper."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from src.core.config import CONFIG
from src.core.device import DEVICE

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Detection:
    """A single detected object in an image.

    Attributes
    ----------
    bbox:
        Bounding box as ``(x1, y1, x2, y2)`` in pixel coordinates.
    confidence:
        Model confidence score in ``[0, 1]``.
    class_name:
        Human-readable class name predicted by the model.
    """

    bbox: tuple[int, int, int, int]
    confidence: float
    class_name: str


class ObjectDetector:
    """Thin wrapper around a YOLOv8 model for single-image and batch detection."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        if model_path is None:
            model_path = CONFIG.pretrained_dir / CONFIG.yolo_model
            # Fall back to the bare model name so Ultralytics can
            # download it automatically if not cached locally.
            if not Path(model_path).exists():
                model_path = CONFIG.yolo_model

        self._device = device or str(DEVICE)
        self._model = YOLO(str(model_path))
        logger.info(
            "YOLOv8 loaded from %s on device %s", model_path, self._device
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        image: np.ndarray,
        conf: float | None = None,
        iou: float | None = None,
    ) -> list[Detection]:
        """Run object detection on a single image.

        Parameters
        ----------
        image:
            BGR numpy array (OpenCV format).
        conf:
            Confidence threshold.  Defaults to ``CONFIG.yolo_conf_threshold``.
        iou:
            IoU threshold for NMS.  Defaults to ``CONFIG.yolo_iou_threshold``.

        Returns
        -------
        list[Detection]
            Detected objects sorted by descending confidence.
        """
        if conf is None:
            conf = CONFIG.yolo_conf_threshold
        if iou is None:
            iou = CONFIG.yolo_iou_threshold

        results = self._model.predict(
            source=image,
            conf=conf,
            iou=iou,
            device=self._device,
            verbose=False,
        )
        return self._parse_results(results)

    def detect_batch(
        self,
        images: list[np.ndarray],
        conf: float | None = None,
        iou: float | None = None,
    ) -> list[list[Detection]]:
        """Run object detection on a batch of images.

        Parameters
        ----------
        images:
            List of BGR numpy arrays.
        conf:
            Confidence threshold.  Defaults to ``CONFIG.yolo_conf_threshold``.
        iou:
            IoU threshold for NMS.  Defaults to ``CONFIG.yolo_iou_threshold``.

        Returns
        -------
        list[list[Detection]]
            One list of detections per input image.
        """
        if conf is None:
            conf = CONFIG.yolo_conf_threshold
        if iou is None:
            iou = CONFIG.yolo_iou_threshold

        if not images:
            return []

        results = self._model.predict(
            source=images,
            conf=conf,
            iou=iou,
            device=self._device,
            verbose=False,
        )

        batch_detections: list[list[Detection]] = []
        for result in results:
            batch_detections.append(self._parse_results([result]))

        logger.info(
            "Batch detection: %d images, %d total detections",
            len(images),
            sum(len(d) for d in batch_detections),
        )
        return batch_detections

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _parse_results(self, results) -> list[Detection]:
        """Convert Ultralytics results to a list of ``Detection``."""
        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self._model.names.get(class_id, str(class_id))
                detections.append(
                    Detection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=confidence,
                        class_name=class_name,
                    )
                )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
