"""Camera capture and real-time inference worker."""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np
from PySide6.QtCore import Signal

from src.core.config import CONFIG, Config
from src.core.device import DEVICE
from src.data.database import DatabaseManager, get_all_products
from src.ml.backbone import FeatureExtractor
from src.ml.embeddings import EmbeddingIndex
from src.video.object_detector import ObjectDetector
from src.workers.base_worker import BaseWorker

logger = logging.getLogger(__name__)


class CameraWorker(BaseWorker):
    """Capture frames from a camera and run object detection + inference.

    Signals (in addition to those inherited from :class:`BaseWorker`)
    -----------------------------------------------------------------
    frame_ready : Signal(object)
        Emitted for every captured frame.  The payload is a BGR numpy
        array with overlay annotations drawn on it.
    detection_ready : Signal(list)
        Emitted when an inference cycle completes.  The payload is a
        list of dicts, each containing ``bbox``, ``product_name``,
        ``confidence`` and ``product_id``.
    """

    frame_ready = Signal(object)
    detection_ready = Signal(list)

    def __init__(
        self,
        camera_index: int = 0,
        config: Config | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._camera_index = camera_index
        self._config = config or CONFIG

    # ------------------------------------------------------------------
    # Worker implementation
    # ------------------------------------------------------------------

    def do_work(self) -> dict:
        """Run the continuous capture-and-detect loop.

        Returns
        -------
        dict
            Summary statistics (``total_frames``, ``total_detections``).
        """
        config = self._config

        # 1. Open camera
        self.progress.emit(0, "Opening camera...")
        cap = cv2.VideoCapture(self._camera_index)
        if not cap.isOpened():
            msg = f"Cannot open camera at index {self._camera_index}"
            logger.error(msg)
            self.error.emit(msg)
            return {"total_frames": 0, "total_detections": 0}

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)
        cap.set(cv2.CAP_PROP_FPS, config.camera_fps)

        try:
            # 2. Load models
            self.progress.emit(10, "Loading detection model...")
            detector = ObjectDetector()

            self.progress.emit(20, "Loading feature extractor...")
            backbone = FeatureExtractor(device=DEVICE)

            self.progress.emit(30, "Loading FAISS index...")
            index = self._load_index()

            # Load product names from database
            self.progress.emit(40, "Loading product names...")
            product_names = self._load_product_names()

            self.progress.emit(50, "Starting live capture...")

            # State for temporal smoothing
            frame_count = 0
            total_detections = 0
            recent_detections: list[list[dict]] = []
            last_results: list[dict] = []

            # 3. Main loop
            while not self.is_cancelled:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Camera read failed -- retrying...")
                    time.sleep(0.05)
                    continue

                frame_count += 1

                # Run detection + inference every N frames
                if frame_count % config.inference_skip_frames == 0:
                    detections = detector.detect(frame)
                    results = self._classify_detections(
                        frame, detections, backbone, index, product_names
                    )

                    # If YOLO found nothing, classify center crop of full frame
                    if not results and index is not None and len(index) > 0:
                        center_result = self._classify_full_frame(
                            frame, backbone, index, product_names
                        )
                        if center_result:
                            results = [center_result]

                    # Temporal smoothing: keep a sliding window
                    recent_detections.append(results)
                    if len(recent_detections) > config.temporal_window:
                        recent_detections.pop(0)

                    last_results = self._smooth_detections(recent_detections)
                    total_detections += len(last_results)

                    self.detection_ready.emit(last_results)

                # Draw overlay and emit frame
                annotated = self._draw_overlay(frame, last_results)
                self.frame_ready.emit(annotated)

        finally:
            cap.release()
            logger.info(
                "Camera released. Captured %d frames, %d detections.",
                frame_count,
                total_detections,
            )

        return {
            "total_frames": frame_count,
            "total_detections": total_detections,
        }

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _classify_full_frame(
        self,
        frame: np.ndarray,
        backbone: FeatureExtractor,
        index: EmbeddingIndex,
        product_names: dict[int, str],
    ) -> dict | None:
        """Classify the center crop of the full frame (fallback when YOLO finds nothing)."""
        from src.video.cropper import ObjectCropper
        from PIL import Image

        cropper = ObjectCropper()
        h, w = frame.shape[:2]
        # Center square crop
        side = min(h, w)
        x1 = (w - side) // 2
        y1 = (h - side) // 2
        crop = cropper.crop_and_resize(frame, bbox=(x1, y1, x1 + side, y1 + side), padding=0.0)

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(crop_rgb)

        embedding = backbone.extract(pil_image)
        neighbours = index.search(embedding, k=self._config.knn_k)

        if not neighbours:
            return None

        product_id, similarity = neighbours[0]
        name = product_names.get(product_id, f"Product {product_id}")
        if similarity < self._config.confidence_threshold:
            name = "Desconocido"

        return {
            "bbox": (x1, y1, x1 + side, y1 + side),
            "product_name": name,
            "confidence": similarity,
            "product_id": product_id,
        }

    def _load_index(self) -> EmbeddingIndex | None:
        """Load the FAISS product index, returning ``None`` if absent."""
        index_path = self._config.embeddings_dir / "product_index.faiss"
        if not index_path.exists():
            logger.warning("No FAISS index found at %s", index_path)
            return None

        index = EmbeddingIndex(dim=self._config.embedding_dim)
        index.load(index_path)
        logger.info("Loaded FAISS index with %d vectors", len(index))
        return index

    def _load_product_names(self) -> dict[int, str]:
        """Load product names from the database."""
        try:
            db = DatabaseManager()
            with db as session:
                products = get_all_products(session)
                return {p.id: p.name for p in products}
        except Exception:
            logger.warning("Could not load product names from database")
            return {}

    def _classify_detections(
        self,
        frame: np.ndarray,
        detections,
        backbone: FeatureExtractor,
        index: EmbeddingIndex | None,
        product_names: dict[int, str] | None = None,
    ) -> list[dict]:
        """Crop each detection, extract embeddings, query the index."""
        if not detections or index is None or len(index) == 0:
            return []

        from src.video.cropper import ObjectCropper
        from PIL import Image

        product_names = product_names or {}
        cropper = ObjectCropper()
        results: list[dict] = []

        for det in detections:
            crop = cropper.crop_and_resize(frame, det.bbox)
            # Convert BGR crop to PIL RGB
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(crop_rgb)

            embedding = backbone.extract(pil_image)
            neighbours = index.search(embedding, k=self._config.knn_k)

            if neighbours:
                product_id, similarity = neighbours[0]
                name = product_names.get(product_id, f"Product {product_id}")
                if similarity < self._config.confidence_threshold:
                    name = "Desconocido"
                results.append({
                    "bbox": det.bbox,
                    "product_name": name,
                    "confidence": similarity,
                    "product_id": product_id,
                })

        return results

    # ------------------------------------------------------------------
    # Temporal smoothing
    # ------------------------------------------------------------------

    @staticmethod
    def _smooth_detections(
        recent: list[list[dict]],
    ) -> list[dict]:
        """Apply simple temporal smoothing across the sliding window.

        Detections that appear in the majority of recent windows are
        kept.  The confidence is averaged across appearances.
        """
        if not recent:
            return []

        # Use the latest window as the candidate set
        latest = recent[-1]
        if not latest or len(recent) < 2:
            return latest

        smoothed: list[dict] = []
        threshold = len(recent) / 2.0

        for det in latest:
            pid = det["product_id"]
            # Count appearances in the window
            appearances = 0
            total_conf = 0.0
            for window in recent:
                for d in window:
                    if d["product_id"] == pid:
                        appearances += 1
                        total_conf += d["confidence"]
                        break

            if appearances >= threshold:
                smoothed.append({
                    "bbox": det["bbox"],
                    "product_name": det["product_name"],
                    "confidence": total_conf / appearances,
                    "product_id": det["product_id"],
                })

        return smoothed

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_overlay(
        frame: np.ndarray, detections: list[dict]
    ) -> np.ndarray:
        """Draw bounding boxes, product names and confidence on the frame.

        Parameters
        ----------
        frame:
            BGR numpy array (will be copied before drawing).
        detections:
            List of result dicts with ``bbox``, ``product_name``,
            ``confidence``.

        Returns
        -------
        np.ndarray
            A copy of *frame* with annotations drawn.
        """
        annotated = frame.copy()
        color = (0, 255, 0)  # Green in BGR
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Label background
            label = f"{det['product_name']} {det['confidence']:.0%}"
            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            label_y = max(y1 - 10, text_h + 5)
            cv2.rectangle(
                annotated,
                (x1, label_y - text_h - 5),
                (x1 + text_w + 5, label_y + baseline),
                color,
                cv2.FILLED,
            )
            # Label text
            cv2.putText(
                annotated,
                label,
                (x1 + 2, label_y - 2),
                font,
                font_scale,
                (0, 0, 0),  # Black text on green bg
                thickness,
                cv2.LINE_AA,
            )

        return annotated
