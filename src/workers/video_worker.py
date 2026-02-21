"""Video processing worker -- frame extraction, detection and cropping."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
from PySide6.QtCore import Signal

from src.core.config import CONFIG, Config
from src.data.database import DatabaseManager, add_product_image, update_product
from src.video.frame_extractor import FrameExtractor
from src.video.object_detector import ObjectDetector
from src.video.object_segmenter import ObjectSegmenter
from src.video.cropper import ObjectCropper
from src.workers.base_worker import BaseWorker

logger = logging.getLogger(__name__)


class VideoProcessingWorker(BaseWorker):
    """Extract frames from a video, detect objects, crop them and persist.

    Signals (in addition to those inherited from :class:`BaseWorker`)
    -----------------------------------------------------------------
    frames_extracted : Signal(int)
        Emitted once frame extraction is complete.  The payload is the
        number of frames that were sampled from the video.
    crops_ready : Signal(list)
        Emitted when all crops have been saved.  The payload is a list
        of absolute file-path strings pointing to the saved images.
    """

    frames_extracted = Signal(int)
    crops_ready = Signal(list)

    def __init__(
        self,
        video_path: str,
        product_id: int,
        product_name: str,
        config: Config | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._video_path = video_path
        self._product_id = product_id
        self._product_name = product_name
        self._config = config or CONFIG

    # ------------------------------------------------------------------
    # Worker implementation
    # ------------------------------------------------------------------

    def do_work(self) -> dict:
        """Run the full video-processing pipeline.

        Returns
        -------
        dict
            ``product_id``, ``num_frames``, ``num_crops``, ``crop_paths``.
        """
        config = self._config

        # 1. Frame extraction (0 -- 30 %)
        self.progress.emit(0, "Extracting frames from video...")
        extractor = FrameExtractor()

        def _frame_progress(current: int, total: int) -> None:
            if total > 0:
                pct = int(30 * current / total)
                self.progress.emit(pct, f"Extracting frames: {current}/{total}")

        frames = extractor.extract_uniform(
            self._video_path,
            fps=config.frame_sample_fps,
            on_progress=_frame_progress,
        )

        num_frames = len(frames)
        self.frames_extracted.emit(num_frames)
        logger.info("Extracted %d frames from %s", num_frames, self._video_path)

        if self.is_cancelled:
            return self._empty_result()

        # 2. Object detection (30 -- 70 %)
        self.progress.emit(30, "Detecting objects in frames...")
        detector = ObjectDetector()

        all_detections: list[list] = []
        for i, (_idx, frame) in enumerate(frames):
            if self.is_cancelled:
                return self._empty_result()

            detections = detector.detect(frame)
            all_detections.append(detections)

            pct = 30 + int(40 * (i + 1) / num_frames) if num_frames > 0 else 70
            self.progress.emit(pct, f"Detecting objects: frame {i + 1}/{num_frames}")

        if self.is_cancelled:
            return self._empty_result()

        # 3. Cropping + background removal (70 -- 90 %)
        use_bg_removal = config.remove_background
        msg = "Cropping and removing background..." if use_bg_removal else "Cropping detections..."
        self.progress.emit(70, msg)
        cropper = ObjectCropper()

        segmenter = None
        if config.use_sam2:
            segmenter = ObjectSegmenter()

        all_crops: list = []
        for i, ((_idx, frame), detections) in enumerate(zip(frames, all_detections)):
            if self.is_cancelled:
                return self._empty_result()

            if detections:
                crops = cropper.process_frame(
                    frame, detections, segmenter=segmenter,
                    remove_bg=use_bg_removal,
                )
                all_crops.extend(crops)
            else:
                # No YOLO detection: use center crop of the full frame
                if use_bg_removal:
                    clean, _ = cropper.remove_background(frame)
                    crop = cv2.resize(clean, (config.crop_size, config.crop_size),
                                      interpolation=cv2.INTER_LANCZOS4)
                else:
                    crop = cropper.crop_and_resize(
                        frame,
                        bbox=(0, 0, frame.shape[1], frame.shape[0]),
                        padding=0.0,
                    )
                all_crops.append(crop)

            total = len(frames)
            pct = 70 + int(20 * (i + 1) / total) if total > 0 else 90
            self.progress.emit(pct, f"Cropping: frame {i + 1}/{total}")

        if self.is_cancelled:
            return self._empty_result()

        # 4. Save crops (90 -- 100 %)
        self.progress.emit(90, "Saving crop images...")
        crop_paths = self._save_crops(all_crops)

        self.progress.emit(100, "Video processing complete.")
        self.crops_ready.emit(crop_paths)

        return {
            "product_id": self._product_id,
            "num_frames": num_frames,
            "num_crops": len(crop_paths),
            "crop_paths": crop_paths,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_crops(self, crops: list) -> list[str]:
        """Save crop images to disk and register them in the database.

        Returns a list of absolute file-path strings.
        """
        product_dir = (
            self._config.images_dir / str(self._product_id)
        )
        product_dir.mkdir(parents=True, exist_ok=True)

        paths: list[str] = []
        total = len(crops)

        db = DatabaseManager()

        for i, crop in enumerate(crops):
            filename = f"{self._product_name}_{i:04d}.png"
            filepath = product_dir / filename
            cv2.imwrite(str(filepath), crop)
            path_str = str(filepath)
            paths.append(path_str)

            # Register in database
            with db as session:
                add_product_image(
                    session,
                    product_id=self._product_id,
                    file_path=path_str,
                    source_video=self._video_path,
                    frame_index=i,
                )

            if total > 0:
                pct = 90 + int(10 * (i + 1) / total)
                self.progress.emit(pct, f"Saving crop {i + 1}/{total}")

        logger.info("Saved %d crops to %s", len(paths), product_dir)
        return paths

    def _empty_result(self) -> dict:
        """Return an empty result dict when work is cancelled."""
        return {
            "product_id": self._product_id,
            "num_frames": 0,
            "num_crops": 0,
            "crop_paths": [],
        }
