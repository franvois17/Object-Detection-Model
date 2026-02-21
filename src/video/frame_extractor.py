"""Frame extraction from video files using OpenCV."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from src.core.config import CONFIG

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]


class FrameExtractor:
    """Extract frames from video files using uniform sampling or scene-change detection."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_uniform(
        self,
        video_path: str | Path,
        fps: float | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> list[tuple[int, np.ndarray]]:
        """Sample frames at a uniform interval.

        Parameters
        ----------
        video_path:
            Path to the source video file.
        fps:
            Desired output frame rate.  For example, ``1.0`` means one
            frame per second regardless of the video's native rate.
            Defaults to ``CONFIG.frame_sample_fps``.
        on_progress:
            Optional callback ``(current_frame, total_frames) -> None``
            invoked after each source frame is inspected.

        Returns
        -------
        list[tuple[int, np.ndarray]]
            Pairs of *(frame_index, BGR image array)*.
        """
        if fps is None:
            fps = CONFIG.frame_sample_fps

        cap = self._open_capture(video_path)
        try:
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            interval = max(1, int(round(src_fps / fps)))

            frames: list[tuple[int, np.ndarray]] = []
            idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if idx % interval == 0:
                    frames.append((idx, frame))

                idx += 1
                if on_progress is not None:
                    on_progress(idx, total_frames)

            logger.info(
                "Uniform extraction: %d frames sampled from %d total "
                "(interval=%d, target_fps=%.2f)",
                len(frames),
                total_frames,
                interval,
                fps,
            )
            return frames
        finally:
            cap.release()

    def extract_scene_change(
        self,
        video_path: str | Path,
        threshold: float | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> list[tuple[int, np.ndarray]]:
        """Extract frames where a scene change is detected.

        Scene changes are identified by comparing the normalised
        histograms of consecutive frames.  When the histogram
        correlation drops below ``1 - threshold`` a new scene is
        assumed.

        Parameters
        ----------
        video_path:
            Path to the source video file.
        threshold:
            Sensitivity (0-1).  A *lower* value requires a larger visual
            change before a frame is emitted.  Defaults to
            ``CONFIG.min_scene_change``.
        on_progress:
            Optional callback ``(current_frame, total_frames) -> None``.

        Returns
        -------
        list[tuple[int, np.ndarray]]
            Pairs of *(frame_index, BGR image array)*.
        """
        if threshold is None:
            threshold = CONFIG.min_scene_change

        cap = self._open_capture(video_path)
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frames: list[tuple[int, np.ndarray]] = []
            prev_hist: np.ndarray | None = None
            idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                hist = self._compute_histogram(frame)

                if prev_hist is None:
                    # Always include the very first frame.
                    frames.append((idx, frame))
                else:
                    similarity = cv2.compareHist(
                        prev_hist, hist, cv2.HISTCMP_CORREL
                    )
                    change = 1.0 - similarity
                    if change > threshold:
                        frames.append((idx, frame))

                prev_hist = hist
                idx += 1
                if on_progress is not None:
                    on_progress(idx, total_frames)

            logger.info(
                "Scene-change extraction: %d frames detected from %d total "
                "(threshold=%.2f)",
                len(frames),
                total_frames,
                threshold,
            )
            return frames
        finally:
            cap.release()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _open_capture(video_path: str | Path) -> cv2.VideoCapture:
        path = str(Path(video_path).resolve())
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(
                f"Cannot open video file: {path}"
            )
        return cap

    @staticmethod
    def _compute_histogram(frame: np.ndarray) -> np.ndarray:
        """Compute a normalised HSV histogram for *frame*."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1], None, [50, 60], [0, 180, 0, 256]
        )
        cv2.normalize(hist, hist)
        return hist
