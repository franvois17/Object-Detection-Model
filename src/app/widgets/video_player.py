"""Video player widget for previewing uploaded videos."""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QTimer, Signal, Slot, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class VideoPlayer(QWidget):
    frame_changed = Signal(int, np.ndarray)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._cap: cv2.VideoCapture | None = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._next_frame)
        self._playing = False
        self._total_frames = 0
        self._current_frame = 0

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._display = QLabel()
        self._display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._display.setMinimumSize(480, 360)
        self._display.setStyleSheet("background-color: #11111b; border-radius: 8px;")
        layout.addWidget(self._display, 1)

        controls = QHBoxLayout()
        self._btn_play = QPushButton("Reproducir")
        self._btn_play.clicked.connect(self._toggle_play)
        controls.addWidget(self._btn_play)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.sliderMoved.connect(self._seek)
        controls.addWidget(self._slider, 1)

        self._lbl_time = QLabel("0 / 0")
        self._lbl_time.setFixedWidth(100)
        controls.addWidget(self._lbl_time)

        layout.addLayout(controls)

    def load_video(self, path: str) -> bool:
        self.stop()
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            return False
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 30
        self._timer.setInterval(int(1000 / fps))
        self._slider.setMaximum(max(0, self._total_frames - 1))
        self._current_frame = 0
        self._show_frame()
        return True

    def stop(self) -> None:
        self._timer.stop()
        self._playing = False
        self._btn_play.setText("Reproducir")
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @Slot()
    def _toggle_play(self) -> None:
        if self._cap is None:
            return
        if self._playing:
            self._timer.stop()
            self._playing = False
            self._btn_play.setText("Reproducir")
        else:
            self._timer.start()
            self._playing = True
            self._btn_play.setText("Pausar")

    @Slot()
    def _next_frame(self) -> None:
        if self._cap is None:
            return
        ret, frame = self._cap.read()
        if not ret:
            self._timer.stop()
            self._playing = False
            self._btn_play.setText("Reproducir")
            return
        self._current_frame = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        self._display_frame(frame)
        self._slider.setValue(self._current_frame)
        self._lbl_time.setText(f"{self._current_frame} / {self._total_frames}")
        self.frame_changed.emit(self._current_frame, frame)

    @Slot(int)
    def _seek(self, pos: int) -> None:
        if self._cap is None:
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        self._current_frame = pos
        self._show_frame()

    def _show_frame(self) -> None:
        if self._cap is None:
            return
        ret, frame = self._cap.read()
        if ret:
            self._display_frame(frame)
            self._lbl_time.setText(f"{self._current_frame} / {self._total_frames}")
            # Seek back so next read gets the right frame
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._current_frame)

    def _display_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        scaled = pixmap.scaled(
            self._display.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._display.setPixmap(scaled)
