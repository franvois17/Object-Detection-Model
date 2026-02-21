"""Live camera feed display widget."""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class CameraView(QWidget):
    """Widget that displays camera frames received via update_frame."""

    clicked = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._display = QLabel()
        self._display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._display.setMinimumSize(640, 480)
        self._display.setStyleSheet("background-color: #11111b; border-radius: 8px;")
        layout.addWidget(self._display)

        self._placeholder_text = "Camara no iniciada"
        self._show_placeholder()

    def _show_placeholder(self) -> None:
        self._display.setText(self._placeholder_text)
        self._display.setStyleSheet(
            "background-color: #11111b; border-radius: 8px; "
            "color: #6c7086; font-size: 18px;"
        )

    def update_frame(self, frame: np.ndarray) -> None:
        """Update display with a BGR numpy frame."""
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

    def clear(self) -> None:
        self._display.clear()
        self._show_placeholder()

    def mousePressEvent(self, event) -> None:
        self.clicked.emit()
        super().mousePressEvent(event)
