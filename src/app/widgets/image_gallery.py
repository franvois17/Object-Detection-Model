"""Scrollable image gallery widget with selection support."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class ImageThumbnail(QWidget):
    """Single thumbnail with checkbox for selection."""

    toggled = Signal(str, bool)

    def __init__(self, image_path: str, size: int = 150, parent=None) -> None:
        super().__init__(parent)
        self.image_path = image_path
        self._size = size

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._img_label = QLabel()
        self._img_label.setFixedSize(size, size)
        self._img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_label.setStyleSheet(
            "background-color: #313244; border-radius: 6px;"
        )
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                size, size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._img_label.setPixmap(scaled)
        layout.addWidget(self._img_label)

        self._check = QCheckBox()
        self._check.setChecked(True)
        self._check.toggled.connect(lambda checked: self.toggled.emit(self.image_path, checked))
        layout.addWidget(self._check, alignment=Qt.AlignmentFlag.AlignCenter)

    @property
    def is_selected(self) -> bool:
        return self._check.isChecked()

    def set_selected(self, selected: bool) -> None:
        self._check.setChecked(selected)


class ImageGallery(QWidget):
    """Scrollable grid of image thumbnails with selection."""

    selection_changed = Signal()

    def __init__(self, columns: int = 5, thumb_size: int = 150, parent=None) -> None:
        super().__init__(parent)
        self._columns = columns
        self._thumb_size = thumb_size
        self._thumbnails: list[ImageThumbnail] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self._container = QWidget()
        self._grid = QGridLayout(self._container)
        self._grid.setSpacing(8)
        self._scroll.setWidget(self._container)

        layout.addWidget(self._scroll)

    def set_images(self, image_paths: list[str]) -> None:
        """Replace all images in the gallery."""
        self.clear()
        for i, path in enumerate(image_paths):
            thumb = ImageThumbnail(path, self._thumb_size)
            thumb.toggled.connect(lambda *_: self.selection_changed.emit())
            row, col = divmod(i, self._columns)
            self._grid.addWidget(thumb, row, col)
            self._thumbnails.append(thumb)

    def add_image(self, image_path: str) -> None:
        idx = len(self._thumbnails)
        thumb = ImageThumbnail(image_path, self._thumb_size)
        thumb.toggled.connect(lambda *_: self.selection_changed.emit())
        row, col = divmod(idx, self._columns)
        self._grid.addWidget(thumb, row, col)
        self._thumbnails.append(thumb)

    def get_selected_paths(self) -> list[str]:
        return [t.image_path for t in self._thumbnails if t.is_selected]

    def get_deselected_paths(self) -> list[str]:
        return [t.image_path for t in self._thumbnails if not t.is_selected]

    def select_all(self) -> None:
        for t in self._thumbnails:
            t.set_selected(True)

    def deselect_all(self) -> None:
        for t in self._thumbnails:
            t.set_selected(False)

    def clear(self) -> None:
        for t in self._thumbnails:
            self._grid.removeWidget(t)
            t.deleteLater()
        self._thumbnails.clear()

    @property
    def count(self) -> int:
        return len(self._thumbnails)

    @property
    def selected_count(self) -> int:
        return sum(1 for t in self._thumbnails if t.is_selected)
