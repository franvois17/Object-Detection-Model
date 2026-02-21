"""Review page - inspect and curate cropped images before training."""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.app.widgets.image_gallery import ImageGallery
from src.data.database import (
    DatabaseManager,
    get_all_products,
    get_product_images,
    update_product,
)
from src.data.storage import StorageManager


class ReviewPage(QWidget):
    def __init__(self, db: DatabaseManager, parent=None) -> None:
        super().__init__(parent)
        self._db = db
        self._storage = StorageManager()
        self._current_product_id: int | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        title = QLabel("Revisar Recortes")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel(
            "Deselecciona las imagenes de mala calidad antes de entrenar. "
            "Solo las imagenes seleccionadas se usaran para el modelo."
        )
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        # Product selector
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Producto:"))
        self._product_combo = QComboBox()
        self._product_combo.setMinimumWidth(300)
        self._product_combo.currentIndexChanged.connect(self._on_product_changed)
        selector_layout.addWidget(self._product_combo)

        self._btn_refresh = QPushButton("Actualizar")
        self._btn_refresh.clicked.connect(self._load_products)
        selector_layout.addWidget(self._btn_refresh)
        selector_layout.addStretch()
        layout.addLayout(selector_layout)

        # Selection controls
        controls = QHBoxLayout()
        btn_all = QPushButton("Seleccionar Todos")
        btn_all.clicked.connect(lambda: self._gallery.select_all())
        controls.addWidget(btn_all)

        btn_none = QPushButton("Deseleccionar Todos")
        btn_none.clicked.connect(lambda: self._gallery.deselect_all())
        controls.addWidget(btn_none)

        controls.addStretch()

        self._count_label = QLabel("")
        controls.addWidget(self._count_label)

        self._btn_delete = QPushButton("Eliminar Deseleccionados")
        self._btn_delete.setObjectName("danger")
        self._btn_delete.clicked.connect(self._delete_deselected)
        controls.addWidget(self._btn_delete)

        layout.addLayout(controls)

        # Image gallery
        self._gallery = ImageGallery(columns=6, thumb_size=140)
        self._gallery.selection_changed.connect(self._update_count)
        layout.addWidget(self._gallery, 1)

    def on_activated(self) -> None:
        self._load_products()

    def _load_products(self) -> None:
        self._product_combo.blockSignals(True)
        self._product_combo.clear()
        with self._db as session:
            products = get_all_products(session)
            for p in products:
                self._product_combo.addItem(
                    f"{p.name} ({p.image_count} imgs)", p.id
                )
        self._product_combo.blockSignals(False)
        if self._product_combo.count() > 0:
            self._on_product_changed(0)

    @Slot(int)
    def _on_product_changed(self, index: int) -> None:
        if index < 0:
            return
        product_id = self._product_combo.itemData(index)
        self._current_product_id = product_id
        self._load_images(product_id)

    def _load_images(self, product_id: int) -> None:
        with self._db as session:
            images = get_product_images(session, product_id)
            paths = [img.file_path for img in images if os.path.exists(img.file_path)]
        self._gallery.set_images(paths)
        self._update_count()

    def _update_count(self) -> None:
        total = self._gallery.count
        selected = self._gallery.selected_count
        self._count_label.setText(f"{selected} / {total} seleccionados")

    @Slot()
    def _delete_deselected(self) -> None:
        deselected = self._gallery.get_deselected_paths()
        if not deselected:
            return

        reply = QMessageBox.question(
            self,
            "Confirmar",
            f"Eliminar {len(deselected)} imagenes deseleccionadas?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Delete files and DB records
        for path in deselected:
            try:
                os.remove(path)
            except OSError:
                pass

        # Update DB image count
        if self._current_product_id is not None:
            with self._db as session:
                remaining = self._gallery.selected_count
                update_product(
                    session, self._current_product_id, image_count=remaining
                )

        # Reload
        if self._current_product_id is not None:
            self._load_images(self._current_product_id)
        self._load_products()
