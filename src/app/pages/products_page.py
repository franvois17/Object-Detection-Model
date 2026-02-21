"""Products page - list, edit, delete registered products."""

from __future__ import annotations

import os

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.data.database import (
    DatabaseManager,
    delete_product,
    get_all_products,
    get_product_images,
    update_product,
)
from src.data.storage import StorageManager


class ProductsPage(QWidget):
    def __init__(self, db: DatabaseManager, parent=None) -> None:
        super().__init__(parent)
        self._db = db
        self._storage = StorageManager()
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        title = QLabel("Productos Registrados")
        title.setObjectName("title")
        layout.addWidget(title)

        # Search + actions
        top = QHBoxLayout()
        self._search = QLineEdit()
        self._search.setPlaceholderText("Buscar producto...")
        self._search.setMaximumWidth(300)
        self._search.textChanged.connect(self._filter_table)
        top.addWidget(self._search)

        top.addStretch()

        btn_refresh = QPushButton("Actualizar")
        btn_refresh.clicked.connect(self._load_products)
        top.addWidget(btn_refresh)

        self._btn_delete = QPushButton("Eliminar Seleccionado")
        self._btn_delete.setObjectName("danger")
        self._btn_delete.clicked.connect(self._delete_selected)
        top.addWidget(self._btn_delete)

        layout.addLayout(top)

        # Products table
        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(
            ["ID", "Nombre", "Imagenes", "Creado", ""]
        )
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)

        layout.addWidget(self._table, 1)

        # Info panel
        self._info = QLabel("Selecciona un producto para ver detalles.")
        self._info.setStyleSheet("color: #a6adc8;")
        layout.addWidget(self._info)

    def on_activated(self) -> None:
        self._load_products()

    def _load_products(self) -> None:
        with self._db as session:
            products = get_all_products(session)
            self._table.setRowCount(len(products))
            for row, p in enumerate(products):
                self._table.setItem(row, 0, QTableWidgetItem(str(p.id)))
                self._table.setItem(row, 1, QTableWidgetItem(p.name))
                self._table.setItem(
                    row, 2, QTableWidgetItem(str(p.image_count))
                )
                created = p.created_at.strftime("%Y-%m-%d %H:%M") if p.created_at else "-"
                self._table.setItem(row, 3, QTableWidgetItem(created))

                btn = QPushButton("Editar")
                btn.setProperty("product_id", p.id)
                btn.clicked.connect(self._edit_product)
                self._table.setCellWidget(row, 4, btn)

    @Slot(str)
    def _filter_table(self, text: str) -> None:
        text = text.lower()
        for row in range(self._table.rowCount()):
            name_item = self._table.item(row, 1)
            match = text in name_item.text().lower() if name_item else False
            self._table.setRowHidden(row, not match and bool(text))

    @Slot()
    def _delete_selected(self) -> None:
        row = self._table.currentRow()
        if row < 0:
            return
        item = self._table.item(row, 0)
        if item is None:
            return
        product_id = int(item.text())
        name_item = self._table.item(row, 1)
        name = name_item.text() if name_item else ""

        reply = QMessageBox.question(
            self,
            "Confirmar eliminacion",
            f"Eliminar producto '{name}' y todas sus imagenes?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Delete files
        self._storage.delete_product_files(product_id)

        # Delete from DB
        with self._db as session:
            delete_product(session, product_id)

        self._load_products()

    @Slot()
    def _edit_product(self) -> None:
        btn = self.sender()
        if btn is None:
            return
        product_id = btn.property("product_id")
        # Simple inline rename via dialog
        from PySide6.QtWidgets import QInputDialog

        new_name, ok = QInputDialog.getText(
            self, "Editar Producto", "Nuevo nombre:"
        )
        if ok and new_name.strip():
            with self._db as session:
                update_product(session, product_id, name=new_name.strip())
            self._load_products()
