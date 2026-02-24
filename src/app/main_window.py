"""Main application window with sidebar navigation."""

from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.app.pages.upload_page import UploadPage
from src.app.pages.training_page import TrainingPage
from src.app.pages.recognition_page import RecognitionPage
from src.core.config import CONFIG
from src.core.device import DEVICE_LABEL
from src.data.database import DatabaseManager


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Detector de Inventario")
        self.setMinimumSize(1100, 700)
        self.resize(1280, 800)

        CONFIG.ensure_dirs()
        self.db = DatabaseManager()
        self.db.create_tables()

        self._build_ui()
        self._update_status()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 16, 0, 16)
        sidebar_layout.setSpacing(0)

        app_label = QLabel("  Inventario")
        app_label.setObjectName("title")
        app_label.setFixedHeight(48)
        sidebar_layout.addWidget(app_label)
        sidebar_layout.addSpacing(16)

        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)
        nav_items = [
            ("Subir Video", 0),
            ("Entrenar", 1),
            ("Reconocer", 2),
        ]
        for label, idx in nav_items:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self._nav_group.addButton(btn, idx)
            sidebar_layout.addWidget(btn)

        sidebar_layout.addStretch()
        main_layout.addWidget(sidebar)

        # Right side: content + status
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # Stacked pages
        self._stack = QStackedWidget()
        self._stack.setObjectName("content_area")

        self._upload_page = UploadPage(self.db)
        self._training_page = TrainingPage(self.db)
        self._recognition_page = RecognitionPage(self.db)

        self._stack.addWidget(self._upload_page)
        self._stack.addWidget(self._training_page)
        self._stack.addWidget(self._recognition_page)

        right_layout.addWidget(self._stack, 1)

        # Status bar
        self._status = QLabel()
        self._status.setObjectName("status_bar")
        self._status.setFixedHeight(32)
        right_layout.addWidget(self._status)

        main_layout.addWidget(right, 1)

        # Connect navigation
        self._nav_group.idClicked.connect(self._on_nav)
        self._nav_group.button(0).setChecked(True)

        # Connect cross-page signals
        self._upload_page.processing_complete.connect(self._update_status)
        self._upload_page.auto_training_complete.connect(self._update_status)
        self._training_page.training_complete.connect(self._update_status)

    @Slot(int)
    def _on_nav(self, idx: int) -> None:
        self._stack.setCurrentIndex(idx)
        page = self._stack.currentWidget()
        if hasattr(page, "on_activated"):
            page.on_activated()

    def _update_status(self) -> None:
        with self.db as session:
            from src.data.database import Product
            count = session.query(Product).count()
        self._status.setText(
            f"  Dispositivo: {DEVICE_LABEL}  |  Productos: {count}"
        )

    def closeEvent(self, event) -> None:
        # Stop camera if running
        if hasattr(self._recognition_page, "stop_camera"):
            self._recognition_page.stop_camera()
        event.accept()
