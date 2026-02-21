"""Upload video page - select video, name product, process frames."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.app.widgets.video_player import VideoPlayer
from src.data.database import DatabaseManager, create_product, get_all_products
from src.workers.video_worker import VideoProcessingWorker


class UploadPage(QWidget):
    processing_complete = Signal()

    def __init__(self, db: DatabaseManager, parent=None) -> None:
        super().__init__(parent)
        self._db = db
        self._video_path: str | None = None
        self._worker: VideoProcessingWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        title = QLabel("Subir Video de Producto")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel(
            "Graba un video de 15-60 segundos del producto desde varios angulos. "
            "El sistema extraera frames automaticamente y detectara el objeto."
        )
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        # Product name input
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Nombre del producto:"))
        self._name_input = QLineEdit()
        self._name_input.setPlaceholderText("Ej: Coca-Cola 600ml")
        self._name_input.setMaximumWidth(400)
        name_layout.addWidget(self._name_input)
        name_layout.addStretch()
        layout.addLayout(name_layout)

        # Video selection
        file_layout = QHBoxLayout()
        self._btn_select = QPushButton("Seleccionar Video")
        self._btn_select.setObjectName("secondary")
        self._btn_select.clicked.connect(self._select_video)
        file_layout.addWidget(self._btn_select)

        self._file_label = QLabel("Ningun video seleccionado")
        self._file_label.setStyleSheet("color: #a6adc8;")
        file_layout.addWidget(self._file_label)
        file_layout.addStretch()
        layout.addLayout(file_layout)

        # Video preview
        self._player = VideoPlayer()
        layout.addWidget(self._player, 1)

        # Process button + progress
        bottom = QHBoxLayout()
        self._btn_process = QPushButton("Procesar Video")
        self._btn_process.setObjectName("primary")
        self._btn_process.setEnabled(False)
        self._btn_process.clicked.connect(self._start_processing)
        bottom.addWidget(self._btn_process)

        self._btn_cancel = QPushButton("Cancelar")
        self._btn_cancel.setObjectName("danger")
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel_processing)
        bottom.addWidget(self._btn_cancel)

        bottom.addStretch()
        layout.addLayout(bottom)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setFixedHeight(10)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet("color: #a6adc8; font-size: 12px;")
        layout.addWidget(self._progress_label)

    @Slot()
    def _select_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar Video",
            "",
            "Videos (*.mp4 *.avi *.mov *.mkv *.webm);;Todos (*)",
        )
        if not path:
            return
        self._video_path = path
        self._file_label.setText(Path(path).name)
        ok = self._player.load_video(path)
        if not ok:
            QMessageBox.warning(self, "Error", "No se pudo abrir el video.")
            return
        self._btn_process.setEnabled(bool(self._name_input.text().strip()))
        self._name_input.textChanged.connect(
            lambda t: self._btn_process.setEnabled(bool(t.strip()) and self._video_path is not None)
        )

    @Slot()
    def _start_processing(self) -> None:
        name = self._name_input.text().strip()
        if not name or not self._video_path:
            return

        # Check for duplicate name
        with self._db as session:
            existing = get_all_products(session)
            if any(p.name == name for p in existing):
                QMessageBox.warning(
                    self, "Duplicado",
                    f"Ya existe un producto llamado '{name}'."
                )
                return

        # Create product in DB
        with self._db as session:
            product = create_product(session, name)
            product_id = product.id

        self._player.stop()
        self._btn_process.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._btn_select.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)

        self._worker = VideoProcessingWorker(
            self._video_path, product_id, name
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.error.connect(self._on_error)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.start()

    @Slot()
    def _cancel_processing(self) -> None:
        if self._worker is not None:
            self._worker.cancel()

    @Slot(int, str)
    def _on_progress(self, percent: int, message: str) -> None:
        self._progress.setValue(percent)
        self._progress_label.setText(message)

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self._reset_ui()
        QMessageBox.critical(self, "Error", message)

    @Slot(object)
    def _on_finished(self, result: dict) -> None:
        self._reset_ui()
        n = result.get("num_crops", 0)
        self._progress_label.setText(
            f"Procesamiento completo: {n} recortes extraidos."
        )
        self.processing_complete.emit()

    def _reset_ui(self) -> None:
        self._btn_process.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._btn_select.setEnabled(True)
        self._progress.setVisible(False)
        self._worker = None
