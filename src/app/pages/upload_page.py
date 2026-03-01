"""Upload video page - select video for new or existing product."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from src.app.widgets.video_player import VideoPlayer
from src.data.database import DatabaseManager, create_product, get_all_products
from src.workers.video_worker import VideoProcessingWorker
from src.workers.training_worker import TrainingWorker


class UploadPage(QWidget):
    processing_complete = Signal()
    auto_training_complete = Signal()

    def __init__(self, db: DatabaseManager, parent=None) -> None:
        super().__init__(parent)
        self._db = db
        self._video_path: str | None = None
        self._worker: VideoProcessingWorker | None = None
        self._training_worker: TrainingWorker | None = None
        self._current_product_name: str = ""
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
            "Puedes subir multiples videos del mismo producto para mejorar el reconocimiento."
        )
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        # Product selection mode
        mode_layout = QHBoxLayout()
        self._radio_new = QRadioButton("Nuevo producto")
        self._radio_new.setChecked(True)
        self._radio_new.toggled.connect(self._on_mode_changed)
        mode_layout.addWidget(self._radio_new)

        self._radio_existing = QRadioButton("Agregar a producto existente")
        self._radio_existing.toggled.connect(self._on_mode_changed)
        mode_layout.addWidget(self._radio_existing)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)

        # New product name input
        self._new_layout = QHBoxLayout()
        self._new_layout.addWidget(QLabel("Nombre:"))
        self._name_input = QLineEdit()
        self._name_input.setPlaceholderText("Ej: Coca-Cola 600ml")
        self._name_input.setMaximumWidth(400)
        self._name_input.textChanged.connect(self._update_process_button)
        self._new_layout.addWidget(self._name_input)
        self._new_layout.addStretch()
        layout.addLayout(self._new_layout)

        # Existing product selector
        self._existing_layout = QHBoxLayout()
        self._existing_layout.addWidget(QLabel("Producto:"))
        self._product_combo = QComboBox()
        self._product_combo.setMinimumWidth(300)
        self._product_combo.currentIndexChanged.connect(self._update_process_button)
        self._existing_layout.addWidget(self._product_combo)

        self._btn_refresh = QPushButton("Actualizar")
        self._btn_refresh.clicked.connect(self._load_products)
        self._existing_layout.addWidget(self._btn_refresh)
        self._existing_layout.addStretch()
        layout.addLayout(self._existing_layout)

        # Initially hide existing product selector
        self._set_existing_visible(False)

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

    def _set_existing_visible(self, visible: bool) -> None:
        for i in range(self._existing_layout.count()):
            w = self._existing_layout.itemAt(i).widget()
            if w:
                w.setVisible(visible)

    def _set_new_visible(self, visible: bool) -> None:
        for i in range(self._new_layout.count()):
            w = self._new_layout.itemAt(i).widget()
            if w:
                w.setVisible(visible)

    @Slot(bool)
    def _on_mode_changed(self, _checked: bool) -> None:
        is_new = self._radio_new.isChecked()
        self._set_new_visible(is_new)
        self._set_existing_visible(not is_new)
        if not is_new:
            self._load_products()
        self._update_process_button()

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
        self._update_process_button()

    def _update_process_button(self) -> None:
        has_video = self._video_path is not None
        if self._radio_new.isChecked():
            has_product = bool(self._name_input.text().strip())
        else:
            has_product = self._product_combo.count() > 0
        self._btn_process.setEnabled(has_video and has_product)

    def on_activated(self) -> None:
        if self._radio_existing.isChecked():
            self._load_products()

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
        self._update_process_button()

    @Slot()
    def _start_processing(self) -> None:
        if not self._video_path:
            return

        if self._radio_new.isChecked():
            name = self._name_input.text().strip()
            if not name:
                return
            # Check for duplicate name
            with self._db as session:
                existing = get_all_products(session)
                if any(p.name == name for p in existing):
                    QMessageBox.warning(
                        self, "Duplicado",
                        f"Ya existe un producto llamado '{name}'. "
                        "Usa 'Agregar a producto existente' para anadir mas videos."
                    )
                    return
            # Create product in DB
            with self._db as session:
                product = create_product(session, name)
                product_id = product.id
        else:
            idx = self._product_combo.currentIndex()
            if idx < 0:
                return
            product_id = self._product_combo.itemData(idx)
            name = self._product_combo.currentText().split(" (")[0]

        self._current_product_name = name
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
        n = result.get("num_crops", 0)
        # Schedule the worker for deletion only after its thread has fully
        # exited. Using deleteLater() is the safe Qt pattern — it lets Qt
        # clean up the object once control returns to the event loop.
        if self._worker is not None:
            self._worker.finished.connect(self._worker.deleteLater)
        self._worker = None
        self._progress_label.setText(
            f"{n} recortes extraidos. Entrenando modelo..."
        )
        self._progress.setValue(0)
        self.processing_complete.emit()
        self._start_auto_training()

    def _start_auto_training(self) -> None:
        self._training_worker = TrainingWorker(mode="incremental")
        self._training_worker.progress.connect(self._on_training_progress)
        self._training_worker.error.connect(self._on_training_error)
        self._training_worker.finished_ok.connect(self._on_training_finished)
        self._training_worker.start()

    @Slot(int, str)
    def _on_training_progress(self, percent: int, message: str) -> None:
        self._progress.setValue(percent)
        self._progress_label.setText(f"Entrenando: {message}")

    @Slot(str)
    def _on_training_error(self, message: str) -> None:
        self._reset_ui()
        self._progress_label.setText(
            f"Video procesado, pero error en entrenamiento: {message}"
        )

    @Slot(object)
    def _on_training_finished(self, result: dict) -> None:
        self._reset_ui()
        name = self._current_product_name
        self._progress_label.setText(
            f"Producto '{name}' listo para reconocimiento"
        )
        self._progress_label.setStyleSheet("color: #a6e3a1; font-size: 12px;")
        self.auto_training_complete.emit()

    def _reset_ui(self) -> None:
        self._btn_process.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._btn_select.setEnabled(True)
        self._progress.setVisible(False)
        self._worker = None
        self._training_worker = None
