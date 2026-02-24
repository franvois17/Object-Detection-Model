"""Training page - simplified with one main button and optional advanced mode."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.app.widgets.training_progress import TrainingProgress
from src.core.config import CONFIG
from src.data.database import DatabaseManager, get_all_products, get_product_images
from src.workers.training_worker import TrainingWorker


class TrainingPage(QWidget):
    training_complete = Signal()

    def __init__(self, db: DatabaseManager, parent=None) -> None:
        super().__init__(parent)
        self._db = db
        self._worker: TrainingWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        title = QLabel("Entrenar Modelo")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel(
            "Entrena el modelo con todos los productos registrados. "
            "El entrenamiento se ejecuta automaticamente al subir un video, "
            "pero puedes re-entrenar manualmente aqui."
        )
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        # Product stats
        self._stats_label = QLabel()
        self._stats_label.setStyleSheet("color: #a6adc8;")
        layout.addWidget(self._stats_label)

        # Product list
        self._product_list_label = QLabel()
        self._product_list_label.setWordWrap(True)
        self._product_list_label.setStyleSheet(
            "background-color: #313244; border-radius: 8px; "
            "padding: 12px; color: #cdd6f4;"
        )
        layout.addWidget(self._product_list_label)

        # Main train button
        self._btn_train = QPushButton("Entrenar Modelo")
        self._btn_train.setObjectName("primary")
        self._btn_train.setMinimumHeight(48)
        self._btn_train.setStyleSheet(
            "QPushButton { font-size: 16px; font-weight: bold; }"
        )
        self._btn_train.clicked.connect(
            lambda: self._start_training("incremental")
        )
        layout.addWidget(self._btn_train)

        # Advanced section (collapsed by default)
        self._btn_advanced_toggle = QPushButton("Entrenamiento Avanzado")
        self._btn_advanced_toggle.setObjectName("secondary")
        self._btn_advanced_toggle.setCheckable(True)
        self._btn_advanced_toggle.clicked.connect(self._toggle_advanced)
        layout.addWidget(
            self._btn_advanced_toggle, alignment=Qt.AlignmentFlag.AlignLeft
        )

        self._advanced_group = QGroupBox("Entrenamiento Completo (Fine-tune)")
        adv_layout = QVBoxLayout(self._advanced_group)
        adv_desc = QLabel(
            "Entrena un clasificador con fine-tuning de EfficientNet. "
            "Mayor precision pero tarda varios minutos."
        )
        adv_desc.setWordWrap(True)
        adv_desc.setStyleSheet("color: #a6adc8;")
        adv_layout.addWidget(adv_desc)

        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Epocas:"))
        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(5, 100)
        self._epochs_spin.setValue(CONFIG.train_epochs)
        epochs_layout.addWidget(self._epochs_spin)
        epochs_layout.addStretch()
        adv_layout.addLayout(epochs_layout)

        btn_row = QHBoxLayout()
        self._btn_full = QPushButton("Iniciar Entrenamiento Completo")
        self._btn_full.setObjectName("primary")
        self._btn_full.clicked.connect(
            lambda: self._start_training("full")
        )
        btn_row.addWidget(self._btn_full)

        self._btn_cancel = QPushButton("Cancelar")
        self._btn_cancel.setObjectName("danger")
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel_training)
        btn_row.addWidget(self._btn_cancel)
        btn_row.addStretch()
        adv_layout.addLayout(btn_row)

        self._advanced_group.setVisible(False)
        layout.addWidget(self._advanced_group)

        # Progress widget
        self._progress = TrainingProgress()
        layout.addWidget(self._progress)

        layout.addStretch()

    @Slot(bool)
    def _toggle_advanced(self, checked: bool) -> None:
        self._advanced_group.setVisible(checked)

    def on_activated(self) -> None:
        self._update_stats()

    def _update_stats(self) -> None:
        with self._db as session:
            products = get_all_products(session)
            total_imgs = sum(p.image_count for p in products)

        self._stats_label.setText(
            f"Productos registrados: {len(products)}  |  "
            f"Imagenes totales: {total_imgs}"
        )

        if products:
            lines = []
            for p in products:
                lines.append(f"  {p.name}  --  {p.image_count} imagenes")
            self._product_list_label.setText("\n".join(lines))
        else:
            self._product_list_label.setText(
                "No hay productos registrados. Sube un video primero."
            )

    def _start_training(self, mode: str) -> None:
        with self._db as session:
            products = get_all_products(session)
            products_with_images = [
                p for p in products
                if len(get_product_images(session, p.id)) > 0
            ]

        if len(products_with_images) < 2:
            QMessageBox.warning(
                self,
                "Productos insuficientes",
                "Se necesitan al menos 2 productos con imagenes para entrenar.",
            )
            return

        self._btn_train.setEnabled(False)
        self._btn_full.setEnabled(False)
        self._btn_cancel.setEnabled(mode == "full")
        self._progress.reset()
        self._progress.set_status(
            "Construyendo indice KNN..."
            if mode == "incremental"
            else "Entrenando clasificador..."
        )

        self._worker = TrainingWorker(mode=mode)
        self._worker.progress.connect(self._progress.set_progress)
        self._worker.progress.connect(
            lambda p, msg: self._progress.set_status(msg)
        )
        self._worker.error.connect(self._on_error)
        self._worker.finished_ok.connect(self._on_finished)
        if mode == "full":
            self._worker.epoch_done.connect(self._progress.update_epoch)
        self._worker.start()

    @Slot()
    def _cancel_training(self) -> None:
        if self._worker is not None:
            self._worker.cancel()

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self._reset_buttons()
        self._progress.set_status(f"Error: {message}")
        QMessageBox.critical(self, "Error de Entrenamiento", message)

    @Slot(object)
    def _on_finished(self, result: dict) -> None:
        self._reset_buttons()
        mode = result.get("mode", "")
        if mode == "incremental":
            n = result.get("num_embeddings", 0)
            self._progress.set_status(
                f"Indice KNN construido con {n} embeddings."
            )
        else:
            acc = result.get("best_accuracy", 0)
            self._progress.set_complete(acc)
        self._update_stats()
        self.training_complete.emit()

    def _reset_buttons(self) -> None:
        self._btn_train.setEnabled(True)
        self._btn_full.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._worker = None
