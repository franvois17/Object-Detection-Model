"""Recognition page - live camera feed with real-time object recognition."""

from __future__ import annotations

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

from src.app.widgets.camera_view import CameraView
from src.data.database import DatabaseManager
from src.workers.camera_worker import CameraWorker


class RecognitionPage(QWidget):
    def __init__(self, db: DatabaseManager, parent=None) -> None:
        super().__init__(parent)
        self._db = db
        self._worker: CameraWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        title = QLabel("Reconocimiento en Vivo")
        title.setObjectName("title")
        layout.addWidget(title)

        # Controls
        controls = QHBoxLayout()

        controls.addWidget(QLabel("Camara:"))
        self._camera_combo = QComboBox()
        self._camera_combo.addItem("Camara 0", 0)
        self._camera_combo.addItem("Camara 1", 1)
        self._camera_combo.setMaximumWidth(150)
        controls.addWidget(self._camera_combo)

        controls.addWidget(QLabel("Modo:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("KNN (rapido)", "knn")
        self._mode_combo.addItem("Clasificador", "classifier")
        self._mode_combo.setMaximumWidth(180)
        controls.addWidget(self._mode_combo)

        controls.addStretch()

        self._btn_start = QPushButton("Iniciar Camara")
        self._btn_start.setObjectName("primary")
        self._btn_start.clicked.connect(self._toggle_camera)
        controls.addWidget(self._btn_start)

        layout.addLayout(controls)

        # Camera view
        self._camera_view = CameraView()
        layout.addWidget(self._camera_view, 1)

        # Detection info
        info_layout = QHBoxLayout()
        self._fps_label = QLabel("FPS: -")
        self._fps_label.setStyleSheet("color: #a6adc8;")
        info_layout.addWidget(self._fps_label)

        self._detection_label = QLabel("Detecciones: -")
        self._detection_label.setStyleSheet("color: #a6adc8;")
        info_layout.addWidget(self._detection_label)

        info_layout.addStretch()
        layout.addLayout(info_layout)

        # Detection results list
        self._results_label = QLabel("")
        self._results_label.setWordWrap(True)
        self._results_label.setStyleSheet(
            "background-color: #313244; border-radius: 8px; "
            "padding: 12px; min-height: 60px;"
        )
        layout.addWidget(self._results_label)

    @Slot()
    def _toggle_camera(self) -> None:
        if self._worker is not None:
            self.stop_camera()
        else:
            self._start_camera()

    def _start_camera(self) -> None:
        camera_idx = self._camera_combo.currentData()
        mode = self._mode_combo.currentData()

        self._worker = CameraWorker(camera_index=camera_idx)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.detection_ready.connect(self._on_detections)
        self._worker.error.connect(self._on_error)
        self._worker.finished_ok.connect(self._on_stopped)
        self._worker.start()

        self._btn_start.setText("Detener Camara")
        self._btn_start.setObjectName("danger")
        self._btn_start.style().unpolish(self._btn_start)
        self._btn_start.style().polish(self._btn_start)
        self._camera_combo.setEnabled(False)

    def stop_camera(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._worker.wait(3000)
            self._worker = None
        self._camera_view.clear()
        self._btn_start.setText("Iniciar Camara")
        self._btn_start.setObjectName("primary")
        self._btn_start.style().unpolish(self._btn_start)
        self._btn_start.style().polish(self._btn_start)
        self._camera_combo.setEnabled(True)
        self._fps_label.setText("FPS: -")
        self._detection_label.setText("Detecciones: -")
        self._results_label.setText("")

    @Slot(object)
    def _on_frame(self, frame) -> None:
        self._camera_view.update_frame(frame)

    @Slot(list)
    def _on_detections(self, detections: list) -> None:
        self._detection_label.setText(f"Detecciones: {len(detections)}")
        if not detections:
            self._results_label.setText("No se detectaron objetos")
            return

        lines = []
        for d in detections:
            name = d.get("product_name", "?")
            conf = d.get("confidence", 0)
            lines.append(f"  {name}  ({conf:.0%})")
        self._results_label.setText("\n".join(lines))

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self.stop_camera()
        QMessageBox.critical(self, "Error de Camara", message)

    @Slot(object)
    def _on_stopped(self, _result) -> None:
        self.stop_camera()

    def on_activated(self) -> None:
        pass
