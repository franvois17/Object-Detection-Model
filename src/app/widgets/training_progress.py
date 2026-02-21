"""Training progress display widget."""

from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)


class TrainingProgress(QWidget):
    """Widget showing training progress with metrics."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self._status_label = QLabel("Listo para entrenar")
        self._status_label.setObjectName("section")
        layout.addWidget(self._status_label)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFixedHeight(12)
        layout.addWidget(self._progress)

        metrics_layout = QHBoxLayout()

        self._epoch_label = QLabel("Epoca: -")
        metrics_layout.addWidget(self._epoch_label)

        self._loss_label = QLabel("Loss: -")
        metrics_layout.addWidget(self._loss_label)

        self._acc_label = QLabel("Accuracy: -")
        metrics_layout.addWidget(self._acc_label)

        self._best_label = QLabel("Mejor: -")
        metrics_layout.addWidget(self._best_label)

        layout.addLayout(metrics_layout)

        # Epoch history
        self._history_label = QLabel("")
        self._history_label.setWordWrap(True)
        self._history_label.setStyleSheet("color: #a6adc8; font-size: 12px;")
        layout.addWidget(self._history_label)

        self._history: list[str] = []
        self._best_acc = 0.0

    def reset(self) -> None:
        self._progress.setValue(0)
        self._status_label.setText("Listo para entrenar")
        self._epoch_label.setText("Epoca: -")
        self._loss_label.setText("Loss: -")
        self._acc_label.setText("Accuracy: -")
        self._best_label.setText("Mejor: -")
        self._history.clear()
        self._history_label.setText("")
        self._best_acc = 0.0

    @Slot(str)
    def set_status(self, text: str) -> None:
        self._status_label.setText(text)

    @Slot(int)
    def set_progress(self, percent: int) -> None:
        self._progress.setValue(percent)

    @Slot(int, float, float)
    def update_epoch(self, epoch: int, train_loss: float, val_accuracy: float) -> None:
        self._epoch_label.setText(f"Epoca: {epoch}")
        self._loss_label.setText(f"Loss: {train_loss:.4f}")
        self._acc_label.setText(f"Accuracy: {val_accuracy:.1%}")

        if val_accuracy > self._best_acc:
            self._best_acc = val_accuracy
        self._best_label.setText(f"Mejor: {self._best_acc:.1%}")

        entry = f"E{epoch}: loss={train_loss:.4f} acc={val_accuracy:.1%}"
        self._history.append(entry)
        # Show last 5 entries
        self._history_label.setText("  |  ".join(self._history[-5:]))

    def set_complete(self, accuracy: float) -> None:
        self._progress.setValue(100)
        self._status_label.setText(
            f"Entrenamiento completo - Accuracy: {accuracy:.1%}"
        )
