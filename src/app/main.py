"""Application entry point."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from src.app.main_window import MainWindow
from src.app.styles import MAIN_STYLE


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Detector de Inventario")
    app.setStyleSheet(MAIN_STYLE)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
