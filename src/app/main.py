"""Application entry point."""

from __future__ import annotations

import logging
import os
import sys
import traceback
from pathlib import Path

# ── Fix sys.stdout/stderr being None in PyInstaller windowed mode ─────────────
# When console=False, PyInstaller sets stdout/stderr to None. PyTorch's
# model-download progress bar calls sys.stdout.write(), which crashes with
# AttributeError: 'NoneType' object has no attribute 'write'.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

# ── Environment tweaks (must happen before any torch/cv2 imports) ─────────────
# Prevent OpenMP duplicate library crash (PyTorch + OpenCV/FAISS conflict)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Limit OpenMP threads to avoid crash on some CPUs when running in a QThread
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")


def _setup_logging() -> None:
    """Write all logs (including uncaught exceptions) to a file in APPDATA."""
    if getattr(sys, "frozen", False):
        log_dir = Path(os.environ.get("APPDATA", Path.home())) / "DetectorInventario"
    else:
        log_dir = Path(__file__).resolve().parents[2] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "app.log"

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    def _handle_exception(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logging.critical(
            "Uncaught exception:\n%s",
            "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
        )

    sys.excepthook = _handle_exception


_setup_logging()

from PySide6.QtWidgets import QApplication

from src.app.main_window import MainWindow
from src.app.styles import MAIN_STYLE

logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Starting Detector de Inventario")
    app = QApplication(sys.argv)
    app.setApplicationName("Detector de Inventario")
    app.setStyleSheet(MAIN_STYLE)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
