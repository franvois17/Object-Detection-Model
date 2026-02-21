"""Base QThread worker with cancellation, progress and error handling."""

from __future__ import annotations

import logging
from abc import abstractmethod

from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)


class BaseWorker(QThread):
    """Abstract base worker that wraps ``do_work()`` in a QThread.

    Signals
    -------
    progress : Signal(int, str)
        Emitted to report progress.  The first argument is a percentage
        (0-100) and the second is a human-readable status message.
    error : Signal(str)
        Emitted when ``do_work()`` raises an unhandled exception.  The
        payload is the string representation of the exception.
    finished_ok : Signal(object)
        Emitted when ``do_work()`` returns successfully.  The payload
        is whatever object ``do_work()`` returned.
    """

    progress = Signal(int, str)
    error = Signal(str)
    finished_ok = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._cancelled: bool = False

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def cancel(self) -> None:
        """Request cancellation of the running work."""
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        """Return ``True`` if cancellation has been requested."""
        return self._cancelled

    # ------------------------------------------------------------------
    # QThread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute ``do_work()`` and emit the appropriate signal."""
        self._cancelled = False
        try:
            result = self.do_work()
            if not self.is_cancelled:
                self.finished_ok.emit(result)
        except Exception as exc:
            logger.exception("Worker %s failed", type(self).__name__)
            self.error.emit(str(exc))

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def do_work(self) -> object:
        """Perform the actual work.  Subclasses **must** override this.

        Returns
        -------
        object
            An arbitrary result payload that will be emitted via
            ``finished_ok``.
        """
        ...
