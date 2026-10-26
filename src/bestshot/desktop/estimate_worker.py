"""Inspection des durées en arrière-plan pour garder la fenêtre disponible."""

from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from bestshot.infrastructure.ffprobe import SubprocessFFprobeRunner
from bestshot.services.extraction_estimate import inspect_video_durations
from bestshot.video.probe import VideoProbe


class EstimateSignals(QObject):
    completed = Signal(int, object)
    failed = Signal(int, str)


class EstimateWorker(QRunnable):
    """Lit uniquement les métadonnées locales, sans charger CLIP."""

    def __init__(self, directory: Path, request_id: int) -> None:
        super().__init__()
        self.signals = EstimateSignals()
        self._directory = directory
        self._request_id = request_id

    @Slot()
    def run(self) -> None:
        try:
            summary = inspect_video_durations(
                self._directory, VideoProbe(SubprocessFFprobeRunner())
            )
        except (OSError, RuntimeError, ValueError) as error:
            self.signals.failed.emit(self._request_id, str(error))
        else:
            self.signals.completed.emit(self._request_id, summary)
