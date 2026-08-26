"""Worker Qt isolant le traitement vidéo de l'interface graphique."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Event

from PySide6.QtCore import QObject, Signal, Slot

from bestshot.services.batch import BatchExportRunner, BatchProgress

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DesktopProcessingJob:
    """Paramètres saisis dans l'interface pour un traitement par lot."""

    source_directory: Path
    output_directory: Path
    image_format: str


class ProcessingWorker(QObject):
    """Exécute un lot dans un QThread sans manipuler directement les widgets."""

    progress = Signal(object)
    completed = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, runner: BatchExportRunner, job: DesktopProcessingJob) -> None:
        super().__init__()
        self._runner = runner
        self._job = job
        self._stop_requested = Event()

    def request_stop(self) -> None:
        """Demande un arrêt coopératif sans toucher aux widgets Qt."""
        self._stop_requested.set()

    @Slot()
    def run(self) -> None:
        """Lance le cas d'usage puis transmet son résultat au thread de l'interface."""
        try:
            result = self._runner.run(
                self._job.source_directory,
                None,
                self._job.output_directory,
                self._job.image_format,
                on_progress=self._emit_progress,
                should_stop=self._stop_requested.is_set,
            )
        except Exception as error:
            logger.exception("Le traitement de lot a échoué avant son terme")
            self.failed.emit(str(error))
        else:
            self.completed.emit(result)
        finally:
            self.finished.emit()

    def _emit_progress(self, progress: BatchProgress) -> None:
        self.progress.emit(progress)
