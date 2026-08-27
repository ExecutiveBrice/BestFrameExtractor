"""Interface locale réduite à l'analyse et à l'export des candidates V2."""
# mypy: ignore-errors

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bestshot.dataset.labels import FrameLabel
from bestshot.dataset.preview_cache import PreviewCache, PreviewCacheError
from bestshot.dataset.repository import DatasetSettings
from bestshot.dataset.sqlite_repository import DatasetRepositoryError, SQLiteDatasetRepository
from bestshot.desktop.video_library import VideoLibraryError, discover_videos
from bestshot.embedding.cache import EmbeddingCache
from bestshot.embedding.dinov2 import DINOv2EmbeddingProvider, DINOv2ModelError
from bestshot.embedding.provider import EmbeddingError
from bestshot.infrastructure.config import (
    ConfigurationError,
    load_dataset_settings,
    load_embedding_settings,
    load_presampling_settings,
)
from bestshot.infrastructure.embedding_frames import (
    EmbeddingFrameReadError,
    PyAVCandidatePreviewReader,
)
from bestshot.infrastructure.selection_export import (
    PyAVSelectedFrameExporter,
    SelectedFrameExportError,
)
from bestshot.infrastructure.temporal_sampling import PyAVTemporalSamplingBackend
from bestshot.sampling.candidate_generator import CandidateGenerationError, CandidateGenerator
from bestshot.sampling.sharpness_ranker import SharpnessRanker
from bestshot.sampling.temporal_sampler import TemporalSampler, TemporalSamplingError
from bestshot.services.candidate_labeling import (
    CandidateLabelingError,
    CandidateLabelingItem,
    CandidateLabelingService,
)
from bestshot.services.embeddings import CANDIDATE_EXPORT_DIRECTORY_NAME, VideoEmbeddingRunner
from bestshot.services.label_selection import (
    LabelDrivenSelectionService,
    LabelSelectionError,
    LabelSelectionResult,
)
from bestshot.services.personal_label_model import PersonalLabelModelError


class DesktopApplicationError(RuntimeError):
    """La fenêtre principale ne peut pas être démarrée dans cet environnement."""


_ANALYSIS_ERRORS = (
    CandidateGenerationError,
    ConfigurationError,
    DatasetRepositoryError,
    DINOv2ModelError,
    EmbeddingError,
    EmbeddingFrameReadError,
    OSError,
    PreviewCacheError,
    SelectedFrameExportError,
    TemporalSamplingError,
    ValueError,
)


@dataclass(frozen=True, slots=True)
class AnalysisSummary:
    """Résultat d'une analyse de lot, sans image en mémoire dans l'interface."""

    analyzed_paths: tuple[Path, ...]
    failed_paths: tuple[Path, ...]
    failure_messages: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CandidateLoadSummary:
    """Résultat de la lecture SQLite demandée par l'onglet d'apprentissage."""

    items: tuple[CandidateLabelingItem, ...]
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateLabelSummary:
    """Résultat de l'écriture d'un label personnel hors du thread UI."""

    label: FrameLabel
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class AISelectionSummary:
    """Résultat de l'entraînement local puis de l'export IA."""

    results: tuple[LabelSelectionResult, ...]
    error_message: str | None = None


def run_desktop_application() -> int:
    """Démarre la seule fenêtre d'analyse locale."""
    qt = _qt()
    try:
        dataset_settings = load_dataset_settings()
    except ConfigurationError as error:
        raise DesktopApplicationError(str(error)) from error

    application = qt.QApplication.instance() or qt.QApplication([])
    window = VideoLibraryWindow(dataset_settings, _create_embedding_runner)
    window.show()
    return application.exec()


def main() -> int:
    """Point d'entrée ``bestshot-gui``."""
    try:
        return run_desktop_application()
    except DesktopApplicationError as error:
        print(error, file=sys.stderr)
        return 1


class VideoLibraryWindow:
    """Analyse les vidéos puis recueille les labels personnels des candidates."""

    def __init__(
        self,
        dataset_settings: DatasetSettings,
        runner_factory: Callable[[], VideoEmbeddingRunner],
        labeling_service_factory: Callable[[], CandidateLabelingService] | None = None,
        ai_selection_service_factory: Callable[[], LabelDrivenSelectionService] | None = None,
    ) -> None:
        qt = _qt()
        self._qt = qt
        self._dataset_settings = dataset_settings
        self._runner_factory = runner_factory
        self._labeling_service_factory = labeling_service_factory or _create_candidate_labeling_service
        self._ai_selection_service_factory = ai_selection_service_factory or _create_ai_selection_service
        self._videos: tuple[Path, ...] = ()
        self._analysis_thread: Any | None = None
        self._analysis_worker: Any | None = None
        self._analysis_events = _AnalysisEventBridge(self)
        self._learning_items: tuple[CandidateLabelingItem, ...] = ()
        self._learning_index = 0
        self._candidate_load_thread: Any | None = None
        self._candidate_load_worker: Any | None = None
        self._candidate_label_thread: Any | None = None
        self._candidate_label_worker: Any | None = None
        self._learning_events = _LearningEventBridge(self)
        self._ai_selection_videos: tuple[Path, ...] = ()
        self._ai_selection_thread: Any | None = None
        self._ai_selection_worker: Any | None = None
        self._ai_selection_events = _AISelectionEventBridge(self)

        self._window = _window_with_close_handler(self._request_close)
        self._window.setWindowTitle("BestShotAI — sélection de candidates")
        root = qt.QWidget()
        root_layout = qt.QVBoxLayout(root)
        self._tabs = qt.QTabWidget()
        root_layout.addWidget(self._tabs)
        analysis_tab = qt.QWidget()
        layout = qt.QVBoxLayout(analysis_tab)

        introduction = qt.QLabel(
            "Choisissez le dossier des vidéos à traiter. BestShotAI sélectionne les frames "
            "les plus nettes dans chaque fenêtre temporelle et les exporte localement."
        )
        introduction.setWordWrap(True)
        layout.addWidget(introduction)

        folder_row = qt.QHBoxLayout()
        self._folder_path = qt.QLineEdit()
        self._folder_path.setReadOnly(True)
        self._folder_path.setPlaceholderText("Aucun dossier sélectionné")
        self._choose_folder_button = qt.QPushButton("Choisir un dossier…")
        self._choose_folder_button.clicked.connect(self._choose_folder)
        folder_row.addWidget(self._folder_path)
        folder_row.addWidget(self._choose_folder_button)
        layout.addLayout(folder_row)

        self._video_list = qt.QListWidget()
        self._video_list.setMinimumHeight(180)
        self._video_list.setSelectionMode(qt.QAbstractItemView.SelectionMode.NoSelection)
        layout.addWidget(self._video_list)

        self._analyse_button = qt.QPushButton("Démarrer le traitement")
        self._analyse_button.setEnabled(False)
        self._analyse_button.clicked.connect(self._start_analysis)
        layout.addWidget(self._analyse_button)

        self._progress = qt.QProgressBar()
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        layout.addWidget(self._progress)
        self._status = qt.QLabel("Choisissez un dossier contenant des vidéos.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._tabs.addTab(analysis_tab, "Analyse vidéo")
        self._tabs.addTab(self._create_learning_tab(), "Apprentissage IA")
        self._tabs.addTab(self._create_ai_selection_tab(), "Sélection IA")
        self._window.setCentralWidget(root)

    def show(self) -> None:
        self._window.resize(820, 600)
        self._window.show()

    def _create_learning_tab(self) -> Any:
        tab = self._qt.QWidget()
        layout = self._qt.QVBoxLayout(tab)
        explanation = self._qt.QLabel(
            "Choisissez le dossier « bestshot-candidates » produit par l'analyse. "
            "Vos choix restent locaux dans SQLite et servent à la sélection IA."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        folder_row = self._qt.QHBoxLayout()
        self._candidate_folder_path = self._qt.QLineEdit()
        self._candidate_folder_path.setReadOnly(True)
        self._candidate_folder_path.setPlaceholderText("Aucun dossier bestshot-candidates sélectionné")
        self._choose_candidate_folder_button = self._qt.QPushButton("Choisir un dossier…")
        self._choose_candidate_folder_button.clicked.connect(self._choose_candidate_folder)
        folder_row.addWidget(self._candidate_folder_path)
        folder_row.addWidget(self._choose_candidate_folder_button)
        layout.addLayout(folder_row)

        self._candidate_image = self._qt.QLabel("Choisissez un dossier de candidates.")
        self._candidate_image.setAlignment(self._qt.Qt.AlignmentFlag.AlignCenter)
        self._candidate_image.setMinimumHeight(360)
        layout.addWidget(self._candidate_image)
        self._candidate_details = self._qt.QLabel()
        self._candidate_details.setWordWrap(True)
        layout.addWidget(self._candidate_details)

        actions = self._qt.QHBoxLayout()
        self._keep_button = self._qt.QPushButton("Accepter")
        self._keep_button.clicked.connect(lambda: self._label_current_candidate(FrameLabel.KEEP))
        self._reject_button = self._qt.QPushButton("Rejeter")
        self._reject_button.clicked.connect(lambda: self._label_current_candidate(FrameLabel.REJECT))
        self._skip_button = self._qt.QPushButton("Passer")
        self._skip_button.clicked.connect(lambda: self._label_current_candidate(FrameLabel.SKIP))
        actions.addWidget(self._keep_button)
        actions.addWidget(self._reject_button)
        actions.addWidget(self._skip_button)
        layout.addLayout(actions)
        self._learning_status = self._qt.QLabel(
            "Aucun modèle n'est entraîné ici : cet onglet enregistre seulement vos labels."
        )
        self._learning_status.setWordWrap(True)
        layout.addWidget(self._learning_status)
        self._set_learning_actions_enabled(False)
        return tab

    def _create_ai_selection_tab(self) -> Any:
        """Offre uniquement le dossier source et le lancement du traitement personnel."""
        tab = self._qt.QWidget()
        layout = self._qt.QVBoxLayout(tab)
        folder_row = self._qt.QHBoxLayout()
        self._ai_selection_folder_path = self._qt.QLineEdit()
        self._ai_selection_folder_path.setReadOnly(True)
        self._ai_selection_folder_path.setPlaceholderText("Aucun dossier de vidéos sélectionné")
        self._choose_ai_selection_folder_button = self._qt.QPushButton("Choisir un dossier…")
        self._choose_ai_selection_folder_button.clicked.connect(self._choose_ai_selection_folder)
        folder_row.addWidget(self._ai_selection_folder_path)
        folder_row.addWidget(self._choose_ai_selection_folder_button)
        layout.addLayout(folder_row)
        self._ai_selection_button = self._qt.QPushButton("Démarrer le traitement")
        self._ai_selection_button.setEnabled(False)
        self._ai_selection_button.clicked.connect(self._start_ai_selection)
        layout.addWidget(self._ai_selection_button)
        layout.addStretch()
        return tab

    def _choose_folder(self) -> None:
        directory = self._qt.QFileDialog.getExistingDirectory(
            self._window,
            "Choisir le dossier des vidéos",
            self._folder_path.text() or str(Path.cwd()),
        )
        if not directory:
            return
        try:
            self._set_videos(discover_videos(Path(directory)))
        except VideoLibraryError as error:
            self._set_videos(())
            self._status.setText(str(error))
            return
        self._folder_path.setText(directory)
        if self._videos:
            self._status.setText(f"{len(self._videos)} vidéo(s) prête(s) à traiter localement.")
        else:
            self._status.setText("Aucune vidéo compatible dans ce dossier.")

    def _set_videos(self, videos: tuple[Path, ...]) -> None:
        self._videos = videos
        self._video_list.clear()
        for video in videos:
            self._video_list.addItem(video.name)
        self._analyse_button.setEnabled(bool(videos) and self._analysis_thread is None)

    def _choose_candidate_folder(self) -> None:
        directory = self._qt.QFileDialog.getExistingDirectory(
            self._window,
            "Choisir le dossier bestshot-candidates",
            self._candidate_folder_path.text() or str(Path.cwd()),
        )
        if not directory:
            return
        self._candidate_folder_path.setText(directory)
        self._start_candidate_load(Path(directory))

    def _start_candidate_load(self, directory: Path) -> None:
        if self._candidate_load_thread is not None or self._candidate_label_thread is not None:
            return
        self._set_learning_actions_enabled(False)
        self._choose_candidate_folder_button.setEnabled(False)
        self._learning_status.setText("Chargement local des candidates indexées…")
        self._candidate_load_thread = self._qt.QThread()
        self._candidate_load_worker = _CandidateLoadWorker(directory, self._labeling_service_factory)
        self._candidate_load_worker.moveToThread(self._candidate_load_thread)
        self._candidate_load_thread.started.connect(self._candidate_load_worker.run)
        connection = self._qt.Qt.ConnectionType.QueuedConnection
        self._candidate_load_worker.completed.connect(
            self._learning_events.finish_candidate_load, connection
        )
        self._candidate_load_thread.finished.connect(
            self._learning_events.release_candidate_load_thread, connection
        )
        self._candidate_load_thread.start()

    def _finish_candidate_load(self, summary: CandidateLoadSummary) -> None:
        self._stop_candidate_load_thread()
        self._choose_candidate_folder_button.setEnabled(True)
        if summary.error_message is not None:
            self._learning_items = ()
            self._candidate_image.setPixmap(self._qt.QPixmap())
            self._candidate_image.setText("Aucune candidate disponible.")
            self._candidate_details.clear()
            self._learning_status.setText(summary.error_message)
            return
        self._learning_items = summary.items
        self._learning_index = next(
            (
                index
                for index, item in enumerate(self._learning_items)
                if item.frame.label is FrameLabel.SKIP
            ),
            0,
        )
        self._show_current_candidate()

    def _show_current_candidate(self) -> None:
        if not self._learning_items:
            self._set_learning_actions_enabled(False)
            return
        item = self._learning_items[self._learning_index]
        preview_path = Path(item.frame.preview_reference)
        pixmap = self._qt.QPixmap(str(preview_path))
        if pixmap.isNull():
            self._candidate_image.setPixmap(self._qt.QPixmap())
            self._candidate_image.setText(f"Aperçu local indisponible : {preview_path.name}")
        else:
            self._candidate_image.setText("")
            self._candidate_image.setPixmap(
                pixmap.scaled(
                    720,
                    360,
                    self._qt.Qt.AspectRatioMode.KeepAspectRatio,
                    self._qt.Qt.TransformationMode.SmoothTransformation,
                )
            )
        frame = item.frame
        self._candidate_details.setText(
            f"{item.video_path.name} — {self._learning_index + 1}/{len(self._learning_items)} — "
            f"{frame.timestamp:.2f} s — frame {frame.frame_index} — netteté {frame.sharpness:.2f}"
        )
        self._learning_status.setText(
            "Choisissez Accepter, Rejeter ou Passer. Les choix sont enregistrés localement."
        )
        self._set_learning_actions_enabled(True)

    def _label_current_candidate(self, label: FrameLabel) -> None:
        if not self._learning_items or self._candidate_label_thread is not None:
            return
        frame_id = self._learning_items[self._learning_index].frame.id
        if frame_id is None:
            self._learning_status.setText("Candidate sans identifiant SQLite.")
            return
        self._set_learning_actions_enabled(False)
        self._learning_status.setText("Enregistrement local du choix…")
        self._candidate_label_thread = self._qt.QThread()
        self._candidate_label_worker = _CandidateLabelWorker(
            frame_id, label, self._labeling_service_factory
        )
        self._candidate_label_worker.moveToThread(self._candidate_label_thread)
        self._candidate_label_thread.started.connect(self._candidate_label_worker.run)
        connection = self._qt.Qt.ConnectionType.QueuedConnection
        self._candidate_label_worker.completed.connect(
            self._learning_events.finish_candidate_label, connection
        )
        self._candidate_label_thread.finished.connect(
            self._learning_events.release_candidate_label_thread, connection
        )
        self._candidate_label_thread.start()

    def _finish_candidate_label(self, summary: CandidateLabelSummary) -> None:
        self._stop_candidate_label_thread()
        if summary.error_message is not None:
            self._learning_status.setText(summary.error_message)
            self._set_learning_actions_enabled(True)
            return
        if self._learning_index + 1 == len(self._learning_items):
            self._learning_status.setText("Toutes les candidates ont été parcourues.")
            self._set_learning_actions_enabled(False)
            return
        self._learning_index += 1
        self._show_current_candidate()

    def _set_learning_actions_enabled(self, enabled: bool) -> None:
        self._keep_button.setEnabled(enabled)
        self._reject_button.setEnabled(enabled)
        self._skip_button.setEnabled(enabled)

    def _choose_ai_selection_folder(self) -> None:
        directory = self._qt.QFileDialog.getExistingDirectory(
            self._window,
            "Choisir le dossier des vidéos à sélectionner",
            self._ai_selection_folder_path.text() or str(Path.cwd()),
        )
        if not directory:
            return
        try:
            self._ai_selection_videos = discover_videos(Path(directory))
        except VideoLibraryError:
            self._ai_selection_videos = ()
        self._ai_selection_folder_path.setText(directory)
        self._ai_selection_button.setEnabled(bool(self._ai_selection_videos))

    def _start_ai_selection(self) -> None:
        if not self._ai_selection_videos or self._ai_selection_thread is not None:
            return
        self._choose_ai_selection_folder_button.setEnabled(False)
        self._ai_selection_button.setEnabled(False)
        self._ai_selection_thread = self._qt.QThread()
        self._ai_selection_worker = _AISelectionWorker(
            self._ai_selection_videos, self._ai_selection_service_factory
        )
        self._ai_selection_worker.moveToThread(self._ai_selection_thread)
        self._ai_selection_thread.started.connect(self._ai_selection_worker.run)
        connection = self._qt.Qt.ConnectionType.QueuedConnection
        self._ai_selection_worker.completed.connect(
            self._ai_selection_events.finish_ai_selection, connection
        )
        self._ai_selection_thread.finished.connect(
            self._ai_selection_events.release_ai_selection_thread, connection
        )
        self._ai_selection_thread.start()

    def _finish_ai_selection(self, summary: AISelectionSummary) -> None:
        self._stop_ai_selection_thread()
        self._choose_ai_selection_folder_button.setEnabled(True)
        self._ai_selection_button.setEnabled(bool(self._ai_selection_videos))
        self._report_ai_selection(summary)

    def _report_ai_selection(self, summary: AISelectionSummary) -> None:
        if summary.error_message is not None:
            self._qt.QMessageBox.warning(self._window, "Sélection IA", summary.error_message)
            return
        exported_count = sum(len(result.exported_paths) for result in summary.results)
        self._qt.QMessageBox.information(
            self._window,
            "Sélection IA",
            f"Traitement terminé : {exported_count} photo(s) exportée(s) dans « bestshot-selection ».",
        )

    def _start_analysis(self) -> None:
        if not self._videos or self._analysis_thread is not None:
            return
        self._set_controls(False)
        self._progress.setRange(0, len(self._videos))
        self._progress.setValue(0)
        self._status.setText("Préparation de l'analyse locale…")

        self._analysis_thread = self._qt.QThread()
        self._analysis_worker = _AnalysisWorker(self._videos, self._runner_factory)
        self._analysis_worker.moveToThread(self._analysis_thread)
        self._analysis_thread.started.connect(self._analysis_worker.run)
        connection = self._qt.Qt.ConnectionType.QueuedConnection
        self._analysis_worker.video_started.connect(
            self._analysis_events.show_video_started, connection
        )
        self._analysis_worker.video_finished.connect(
            self._analysis_events.show_video_finished, connection
        )
        self._analysis_worker.video_failed.connect(
            self._analysis_events.show_video_failed, connection
        )
        self._analysis_worker.completed.connect(self._analysis_events.finish_analysis, connection)
        self._analysis_thread.finished.connect(
            self._analysis_events.release_analysis_thread, connection
        )
        self._analysis_thread.start()

    def _show_video_started(self, path: Path, position: int, total: int) -> None:
        self._status.setText(f"Traitement {position}/{total} : {path.name}")

    def _show_video_finished(self, path: Path, position: int, total: int) -> None:
        self._progress.setValue(position)
        self._status.setText(f"Candidates exportées {position}/{total} : {path.name}")

    def _show_video_failed(self, path: Path, message: str, position: int, total: int) -> None:
        self._progress.setValue(position)
        self._status.setText(f"Vidéo ignorée {position}/{total} : {path.name} — {message}")

    def _finish_analysis(self, summary: AnalysisSummary) -> None:
        self._set_controls(True)
        if summary.analyzed_paths:
            failures = f", {len(summary.failed_paths)} échec(s)" if summary.failed_paths else ""
            self._status.setText(
                f"Traitement terminé : {len(summary.analyzed_paths)} vidéo(s), candidates "
                f"exportées dans « {CANDIDATE_EXPORT_DIRECTORY_NAME} »{failures}."
            )
        else:
            error_detail = summary.failure_messages[0] if summary.failure_messages else "Erreur inconnue."
            self._status.setText(f"Aucune vidéo n'a pu être analysée. Erreur : {error_detail}")
        self._stop_analysis_thread()

    def _set_controls(self, enabled: bool) -> None:
        self._choose_folder_button.setEnabled(enabled)
        self._analyse_button.setEnabled(enabled and bool(self._videos))

    def _stop_analysis_thread(self) -> None:
        thread = self._analysis_thread
        if thread is None:
            return
        thread.requestInterruption()
        thread.quit()
        if thread.wait(2_000):
            self._release_analysis_thread()

    def _release_analysis_thread(self) -> None:
        if self._analysis_thread is not None and not self._analysis_thread.isRunning():
            self._analysis_thread = None
            self._analysis_worker = None

    def _stop_candidate_load_thread(self) -> None:
        thread = self._candidate_load_thread
        if thread is None:
            return
        thread.requestInterruption()
        thread.quit()
        if thread.wait(2_000):
            self._release_candidate_load_thread()

    def _release_candidate_load_thread(self) -> None:
        if self._candidate_load_thread is not None and not self._candidate_load_thread.isRunning():
            self._candidate_load_thread = None
            self._candidate_load_worker = None

    def _stop_candidate_label_thread(self) -> None:
        thread = self._candidate_label_thread
        if thread is None:
            return
        thread.requestInterruption()
        thread.quit()
        if thread.wait(2_000):
            self._release_candidate_label_thread()

    def _release_candidate_label_thread(self) -> None:
        if self._candidate_label_thread is not None and not self._candidate_label_thread.isRunning():
            self._candidate_label_thread = None
            self._candidate_label_worker = None

    def _stop_ai_selection_thread(self) -> None:
        thread = self._ai_selection_thread
        if thread is None:
            return
        thread.requestInterruption()
        thread.quit()
        if thread.wait(2_000):
            self._release_ai_selection_thread()

    def _release_ai_selection_thread(self) -> None:
        if self._ai_selection_thread is not None and not self._ai_selection_thread.isRunning():
            self._ai_selection_thread = None
            self._ai_selection_worker = None

    def _request_close(self) -> bool:
        for thread in (
            self._analysis_thread,
            self._candidate_load_thread,
            self._candidate_label_thread,
            self._ai_selection_thread,
        ):
            if thread is None or not thread.isRunning():
                continue
            thread.requestInterruption()
            self._status.setText(
                "Annulation demandée : le traitement en cours doit se terminer avant fermeture."
            )
            return False
        self._stop_analysis_thread()
        self._stop_candidate_load_thread()
        self._stop_candidate_label_thread()
        self._stop_ai_selection_thread()
        return True


def _window_with_close_handler(close_handler: Callable[[], bool]) -> Any:
    """Crée une fenêtre qui ne laisse jamais Qt détruire un thread actif."""
    qt = _qt()

    class MainWindow(qt.QMainWindow):
        def closeEvent(self, event: Any) -> None:
            if close_handler():
                event.accept()
                return
            event.ignore()

    return MainWindow()


class _AnalysisEventBridge:
    """Ramène les signaux du worker dans le thread propriétaire des widgets."""

    def __new__(cls, window: VideoLibraryWindow) -> Any:
        qt = _qt()

        class Bridge(qt.QObject):
            @qt.Slot(object, int, int)
            def show_video_started(self, path: Path, position: int, total: int) -> None:
                window._show_video_started(path, position, total)

            @qt.Slot(object, int, int)
            def show_video_finished(self, path: Path, position: int, total: int) -> None:
                window._show_video_finished(path, position, total)

            @qt.Slot(object, str, int, int)
            def show_video_failed(
                self, path: Path, message: str, position: int, total: int
            ) -> None:
                window._show_video_failed(path, message, position, total)

            @qt.Slot(object)
            def finish_analysis(self, summary: AnalysisSummary) -> None:
                window._finish_analysis(summary)

            @qt.Slot()
            def release_analysis_thread(self) -> None:
                window._release_analysis_thread()

        return Bridge()


class _AnalysisWorker:
    """Exécute l'analyse et les accès SQLite hors du thread UI."""

    def __new__(
        cls,
        video_paths: tuple[Path, ...],
        runner_factory: Callable[[], VideoEmbeddingRunner],
    ) -> Any:
        qt = _qt()

        class Worker(qt.QObject):
            video_started = qt.Signal(object, int, int)
            video_finished = qt.Signal(object, int, int)
            video_failed = qt.Signal(object, str, int, int)
            completed = qt.Signal(object)

            def __init__(self) -> None:
                super().__init__()
                self._video_paths = video_paths
                self._runner_factory = runner_factory

            @qt.Slot()
            def run(self) -> None:
                analyzed: list[Path] = []
                failed: list[Path] = []
                failure_messages: list[str] = []
                total = len(self._video_paths)
                try:
                    runner = self._runner_factory()
                except _ANALYSIS_ERRORS as error:
                    for position, path in enumerate(self._video_paths, start=1):
                        failed.append(path)
                        failure_messages.append(str(error))
                        self.video_failed.emit(path, str(error), position, total)
                    self.completed.emit(AnalysisSummary((), tuple(failed), tuple(failure_messages)))
                    return
                for position, path in enumerate(self._video_paths, start=1):
                    if qt.QThread.currentThread().isInterruptionRequested():
                        break
                    self.video_started.emit(path, position, total)
                    try:
                        runner.run(path)
                    except _ANALYSIS_ERRORS as error:
                        failed.append(path)
                        failure_messages.append(str(error))
                        self.video_failed.emit(path, str(error), position, total)
                    else:
                        analyzed.append(path)
                        self.video_finished.emit(path, position, total)
                self.completed.emit(AnalysisSummary(tuple(analyzed), tuple(failed), tuple(failure_messages)))

        return Worker()


class _LearningEventBridge:
    """Reçoit les réponses SQLite dans le thread propriétaire des widgets Qt."""

    def __new__(cls, window: VideoLibraryWindow) -> Any:
        qt = _qt()

        class Bridge(qt.QObject):
            @qt.Slot(object)
            def finish_candidate_load(self, summary: CandidateLoadSummary) -> None:
                window._finish_candidate_load(summary)

            @qt.Slot()
            def release_candidate_load_thread(self) -> None:
                window._release_candidate_load_thread()

            @qt.Slot(object)
            def finish_candidate_label(self, summary: CandidateLabelSummary) -> None:
                window._finish_candidate_label(summary)

            @qt.Slot()
            def release_candidate_label_thread(self) -> None:
                window._release_candidate_label_thread()

        return Bridge()


class _CandidateLoadWorker:
    """Charge les candidates et leurs labels depuis SQLite hors du thread UI."""

    def __new__(
        cls,
        directory: Path,
        service_factory: Callable[[], CandidateLabelingService],
    ) -> Any:
        qt = _qt()

        class Worker(qt.QObject):
            completed = qt.Signal(object)

            def __init__(self) -> None:
                super().__init__()
                self._directory = directory
                self._service_factory = service_factory

            @qt.Slot()
            def run(self) -> None:
                try:
                    items = self._service_factory().list_candidates(self._directory)
                except (CandidateLabelingError, DatasetRepositoryError, OSError, ValueError) as error:
                    self.completed.emit(CandidateLoadSummary((), str(error)))
                    return
                self.completed.emit(CandidateLoadSummary(items))

        return Worker()


class _CandidateLabelWorker:
    """Écrit un unique label local hors du thread UI."""

    def __new__(
        cls,
        frame_id: int,
        label: FrameLabel,
        service_factory: Callable[[], CandidateLabelingService],
    ) -> Any:
        qt = _qt()

        class Worker(qt.QObject):
            completed = qt.Signal(object)

            def __init__(self) -> None:
                super().__init__()
                self._frame_id = frame_id
                self._label = label
                self._service_factory = service_factory

            @qt.Slot()
            def run(self) -> None:
                try:
                    self._service_factory().set_label(self._frame_id, self._label)
                except (CandidateLabelingError, DatasetRepositoryError, OSError, ValueError) as error:
                    self.completed.emit(CandidateLabelSummary(self._label, str(error)))
                    return
                self.completed.emit(CandidateLabelSummary(self._label))

        return Worker()


class _AISelectionEventBridge:
    """Reçoit le résultat de la sélection personnelle hors du thread UI."""

    def __new__(cls, window: VideoLibraryWindow) -> Any:
        qt = _qt()

        class Bridge(qt.QObject):
            @qt.Slot(object)
            def finish_ai_selection(self, summary: AISelectionSummary) -> None:
                window._finish_ai_selection(summary)

            @qt.Slot()
            def release_ai_selection_thread(self) -> None:
                window._release_ai_selection_thread()

        return Bridge()


class _AISelectionWorker:
    """Entraîne la tête locale et exporte les candidates KEEP hors du thread UI."""

    def __new__(
        cls,
        video_paths: tuple[Path, ...],
        service_factory: Callable[[], LabelDrivenSelectionService],
    ) -> Any:
        qt = _qt()

        class Worker(qt.QObject):
            completed = qt.Signal(object)

            def __init__(self) -> None:
                super().__init__()
                self._video_paths = video_paths
                self._service_factory = service_factory

            @qt.Slot()
            def run(self) -> None:
                try:
                    service = self._service_factory()
                    service.train()
                    results: list[LabelSelectionResult] = []
                    for path in self._video_paths:
                        if qt.QThread.currentThread().isInterruptionRequested():
                            break
                        results.append(service.select_video(path))
                except (
                    DatasetRepositoryError,
                    LabelSelectionError,
                    OSError,
                    PersonalLabelModelError,
                    RuntimeError,
                    ValueError,
                ) as error:
                    self.completed.emit(AISelectionSummary((), str(error)))
                    return
                self.completed.emit(AISelectionSummary(tuple(results)))

        return Worker()


def _create_embedding_runner() -> VideoEmbeddingRunner:
    """Assemble le pipeline V2, utilisé exclusivement par le worker d'analyse."""
    presampling_settings = load_presampling_settings()
    embedding_settings = load_embedding_settings()
    dataset_settings = load_dataset_settings()
    generator = CandidateGenerator(
        TemporalSampler(PyAVTemporalSamplingBackend(), presampling_settings),
        SharpnessRanker(),
        presampling_settings,
    )
    return VideoEmbeddingRunner(
        generator,
        PyAVCandidatePreviewReader(),
        DINOv2EmbeddingProvider(embedding_settings),
        EmbeddingCache(embedding_settings.embedding_cache_dir),
        presampling_settings.analysis_max_width,
        SQLiteDatasetRepository(dataset_settings.database_path),
        PreviewCache(dataset_settings.preview_cache_dir),
        PyAVSelectedFrameExporter(),
    )


def _create_candidate_labeling_service() -> CandidateLabelingService:
    """Assemble la collecte de labels, ouverte seulement dans le worker Qt dédié."""
    dataset_settings = load_dataset_settings()
    return CandidateLabelingService(SQLiteDatasetRepository(dataset_settings.database_path))


def _create_ai_selection_service() -> LabelDrivenSelectionService:
    """Assemble l'inférence IA sans ingérer les vidéos sélectionnées au dataset."""
    presampling_settings = load_presampling_settings()
    embedding_settings = load_embedding_settings()
    dataset_settings = load_dataset_settings()
    embedder = VideoEmbeddingRunner(
        CandidateGenerator(
            TemporalSampler(PyAVTemporalSamplingBackend(), presampling_settings),
            SharpnessRanker(),
            presampling_settings,
        ),
        PyAVCandidatePreviewReader(),
        DINOv2EmbeddingProvider(embedding_settings),
        EmbeddingCache(embedding_settings.embedding_cache_dir),
        presampling_settings.analysis_max_width,
    )
    return LabelDrivenSelectionService(
        SQLiteDatasetRepository(dataset_settings.database_path),
        embedder,
    )


def _qt() -> Any:
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
    except ImportError as error:
        raise DesktopApplicationError(
            "Installez les extras : pip install -e '.[desktop,embedding]'."
        ) from error

    class Qt:
        QApplication = QtWidgets.QApplication
        QAbstractItemView = QtWidgets.QAbstractItemView
        QFileDialog = QtWidgets.QFileDialog
        QHBoxLayout = QtWidgets.QHBoxLayout
        QLabel = QtWidgets.QLabel
        QLineEdit = QtWidgets.QLineEdit
        QListWidget = QtWidgets.QListWidget
        QMainWindow = QtWidgets.QMainWindow
        QMessageBox = QtWidgets.QMessageBox
        QProgressBar = QtWidgets.QProgressBar
        QPixmap = QtGui.QPixmap
        QPushButton = QtWidgets.QPushButton
        QTabWidget = QtWidgets.QTabWidget
        QThread = QtCore.QThread
        QVBoxLayout = QtWidgets.QVBoxLayout
        QWidget = QtWidgets.QWidget
        QObject = QtCore.QObject
        Qt = QtCore.Qt
        Signal = QtCore.Signal
        Slot = QtCore.Slot

    return Qt
