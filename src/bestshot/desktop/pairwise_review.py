"""Écran PySide6 local de comparaison pairwise, avec accès SQLite hors du thread UI."""
# mypy: ignore-errors

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bestshot.dataset.repository import FrameRecord
from bestshot.dataset.sqlite_repository import DatasetRepositoryError, SQLiteDatasetRepository
from bestshot.domain.preferences import PreferenceChoice
from bestshot.learning.pair_generator import GeneratedPair, PairGenerationSettings
from bestshot.services.preferences import (
    PreferenceServiceError,
    generate_photo_pool_preferences,
    generate_video_preferences,
    record_preference,
)


class PreferenceWindowError(RuntimeError):
    """L'écran de préférences ne peut pas être démarré dans cet environnement."""


def run_pairwise_review(
    database_path: Path,
    video_path: Path,
    settings: PairGenerationSettings,
    *,
    include_reviewed: bool = False,
) -> int:
    """Démarre l'écran optionnel de comparaison, strictement local."""
    qt = _qt()
    application = qt.QApplication.instance() or qt.QApplication([])
    window = PairwiseReviewWindow(database_path, (video_path,), settings, include_reviewed=include_reviewed)
    window.show()
    return application.exec()


class PairwiseReviewWindow:
    """Deux aperçus candidats avec décisions immédiates et raccourcis clavier."""

    def __init__(
        self,
        database_path: Path,
        video_paths: Sequence[Path],
        settings: PairGenerationSettings,
        *,
        include_reviewed: bool = False,
        source_kind: str = "video",
    ) -> None:
        qt = _qt()
        self._qt = qt
        self._database_path = database_path
        self._video_paths = tuple(video_paths)
        if not self._video_paths:
            raise ValueError("Au moins une source locale est requise pour la comparaison.")
        if source_kind not in {"video", "photo_pool"}:
            raise ValueError("Le type de source de comparaison est inconnu.")
        self._source_kind = source_kind
        self._settings = settings
        self._current_pair: GeneratedPair | None = None
        self._events = _ReviewEventBridge(self)
        self._thread = qt.QThread()
        self._worker = _PreferenceWorker(
            database_path,
            self._video_paths,
            settings,
            include_reviewed,
            source_kind,
        )
        self._worker.moveToThread(self._thread)
        self._request_pair = _SignalEmitter()
        self._submit_choice = _SignalEmitter()
        connection = qt.Qt.ConnectionType.QueuedConnection
        self._request_pair.trigger.connect(self._worker.load_next, qt.Qt.ConnectionType.QueuedConnection)
        self._submit_choice.choice.connect(self._worker.save_and_load, qt.Qt.ConnectionType.QueuedConnection)
        self._worker.pair_ready.connect(self._events.show_pair, connection)
        self._worker.empty.connect(self._events.show_empty, connection)
        self._worker.failed.connect(self._events.show_error, connection)

        self._window = _window_with_close_handler(self._request_close)
        self._window.setWindowTitle("BestShotAI — préférences personnelles")
        root = qt.QWidget()
        layout = qt.QVBoxLayout(root)
        self._status = qt.QLabel("Chargement des paires locales…")
        layout.addWidget(self._status)
        previews = qt.QHBoxLayout()
        self._first_image = self._image_label()
        self._second_image = self._image_label()
        previews.addWidget(self._first_image)
        previews.addWidget(self._second_image)
        layout.addLayout(previews)
        self._metadata = qt.QLabel()
        layout.addWidget(self._metadata)
        buttons = qt.QHBoxLayout()
        self._first_button = self._button("← Premier", PreferenceChoice.FIRST)
        self._second_button = self._button("Second →", PreferenceChoice.SECOND)
        self._equal_button = self._button("Égal (Espace)", PreferenceChoice.EQUAL)
        self._skip_button = self._button("Passer (Échap)", PreferenceChoice.SKIP)
        for button in (self._first_button, self._second_button, self._equal_button, self._skip_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self._window.setCentralWidget(root)
        self._add_shortcuts()
        self._thread.start()
        self._request_pair.trigger.emit()

    def show(self) -> None:
        self._window.resize(1200, 700)
        self._window.show()

    def _image_label(self) -> Any:
        label = self._qt.QLabel()
        label.setAlignment(self._qt.Qt.AlignmentFlag.AlignCenter)
        label.setMinimumSize(400, 300)
        return label

    def _button(self, text: str, choice: PreferenceChoice) -> Any:
        button = self._qt.QPushButton(text)
        button.clicked.connect(lambda: self._choose(choice))
        return button

    def _add_shortcuts(self) -> None:
        shortcuts = (
            (self._qt.Qt.Key.Key_Left, PreferenceChoice.FIRST),
            (self._qt.Qt.Key.Key_Right, PreferenceChoice.SECOND),
            (self._qt.Qt.Key.Key_Space, PreferenceChoice.EQUAL),
            (self._qt.Qt.Key.Key_Escape, PreferenceChoice.SKIP),
        )
        for key, choice in shortcuts:
            shortcut = self._qt.QShortcut(self._qt.QKeySequence(key), self._window)
            shortcut.activated.connect(lambda selected=choice: self._choose(selected))

    def _choose(self, choice: PreferenceChoice) -> None:
        if self._current_pair is None:
            return
        self._set_actions_enabled(False)
        self._status.setText("Enregistrement local…")
        self._submit_choice.choice.emit(
            self._current_pair.first_frame_id,
            self._current_pair.second_frame_id,
            choice,
        )

    def _show_pair(self, pair: GeneratedPair, first: FrameRecord, second: FrameRecord, total: int) -> None:
        self._current_pair = pair
        self._set_actions_enabled(True)
        self._status.setText(f"Paires disponibles restantes : {total}")
        self._metadata.setText(
            " | ".join(
                (
                    f"{'Pool photo' if self._source_kind == 'photo_pool' else 'Vidéo'} : {first.video_id}",
                    self._frame_description("Premier", first),
                    self._frame_description("Second", second),
                    f"Stratégie : {pair.reason}",
                )
            )
        )
        self._set_preview(self._first_image, first.preview_reference)
        self._set_preview(self._second_image, second.preview_reference)

    def _show_empty(self) -> None:
        self._current_pair = None
        self._set_actions_enabled(False)
        source_label = "ce pool de photos" if self._source_kind == "photo_pool" else "cette vidéo"
        self._status.setText(f"Aucune nouvelle paire à comparer pour {source_label}.")
        self._metadata.clear()
        self._first_image.clear()
        self._second_image.clear()

    def _show_error(self, message: str) -> None:
        self._current_pair = None
        self._set_actions_enabled(False)
        self._status.setText(message)

    def _set_preview(self, target: Any, reference: str) -> None:
        pixmap = self._qt.QPixmap(reference)
        target.setPixmap(
            pixmap.scaled(
                target.size(),
                self._qt.Qt.AspectRatioMode.KeepAspectRatio,
                self._qt.Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _set_actions_enabled(self, enabled: bool) -> None:
        for button in (self._first_button, self._second_button, self._equal_button, self._skip_button):
            button.setEnabled(enabled)

    def _frame_description(self, label: str, frame: FrameRecord) -> str:
        if self._source_kind == "photo_pool":
            return f"{label} : photo {frame.frame_index + 1}"
        return f"{label} : {frame.timestamp:.3f}s · frame {frame.frame_index}"

    def _request_close(self) -> bool:
        """Attend l'arrêt du worker avant que Qt puisse détruire sa fenêtre."""
        self._thread.quit()
        if self._thread.wait(2_000):
            return True
        self._status.setText("Fermeture en attente de l'enregistrement local…")
        return False


def _window_with_close_handler(close_handler: Any) -> Any:
    """Intercepte Close pour ne jamais laisser un QThread survivre à sa fenêtre."""
    qt = _qt()

    class MainWindow(qt.QMainWindow):
        def closeEvent(self, event: Any) -> None:
            if close_handler():
                event.accept()
                return
            event.ignore()

    return MainWindow()


class _ReviewEventBridge:
    """Assure que les aperçus et les boutons sont mis à jour dans le thread UI."""

    def __new__(cls, window: PairwiseReviewWindow) -> Any:
        qt = _qt()

        class Bridge(qt.QObject):
            @qt.Slot(object, object, object, int)
            def show_pair(
                self, pair: GeneratedPair, first: FrameRecord, second: FrameRecord, total: int
            ) -> None:
                window._show_pair(pair, first, second, total)

            @qt.Slot()
            def show_empty(self) -> None:
                window._show_empty()

            @qt.Slot(str)
            def show_error(self, message: str) -> None:
                window._show_error(message)

        return Bridge()


class _SignalEmitter:
    """Pont de signaux créé dynamiquement pour ne charger PySide6 qu'à l'exécution."""

    def __new__(cls) -> Any:
        qt = _qt()

        class Emitter(qt.QObject):
            trigger = qt.Signal()
            choice = qt.Signal(int, int, object)

        return Emitter()


class _PreferenceWorker:
    """Exécute la lecture et l'écriture SQLite sur un thread de travail."""

    def __new__(
        cls,
        database_path: Path,
        video_paths: tuple[Path, ...],
        settings: PairGenerationSettings,
        include_reviewed: bool,
        source_kind: str,
    ) -> Any:
        qt = _qt()

        class Worker(qt.QObject):
            pair_ready = qt.Signal(object, object, object, int)
            empty = qt.Signal()
            failed = qt.Signal(str)

            def __init__(self) -> None:
                super().__init__()
                self._database_path = database_path
                self._repository: SQLiteDatasetRepository | None = None
                self._video_paths = video_paths
                self._include_reviewed = include_reviewed
                self._source_kind = source_kind
                self._handled_pairs: set[tuple[int, int]] = set()

            def _get_repository(self) -> SQLiteDatasetRepository:
                """Ouvre SQLite à la première requête, une fois dans le worker."""
                if self._repository is None:
                    self._repository = SQLiteDatasetRepository(self._database_path)
                return self._repository

            @qt.Slot()
            def load_next(self) -> None:
                errors: list[str] = []
                for video_path in self._video_paths:
                    try:
                        repository = self._get_repository()
                        if self._source_kind == "photo_pool":
                            pairs = generate_photo_pool_preferences(
                                repository,
                                video_path,
                                settings,
                                include_reviewed=self._include_reviewed,
                                return_all=True,
                            )
                        else:
                            pairs = generate_video_preferences(
                                repository,
                                video_path,
                                settings,
                                include_reviewed=self._include_reviewed,
                                return_all=True,
                            )
                        pairs = [
                            pair
                            for pair in pairs
                            if (pair.first_frame_id, pair.second_frame_id) not in self._handled_pairs
                        ]
                        if not pairs:
                            continue
                        pair = pairs[0]
                        frames = repository.get_frames_by_ids(
                            {pair.first_frame_id, pair.second_frame_id}
                        )
                        first = frames[pair.first_frame_id]
                        second = frames[pair.second_frame_id]
                        self.pair_ready.emit(pair, first, second, len(pairs))
                        return
                    except (
                        DatasetRepositoryError,
                        PreferenceServiceError,
                        RuntimeError,
                        KeyError,
                        ValueError,
                    ) as error:
                        errors.append(str(error))
                if errors and len(self._video_paths) == 1:
                    self.failed.emit(errors[0])
                    return
                self.empty.emit()

            @qt.Slot(int, int, object)
            def save_and_load(self, first_id: int, second_id: int, choice: PreferenceChoice) -> None:
                try:
                    record_preference(self._get_repository(), first_id, second_id, choice)
                except (DatasetRepositoryError, RuntimeError, ValueError) as error:
                    self.failed.emit(str(error))
                    return
                self._handled_pairs.add((min(first_id, second_id), max(first_id, second_id)))
                self.load_next()

        return Worker()


def _qt() -> Any:
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
    except ImportError as error:
        raise PreferenceWindowError("Installez l'extra : pip install -e '.[desktop]'.") from error

    class Qt:
        QApplication = QtWidgets.QApplication
        QHBoxLayout = QtWidgets.QHBoxLayout
        QLabel = QtWidgets.QLabel
        QMainWindow = QtWidgets.QMainWindow
        QPixmap = QtGui.QPixmap
        QPushButton = QtWidgets.QPushButton
        QShortcut = QtGui.QShortcut
        QKeySequence = QtGui.QKeySequence
        QThread = QtCore.QThread
        QVBoxLayout = QtWidgets.QVBoxLayout
        QWidget = QtWidgets.QWidget
        QObject = QtCore.QObject
        Qt = QtCore.Qt
        Signal = QtCore.Signal
        Slot = QtCore.Slot

    return Qt
