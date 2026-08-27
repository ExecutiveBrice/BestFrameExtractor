"""Écran PySide6 local de comparaison pairwise, avec accès SQLite hors du thread UI."""
# mypy: ignore-errors

from __future__ import annotations

from pathlib import Path
from typing import Any

from bestshot.dataset.repository import FrameRecord
from bestshot.dataset.sqlite_repository import DatasetRepositoryError, SQLiteDatasetRepository
from bestshot.domain.preferences import PreferenceChoice
from bestshot.learning.pair_generator import GeneratedPair, PairGenerationSettings
from bestshot.services.preferences import (
    PreferenceServiceError,
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
    window = PairwiseReviewWindow(database_path, video_path, settings, include_reviewed=include_reviewed)
    window.show()
    return application.exec()


class PairwiseReviewWindow:
    """Deux aperçus candidats avec décisions immédiates et raccourcis clavier."""

    def __init__(
        self,
        database_path: Path,
        video_path: Path,
        settings: PairGenerationSettings,
        *,
        include_reviewed: bool = False,
    ) -> None:
        qt = _qt()
        self._qt = qt
        self._database_path = database_path
        self._video_path = video_path
        self._settings = settings
        self._current_pair: GeneratedPair | None = None
        self._thread = qt.QThread()
        self._worker = _PreferenceWorker(database_path, video_path, settings, include_reviewed)
        self._worker.moveToThread(self._thread)
        self._thread.start()
        self._request_pair = _SignalEmitter()
        self._submit_choice = _SignalEmitter()
        self._request_pair.trigger.connect(self._worker.load_next, qt.Qt.ConnectionType.QueuedConnection)
        self._submit_choice.choice.connect(self._worker.save_and_load, qt.Qt.ConnectionType.QueuedConnection)
        self._worker.pair_ready.connect(self._show_pair)
        self._worker.empty.connect(self._show_empty)
        self._worker.failed.connect(self._show_error)

        self._window = qt.QMainWindow()
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
        self._window.destroyed.connect(self._shutdown)
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
        self._status.setText(f"Paires restantes proposées : {total}")
        self._metadata.setText(
            " | ".join(
                (
                    f"Premier : {first.timestamp:.3f}s · frame {first.frame_index}",
                    f"Second : {second.timestamp:.3f}s · frame {second.frame_index}",
                    f"Stratégie : {pair.reason}",
                )
            )
        )
        self._set_preview(self._first_image, first.preview_reference)
        self._set_preview(self._second_image, second.preview_reference)

    def _show_empty(self) -> None:
        self._current_pair = None
        self._set_actions_enabled(False)
        self._status.setText("Aucune nouvelle paire à comparer pour cette vidéo.")
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

    def _shutdown(self) -> None:
        self._thread.quit()
        self._thread.wait(2_000)


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
        video_path: Path,
        settings: PairGenerationSettings,
        include_reviewed: bool,
    ) -> Any:
        qt = _qt()

        class Worker(qt.QObject):
            pair_ready = qt.Signal(object, object, object, int)
            empty = qt.Signal()
            failed = qt.Signal(str)

            def __init__(self) -> None:
                super().__init__()
                self._repository = SQLiteDatasetRepository(database_path)
                self._include_reviewed = include_reviewed
                self._handled_pairs: set[tuple[int, int]] = set()

            @qt.Slot()
            def load_next(self) -> None:
                try:
                    pairs = generate_video_preferences(
                        self._repository,
                        video_path,
                        settings,
                        include_reviewed=self._include_reviewed,
                    )
                    pairs = [
                        pair
                        for pair in pairs
                        if (pair.first_frame_id, pair.second_frame_id) not in self._handled_pairs
                    ]
                    if not pairs:
                        self.empty.emit()
                        return
                    pair = pairs[0]
                    frames = self._repository.get_frames_by_ids(
                        {pair.first_frame_id, pair.second_frame_id}
                    )
                    first = frames[pair.first_frame_id]
                    second = frames[pair.second_frame_id]
                    self.pair_ready.emit(pair, first, second, len(pairs))
                except (DatasetRepositoryError, PreferenceServiceError, RuntimeError, KeyError, ValueError) as error:
                    self.failed.emit(str(error))

            @qt.Slot(int, int, object)
            def save_and_load(self, first_id: int, second_id: int, choice: PreferenceChoice) -> None:
                try:
                    record_preference(self._repository, first_id, second_id, choice)
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
