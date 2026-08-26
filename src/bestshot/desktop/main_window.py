"""Fenêtre principale de l'application de bureau locale."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QUrl, Slot
from PySide6.QtGui import QDesktopServices, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from bestshot.desktop.factory import (
    DesktopSelectionParameters,
    create_batch_export_runner,
    load_desktop_selection_parameters,
)
from bestshot.desktop.worker import DesktopProcessingJob, ProcessingWorker
from bestshot.services.batch import BatchProgress, BatchResult


class MainWindow(QMainWindow):
    """Permet de lancer un lot local et de consulter les photos exportées une à une."""

    def __init__(self) -> None:
        super().__init__()
        self._thread: QThread | None = None
        self._worker: ProcessingWorker | None = None
        self._photos: tuple[Path, ...] = ()
        self._photo_index = 0

        self.setWindowTitle("BestShotAI")
        self.setMinimumSize(860, 640)
        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(12)

        introduction = QLabel(
            "Sélection locale des meilleures images de vos vidéos. "
            "Les vidéos restent sur cet ordinateur."
        )
        introduction.setWordWrap(True)
        layout.addWidget(introduction)

        form = QFormLayout()
        self.source_input = QLineEdit()
        self.source_input.setPlaceholderText("Dossier contenant les vidéos")
        source_row = self._path_row(self.source_input, "Parcourir les vidéos", self._choose_source_directory)
        form.addRow("Vidéos source", source_row)

        self.output_input = QLineEdit()
        self.output_input.setPlaceholderText("Dossier où enregistrer les photos")
        output_row = self._path_row(self.output_input, "Parcourir la destination", self._choose_output_directory)
        form.addRow("Photos exportées", output_row)

        defaults = load_desktop_selection_parameters()
        self.minimum_score_input = QDoubleSpinBox()
        self.minimum_score_input.setRange(0.0, 1.0)
        self.minimum_score_input.setDecimals(2)
        self.minimum_score_input.setSingleStep(0.05)
        self.minimum_score_input.setValue(defaults.minimum_score)
        self.minimum_score_input.setSuffix(" / 1")
        self.minimum_score_input.setToolTip("Plus ce score est élevé, moins de photos seront retenues.")
        form.addRow("Score de qualité minimal", self.minimum_score_input)

        self.temporal_window_input = QSpinBox()
        self.temporal_window_input.setRange(100, 60_000)
        self.temporal_window_input.setSingleStep(100)
        self.temporal_window_input.setValue(defaults.temporal_window_ms)
        self.temporal_window_input.setSuffix(" ms")
        self.temporal_window_input.setToolTip(
            "Une fenêtre plus longue élimine davantage de photos similaires prises à la suite."
        )
        form.addRow("Fenêtre anti-doublons", self.temporal_window_input)

        self.similarity_threshold_input = QDoubleSpinBox()
        self.similarity_threshold_input.setRange(0.0, 1.0)
        self.similarity_threshold_input.setDecimals(2)
        self.similarity_threshold_input.setSingleStep(0.05)
        self.similarity_threshold_input.setValue(defaults.similarity_threshold)
        self.similarity_threshold_input.setSuffix(" / 1")
        self.similarity_threshold_input.setToolTip(
            "Un seuil plus bas considère davantage d'images comme des doublons."
        )
        form.addRow("Seuil de similarité visuelle", self.similarity_threshold_input)

        self.format_input = QComboBox()
        self.format_input.addItem("JPEG", "jpeg")
        self.format_input.addItem("PNG", "png")
        form.addRow("Format", self.format_input)
        layout.addLayout(form)

        selection_help = QLabel(
            "Sélectivité : augmentez le score minimal, agrandissez la fenêtre anti-doublons "
            "ou baissez le seuil de similarité pour obtenir moins de photos."
        )
        selection_help.setWordWrap(True)
        layout.addWidget(selection_help)

        action_row = QHBoxLayout()
        self.start_button = QPushButton("Lancer la sélection")
        self.start_button.clicked.connect(self._start_processing)
        self.open_output_button = QPushButton("Ouvrir les résultats")
        self.open_output_button.setEnabled(False)
        self.open_output_button.clicked.connect(self._open_output_directory)
        self.stop_button = QPushButton("Arrêter")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._request_stop)
        self.selection_count_label = QLabel("Photos retenues : 0 — exportées : 0")
        action_row.addWidget(self.start_button)
        action_row.addWidget(self.open_output_button)
        action_row.addWidget(self.stop_button)
        action_row.addWidget(self.selection_count_label)
        action_row.addStretch()
        layout.addLayout(action_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        self.status_label = QLabel("Prêt à traiter un dossier de vidéos.")
        layout.addWidget(self.status_label)

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumBlockCount(500)
        self.log_output.setPlaceholderText("La progression du traitement apparaîtra ici.")
        layout.addWidget(self.log_output)

        viewer_title = QLabel("Photos retenues")
        layout.addWidget(viewer_title)
        self.preview_label = QLabel("Aucune photo exportée pour le moment.")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(260)
        self.preview_label.setFrameShape(QFrame.Shape.StyledPanel)
        self.preview_label.setWordWrap(True)
        layout.addWidget(self.preview_label, stretch=1)

        navigation = QHBoxLayout()
        self.previous_button = QPushButton("Précédente")
        self.previous_button.clicked.connect(self._show_previous_photo)
        self.next_button = QPushButton("Suivante")
        self.next_button.clicked.connect(self._show_next_photo)
        self.photo_label = QLabel("0 photo")
        navigation.addWidget(self.previous_button)
        navigation.addWidget(self.next_button)
        navigation.addWidget(self.photo_label)
        navigation.addStretch()
        layout.addLayout(navigation)
        self._update_photo_controls()

    @staticmethod
    def _path_row(input_widget: QLineEdit, label: str, callback: Callable[[], None]) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        button = QPushButton(label)
        button.clicked.connect(callback)
        layout.addWidget(input_widget, stretch=1)
        layout.addWidget(button)
        return row

    @Slot()
    def _choose_source_directory(self) -> None:
        selected = self._choose_directory("Choisir le dossier de vidéos", self.source_input.text())
        if selected:
            self.source_input.setText(selected)

    @Slot()
    def _choose_output_directory(self) -> None:
        selected = self._choose_directory("Choisir le dossier de destination", self.output_input.text())
        if selected:
            self.output_input.setText(selected)

    def _choose_directory(self, title: str, current_path: str) -> str:
        return QFileDialog.getExistingDirectory(
            self,
            title,
            current_path or str(Path.home()),
            QFileDialog.Option.ShowDirsOnly,
        )

    @Slot()
    def _start_processing(self) -> None:
        source_text = self.source_input.text().strip()
        output_text = self.output_input.text().strip()
        if not source_text:
            self._show_validation_error("Choisissez un dossier source contenant les vidéos.")
            return
        if not output_text:
            self._show_validation_error("Choisissez le dossier dans lequel enregistrer les photos.")
            return

        source_directory = Path(source_text).expanduser()
        output_directory = Path(output_text).expanduser()
        if not source_directory.is_dir():
            self._show_validation_error("Choisissez un dossier source existant contenant les vidéos.")
            return

        self._photos = ()
        self._photo_index = 0
        self._update_photo_controls()
        self.preview_label.setText("Analyse des vidéos en cours…")
        self.preview_label.setPixmap(QPixmap())
        self.log_output.clear()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.start_button.setEnabled(False)
        self.open_output_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self._update_selection_counts(0, 0)
        self.status_label.setText("Préparation du traitement local…")

        job = DesktopProcessingJob(
            source_directory=source_directory,
            output_directory=output_directory,
            image_format=str(self.format_input.currentData()),
        )
        try:
            runner = create_batch_export_runner(
                DesktopSelectionParameters(
                    minimum_score=self.minimum_score_input.value(),
                    similarity_threshold=self.similarity_threshold_input.value(),
                    temporal_window_ms=self.temporal_window_input.value(),
                )
            )
        except (OSError, RuntimeError, ValueError) as error:
            self._show_validation_error(f"Configuration de traitement invalide : {error}")
            self.start_button.setEnabled(True)
            return

        self._thread = QThread(self)
        self._worker = ProcessingWorker(runner, job)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.completed.connect(self._on_completed)
        self._worker.failed.connect(self._on_failure)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.finished.connect(self._thread.deleteLater)
        self.stop_button.setEnabled(True)
        self._thread.start()

    @Slot(object)
    def _on_progress(self, progress: BatchProgress) -> None:
        self.progress_bar.setRange(0, max(progress.total, 1))
        completed = (
            progress.current
            if progress.state in {"completed", "failed", "stopped"}
            else progress.current - 1
        )
        self.progress_bar.setValue(max(completed, 0))
        self._update_selection_counts(progress.selected_total, progress.exported_total)
        if progress.state == "exported":
            self._add_exported_photos(progress.image_paths, show_newest=True)
        elif progress.state == "completed":
            self._add_exported_photos(progress.image_paths, show_newest=True)
            self.open_output_button.setEnabled(True)
        elif progress.state == "stopped":
            self._add_exported_photos(progress.image_paths, show_newest=False)
        state_labels = {
            "started": "Analyse",
            "selected": "Sélection",
            "exported": "Export",
            "completed": "Terminé",
            "failed": "Échec",
            "stopped": "Arrêté",
        }
        message = f"{state_labels[progress.state]} : {progress.video_path.name} ({progress.current}/{progress.total})"
        if progress.state == "selected":
            message += f" — {progress.selected_total} photo(s) retenue(s)"
        elif progress.state == "exported":
            message += f" — {progress.exported_total} photo(s) exportée(s)"
        elif progress.state == "completed":
            message += f" — {len(progress.image_paths)} photo(s) disponible(s)"
        elif progress.state == "stopped":
            message += " — arrêt demandé"
        self.status_label.setText(message)
        self.log_output.appendPlainText(message)

    @Slot(object)
    def _on_completed(self, result: BatchResult) -> None:
        self._add_exported_photos(
            tuple(image_path for video_result in result.successes for image_path in video_result.image_paths),
            show_newest=False,
        )
        self.progress_bar.setValue(self.progress_bar.maximum())
        self._update_selection_counts(result.selected_count, result.exported_count)
        self.open_output_button.setEnabled(bool(result.successes))
        if result.cancelled:
            message = (
                f"Arrêté : {result.selected_count} photo(s) retenue(s), "
                f"{result.exported_count} exportée(s)."
            )
        elif not result.videos:
            message = "Aucune vidéo compatible dans le dossier sélectionné."
        elif result.failures:
            message = (
                f"Terminé : {len(result.successes)} vidéo(s) traitée(s), "
                f"{len(result.failures)} échec(s)."
            )
        else:
            message = f"Terminé : {len(result.successes)} vidéo(s) traitée(s)."
        self.status_label.setText(message)
        self.log_output.appendPlainText(message)
        for failure in result.failures:
            self.log_output.appendPlainText(f"Échec — {failure.video_path.name} : {failure.error}")
        if not self._photos:
            self._display_current_photo()

    @Slot(str)
    def _on_failure(self, error: str) -> None:
        message = f"Le traitement n'a pas pu démarrer : {error}"
        self.status_label.setText(message)
        self.log_output.appendPlainText(message)
        QMessageBox.critical(self, "BestShotAI", message)

    @Slot()
    def _on_thread_finished(self) -> None:
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._worker = None
        self._thread = None

    @Slot()
    def _request_stop(self) -> None:
        if self._worker is None:
            return
        self._worker.request_stop()
        self.stop_button.setEnabled(False)
        self.status_label.setText("Arrêt demandé : export en cours puis lot interrompu.")
        self.log_output.appendPlainText("Arrêt demandé par l'utilisateur.")

    @Slot()
    def _open_output_directory(self) -> None:
        output_directory = Path(self.output_input.text()).expanduser()
        if output_directory.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_directory.resolve())))

    @Slot()
    def _show_previous_photo(self) -> None:
        if self._photo_index > 0:
            self._photo_index -= 1
            self._display_current_photo()

    @Slot()
    def _show_next_photo(self) -> None:
        if self._photo_index + 1 < len(self._photos):
            self._photo_index += 1
            self._display_current_photo()

    def _display_current_photo(self) -> None:
        self._update_photo_controls()
        if not self._photos:
            self.preview_label.setText("Aucune photo n'a été exportée.")
            self.preview_label.setPixmap(QPixmap())
            return
        photo_path = self._photos[self._photo_index]
        pixmap = QPixmap(str(photo_path))
        if pixmap.isNull():
            self.preview_label.setText(f"Impossible de charger l'aperçu : {photo_path.name}")
            return
        self.preview_label.setText("")
        self.preview_label.setPixmap(
            pixmap.scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _add_exported_photos(self, image_paths: tuple[Path, ...], show_newest: bool) -> None:
        """Ajoute les exports d'une vidéo et affiche immédiatement les nouveaux fichiers."""
        new_paths = tuple(path for path in image_paths if path not in self._photos)
        if not new_paths:
            return
        first_new_index = len(self._photos)
        self._photos += new_paths
        if show_newest:
            self._photo_index = first_new_index
        self._display_current_photo()

    def _update_photo_controls(self) -> None:
        total = len(self._photos)
        self.previous_button.setEnabled(self._photo_index > 0)
        self.next_button.setEnabled(self._photo_index + 1 < total)
        self.photo_label.setText(f"{self._photo_index + 1} / {total}" if total else "0 photo")

    def _update_selection_counts(self, selected_total: int, exported_total: int) -> None:
        self.selection_count_label.setText(
            f"Photos retenues : {selected_total} — exportées : {exported_total}"
        )

    def _show_validation_error(self, message: str) -> None:
        self.status_label.setText(message)
        QMessageBox.warning(self, "BestShotAI", message)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._photos:
            self._display_current_photo()
