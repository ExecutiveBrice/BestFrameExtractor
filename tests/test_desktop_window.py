from __future__ import annotations

import os
from pathlib import Path
from time import monotonic
from unittest.mock import Mock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QComboBox, QDoubleSpinBox, QLabel, QSpinBox

from bestshot.desktop import estimate_worker, main_window
from bestshot.desktop.main_window import MainWindow
from bestshot.services.batch import BatchProgress, BatchResult
from bestshot.services.extraction_estimate import VideoDurationSummary


@pytest.fixture(scope="module")
def qt_application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_main_window_exposes_local_batch_controls(qt_application: QApplication) -> None:
    window = MainWindow()

    assert "Dossier contenant les vidéos" in window.source_input.placeholderText()
    assert not window.findChildren(QComboBox)
    assert not window.findChildren(QDoubleSpinBox)
    labels = {label.text() for label in window.findChildren(QLabel)}
    assert labels.isdisjoint({
        "Approche", "Score de qualité minimal", "Fenêtre anti-doublons",
        "Seuil de similarité visuelle", "Format",
    })
    assert "CLIP" in window.selection_help.text()
    assert "JPEG" in window.selection_help.text()
    assert not window.stop_button.isEnabled()
    assert window.selection_count_label.text() == "Photos retenues : 0 — exportées : 0"
    assert not window.previous_button.isEnabled()
    assert not window.next_button.isEnabled()
    assert window.pool_fps_input.value() == 3
    assert window.pool_fps_input.isEnabled()


def test_sampling_rate_uses_whole_photos_per_second(qt_application: QApplication) -> None:
    window = MainWindow()
    assert isinstance(window.pool_fps_input, QSpinBox)
    assert window.pool_fps_input.minimum() == 1
    assert window.pool_fps_input.maximum() == 10
    assert window.pool_fps_input.singleStep() == 1

    window.pool_fps_input.stepUp()
    assert window.pool_fps_input.value() == 4
    window.pool_fps_input.stepDown()
    assert window.pool_fps_input.value() == 3


def test_source_duration_estimate_updates_with_sampling_rate(
    qt_application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inspect = Mock(return_value=VideoDurationSummary(60, 1, 0))
    monkeypatch.setattr(estimate_worker, "inspect_video_durations", inspect)
    window = MainWindow()
    assert "Choisissez un dossier" in window.estimate_label.text()

    window.source_input.setText(str(tmp_path))
    assert "Lecture des durées" in window.estimate_label.text()
    deadline = monotonic() + 5
    while window._duration_summary is None and monotonic() < deadline:
        qt_application.processEvents()

    assert "180 images (60 s × 3 photos/s)" in window.estimate_label.text()
    assert "avant filtrage qualité et CLIP" in window.estimate_label.text()
    window.pool_fps_input.setValue(5)
    assert "300 images (60 s × 5 photos/s)" in window.estimate_label.text()
    inspect.assert_called_once()


def test_source_changes_discard_outdated_estimates(qt_application: QApplication) -> None:
    window = MainWindow()
    window.source_input.setText("old-source")
    old_request = window._estimate_request_id
    window._on_durations_ready(old_request, VideoDurationSummary(60, 1, 0))
    assert "180 images" in window.estimate_label.text()

    window.source_input.setText("new-source")
    window._on_durations_ready(old_request, VideoDurationSummary(10, 1, 0))
    window._on_estimate_failed(old_request, "Ancienne erreur")
    assert "Lecture des durées" in window.estimate_label.text()
    window.source_input.clear()
    assert "Choisissez un dossier" in window.estimate_label.text()
    assert not window._estimate_timer.isActive()


@pytest.mark.parametrize(
    ("summary", "expected"),
    [
        (VideoDurationSummary(0, 0, 0), "Aucune vidéo compatible"),
        (VideoDurationSummary(0, 2, 2), "Estimation indisponible"),
        (VideoDurationSummary(60, 2, 1), "Estimation partielle : 1 vidéo(s) sans durée"),
    ],
)
def test_estimate_explains_missing_video_durations(
    qt_application: QApplication, summary: VideoDurationSummary, expected: str
) -> None:
    window = MainWindow()
    window._on_durations_ready(window._estimate_request_id, summary)
    assert expected in window.estimate_label.text()


def test_estimate_handles_invalid_source_directory(
    qt_application: QApplication, tmp_path: Path
) -> None:
    window = MainWindow()
    window.source_input.setText(str(tmp_path / "absent"))
    window._estimate_timer.stop()
    window._inspect_source_durations()
    assert "dossier source existant" in window.estimate_label.text()


def test_start_processing_uses_clip_with_selected_rate_and_jpeg(
    qt_application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = Mock()
    runner.run.return_value = BatchResult(tmp_path, ())
    factory = Mock(return_value=runner)
    monkeypatch.setattr(main_window, "create_photo_pool_runner", factory)
    window = MainWindow()
    output_directory = tmp_path / "photos"
    window.source_input.setText(str(tmp_path))
    window.output_input.setText(str(output_directory))
    window.pool_fps_input.setValue(7)

    window._start_processing()
    deadline = monotonic() + 5
    while window._thread is not None and monotonic() < deadline:
        qt_application.processEvents()
    if window._thread is not None:
        window._thread.quit()
        window._thread.wait()
        pytest.fail("Le thread de traitement ne s'est pas terminé.")

    factory.assert_called_once_with(7)
    runner.run.assert_called_once()
    assert runner.run.call_args.args == (tmp_path, None, output_directory, "jpeg")
    assert window.start_button.isEnabled()


def test_main_window_displays_one_exported_photo_at_a_time(
    qt_application: QApplication, tmp_path: Path
) -> None:
    image_path = tmp_path / "selected.jpg"
    image = QImage(40, 30, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.green)
    assert image.save(str(image_path))
    window = MainWindow()
    window._photos = (image_path,)

    window._display_current_photo()

    pixmap = window.preview_label.pixmap()
    assert pixmap is not None
    assert not pixmap.isNull()
    assert window.photo_label.text() == "1 / 1"


def test_main_window_displays_exports_when_a_video_completes(
    qt_application: QApplication, tmp_path: Path
) -> None:
    image_path = tmp_path / "ready.jpg"
    image = QImage(40, 30, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.blue)
    assert image.save(str(image_path))
    window = MainWindow()

    window._on_progress(
        BatchProgress(
            1,
            2,
            Path("videos/clip.mp4"),
            "exported",
            (image_path,),
            selected_total=4,
            exported_total=1,
        )
    )

    pixmap = window.preview_label.pixmap()
    assert pixmap is not None
    assert not pixmap.isNull()
    assert window.photo_label.text() == "1 / 1"
    assert "1 photo(s) exportée(s)" in window.status_label.text()
    assert window.selection_count_label.text() == "Photos retenues : 4 — exportées : 1"
