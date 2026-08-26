from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from bestshot.desktop.main_window import MainWindow
from bestshot.services.batch import BatchProgress


@pytest.fixture(scope="module")
def qt_application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_main_window_exposes_local_batch_controls(qt_application: QApplication) -> None:
    window = MainWindow()

    assert "Dossier contenant les vidéos" in window.source_input.placeholderText()
    assert window.minimum_score_input.value() == 0.55
    assert window.temporal_window_input.value() == 1_000
    assert window.similarity_threshold_input.value() == 0.90
    assert window.format_input.currentData() == "jpeg"
    assert not window.stop_button.isEnabled()
    assert window.selection_count_label.text() == "Photos retenues : 0 — exportées : 0"
    assert not window.previous_button.isEnabled()
    assert not window.next_button.isEnabled()


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
