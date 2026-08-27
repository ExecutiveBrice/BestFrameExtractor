"""Régressions Qt de l'écran unique d'analyse locale."""

from __future__ import annotations

import os
from pathlib import Path
from threading import get_ident
from typing import override

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from bestshot.dataset.labels import FrameLabel
from bestshot.dataset.repository import DatasetSettings, FrameRecord
from bestshot.desktop.application import AnalysisSummary, VideoLibraryWindow
from bestshot.services.candidate_labeling import CandidateLabelingItem
from bestshot.services.embeddings import EmbeddingReport
from bestshot.services.label_selection import LabelSelectionResult


def test_analysis_updates_are_delivered_in_the_ui_thread(tmp_path: Path) -> None:
    application = QApplication.instance() or QApplication([])
    assert application is not None
    main_thread_id = get_ident()
    worker_thread_ids: list[int] = []
    ui_thread_ids: list[int] = []

    class Runner:
        def run(self, video_path: Path) -> EmbeddingReport:
            del video_path
            worker_thread_ids.append(get_ident())
            return EmbeddingReport("cpu", "test", 0, 0, 0.0)

    class Window(VideoLibraryWindow):
        @override
        def _show_video_finished(self, path: Path, position: int, total: int) -> None:
            ui_thread_ids.append(get_ident())
            super()._show_video_finished(path, position, total)

    window = Window(DatasetSettings(tmp_path / "dataset.sqlite", tmp_path / "previews"), Runner)
    window._set_videos((tmp_path / "clip.mp4",))
    window._start_analysis()

    event_loop = QEventLoop()

    def stop_when_finished() -> None:
        if window._analysis_thread is None:
            event_loop.quit()
            return
        QTimer.singleShot(10, stop_when_finished)

    QTimer.singleShot(10, stop_when_finished)
    event_loop.exec()

    assert worker_thread_ids and worker_thread_ids[0] != main_thread_id
    assert ui_thread_ids == [main_thread_id]
    assert "candidates exportées" in window._status.text()
    window._window.close()


def test_analysis_error_is_kept_visible_after_the_worker_finishes(tmp_path: Path) -> None:
    application = QApplication.instance() or QApplication([])
    assert application is not None
    window = VideoLibraryWindow(
        DatasetSettings(tmp_path / "dataset.sqlite", tmp_path / "previews"),
        lambda: None,
    )

    window._finish_analysis(
        AnalysisSummary((), (tmp_path / "clip.mp4",), ("torchvision est introuvable",))
    )

    assert "torchvision est introuvable" in window._status.text()
    window._window.close()


def test_learning_tab_reads_and_writes_labels_outside_the_ui_thread(tmp_path: Path) -> None:
    application = QApplication.instance() or QApplication([])
    assert application is not None
    main_thread_id = get_ident()
    worker_thread_ids: list[int] = []
    labels: list[tuple[int, FrameLabel]] = []
    candidates_directory = tmp_path / "bestshot-candidates"
    candidates_directory.mkdir()

    class Service:
        def list_candidates(self, directory: Path) -> tuple[CandidateLabelingItem, ...]:
            assert directory == candidates_directory
            worker_thread_ids.append(get_ident())
            return (
                CandidateLabelingItem(
                    tmp_path / "clip.mp4",
                    FrameRecord(1, 0.0, 7, "missing-preview.jpg", 1.0, "embedding.json", id=11),
                ),
            )

        def set_label(self, frame_id: int, label: FrameLabel) -> None:
            worker_thread_ids.append(get_ident())
            labels.append((frame_id, label))

    window = VideoLibraryWindow(
        DatasetSettings(tmp_path / "dataset.sqlite", tmp_path / "previews"),
        lambda: None,
        Service,
    )
    window._start_candidate_load(candidates_directory)

    event_loop = QEventLoop()

    def wait_for_load() -> None:
        if window._candidate_load_thread is None:
            event_loop.quit()
            return
        QTimer.singleShot(10, wait_for_load)

    QTimer.singleShot(10, wait_for_load)
    event_loop.exec()
    window._label_current_candidate(FrameLabel.KEEP)

    def wait_for_label() -> None:
        if window._candidate_label_thread is None:
            event_loop.quit()
            return
        QTimer.singleShot(10, wait_for_label)

    QTimer.singleShot(10, wait_for_label)
    event_loop.exec()

    assert worker_thread_ids and all(thread_id != main_thread_id for thread_id in worker_thread_ids)
    assert labels == [(11, FrameLabel.KEEP)]
    assert window._tabs.count() == 3
    window._window.close()


def test_ai_selection_tab_trains_and_exports_in_a_worker(tmp_path: Path) -> None:
    application = QApplication.instance() or QApplication([])
    assert application is not None
    main_thread_id = get_ident()
    worker_thread_ids: list[int] = []
    reports: list[object] = []

    class Service:
        def train(self) -> object:
            worker_thread_ids.append(get_ident())
            return object()

        def select_video(self, path: Path) -> LabelSelectionResult:
            worker_thread_ids.append(get_ident())
            return LabelSelectionResult(path, (path.parent / "bestshot-selection" / "frame.jpg",))

    class Window(VideoLibraryWindow):
        @override
        def _report_ai_selection(self, summary: object) -> None:
            reports.append(summary)

    video_path = tmp_path / "clip.mp4"
    window = Window(
        DatasetSettings(tmp_path / "dataset.sqlite", tmp_path / "previews"),
        lambda: None,
        ai_selection_service_factory=Service,
    )
    window._ai_selection_videos = (video_path,)
    window._start_ai_selection()
    event_loop = QEventLoop()

    def wait_for_selection() -> None:
        if window._ai_selection_thread is None:
            event_loop.quit()
            return
        QTimer.singleShot(10, wait_for_selection)

    QTimer.singleShot(10, wait_for_selection)
    event_loop.exec()

    assert worker_thread_ids and all(thread_id != main_thread_id for thread_id in worker_thread_ids)
    assert len(reports) == 1
    window._window.close()
