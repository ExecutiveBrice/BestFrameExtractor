from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

pytest.importorskip("PySide6")

from bestshot.desktop.worker import DesktopProcessingJob, ProcessingWorker
from bestshot.services.batch import BatchExportRunner, BatchProgress, BatchResult


class FakeRunner:
    def __init__(self) -> None:
        self.stop_was_requested = False

    def run(
        self,
        directory: Path,
        count: int | None,
        output_directory: Path,
        image_format: str = "jpeg",
        on_progress: object = None,
        should_stop: object = None,
    ) -> BatchResult:
        assert directory == Path("videos")
        assert count is None
        assert output_directory == Path("photos")
        assert image_format == "png"
        assert callable(on_progress)
        assert callable(should_stop)
        self.stop_was_requested = should_stop()
        on_progress(BatchProgress(1, 1, Path("videos/clip.mp4"), "completed"))
        return BatchResult(directory, ())


def test_worker_emits_progress_completion_and_finished() -> None:
    runner = FakeRunner()
    worker = ProcessingWorker(
        cast(BatchExportRunner, runner),
        DesktopProcessingJob(Path("videos"), Path("photos"), "png"),
    )
    progress: list[BatchProgress] = []
    results: list[BatchResult] = []
    failures: list[str] = []
    finished: list[bool] = []
    worker.progress.connect(progress.append)
    worker.completed.connect(results.append)
    worker.failed.connect(failures.append)
    worker.finished.connect(lambda: finished.append(True))

    worker.run()

    assert [event.state for event in progress] == ["completed"]
    assert results == [BatchResult(Path("videos"), ())]
    assert failures == []
    assert finished == [True]
    assert not runner.stop_was_requested


def test_worker_forwards_a_stop_request_to_the_batch_runner() -> None:
    runner = FakeRunner()
    worker = ProcessingWorker(
        cast(BatchExportRunner, runner),
        DesktopProcessingJob(Path("videos"), Path("photos"), "png"),
    )

    worker.request_stop()
    worker.run()

    assert runner.stop_was_requested
