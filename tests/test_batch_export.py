from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from bestshot.domain.selection import SelectionResult
from bestshot.selection.exporter import ExportResult
from bestshot.services.batch import BatchExportRunner, BatchProgress


@dataclass(frozen=True)
class FakeSelection:
    selected: tuple[object, ...]


class FakeSelector:
    def select(self, video_path: Path, count: int | None) -> SelectionResult:
        if video_path.name == "broken.mp4":
            raise RuntimeError("vidéo illisible")
        assert count == 2
        return cast(SelectionResult, FakeSelection((object(), object())))


class FakeExporter:
    def export(
        self,
        video_path: Path,
        selection: SelectionResult,
        output_directory: Path,
        image_format: str = "jpeg",
        *,
        on_image_exported: object = None,
        should_stop: object = None,
    ) -> ExportResult:
        assert selection is not None
        assert image_format == "png"
        assert callable(on_image_exported)
        assert should_stop is None or callable(should_stop)
        image_path = output_directory / f"{video_path.stem}_0001.png"
        on_image_exported(image_path)
        return ExportResult(output_directory, (image_path,), output_directory / "manifest.json")


def test_batch_export_runner_reports_progress_and_preserves_exported_paths(tmp_path: Path) -> None:
    (tmp_path / "accepted.mov").touch()
    (tmp_path / "broken.mp4").touch()
    progress_events: list[BatchProgress] = []

    result = BatchExportRunner(FakeSelector(), FakeExporter()).run(
        tmp_path,
        count=2,
        output_directory=tmp_path / "photos",
        image_format="png",
        on_progress=progress_events.append,
    )

    assert len(result.successes) == 1
    assert result.successes[0].image_paths == (
        tmp_path / "photos" / "accepted" / "accepted_0001.png",
    )
    assert result.failures[0].error == "vidéo illisible"
    assert progress_events[2].image_paths == (
        tmp_path / "photos" / "accepted" / "accepted_0001.png",
    )
    assert [(event.video_path.name, event.state) for event in progress_events] == [
        ("accepted.mov", "started"),
        ("accepted.mov", "selected"),
        ("accepted.mov", "exported"),
        ("accepted.mov", "completed"),
        ("broken.mp4", "started"),
        ("broken.mp4", "failed"),
    ]
    assert progress_events[1].selected_total == 2
    assert progress_events[2].exported_total == 1


class StopAfterSelection:
    def __init__(self) -> None:
        self.stop_requested = False

    def select(self, video_path: Path, count: int | None) -> SelectionResult:
        self.stop_requested = True
        return cast(SelectionResult, FakeSelection((object(), object(), object())))


class ExporterThatMustNotRun:
    def export(self, *args: object, **kwargs: object) -> ExportResult:
        raise AssertionError("L'export ne doit pas démarrer après une demande d'arrêt.")


def test_batch_export_runner_stops_after_a_selection_before_exporting(tmp_path: Path) -> None:
    (tmp_path / "clip.mp4").touch()
    selector = StopAfterSelection()
    progress_events: list[BatchProgress] = []

    result = BatchExportRunner(selector, cast(FakeExporter, ExporterThatMustNotRun())).run(
        tmp_path,
        count=None,
        output_directory=tmp_path / "photos",
        should_stop=lambda: selector.stop_requested,
        on_progress=progress_events.append,
    )

    assert result.cancelled
    assert result.selected_count == 3
    assert result.exported_count == 0
    assert result.videos[0].cancelled
    assert [event.state for event in progress_events] == ["started", "selected", "stopped"]
