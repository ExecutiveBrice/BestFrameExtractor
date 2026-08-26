"""Tests de l'export natif et du manifeste JSON."""

import json
from pathlib import Path

from bestshot.domain.candidate_frame import CandidateFrame, PreviewImage
from bestshot.domain.composite_score import AestheticScore, CompositeScore, CompositionScore
from bestshot.domain.deduplication import DeduplicationResult
from bestshot.domain.face_analysis import FaceScore
from bestshot.domain.refinement import RankedCandidate
from bestshot.domain.selection import SelectionResult
from bestshot.domain.technical_score import TechnicalScore
from bestshot.selection.exporter import ExportSettings, FinalExporter


class FakeFrameExporter:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, float, Path, int]] = []

    def extract(self, video_path: Path, timestamp: float, output_path: Path, jpeg_quality: int) -> None:
        self.calls.append((video_path, timestamp, output_path, jpeg_quality))
        output_path.write_bytes(b"native-frame")


def _selection(timestamps: tuple[float, ...] = (12.3,)) -> SelectionResult:
    preview = PreviewImage(1, 1, b"\0\0\0")
    technical = TechnicalScore(0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9)
    face = FaceScore((), None, None, None, None, None, None, None, None)
    composite = CompositeScore(
        0.91,
        "no_people",
        technical,
        face,
        AestheticScore(0.5, is_neutral=True),
        CompositionScore(0.5, is_neutral=True),
        (),
    )
    ranked = tuple(
        RankedCandidate(
            CandidateFrame(2, timestamp, round(timestamp * 10), 1920, 1080, preview),
            composite,
        )
        for timestamp in timestamps
    )
    return SelectionResult(30, ranked, (), DeduplicationResult(ranked, ()))


def test_export_writes_native_named_jpeg_and_manifest(tmp_path: Path) -> None:
    runner = FakeFrameExporter()
    result = FinalExporter(runner, ExportSettings(2)).export(
        Path("vacances.mp4"), _selection(), tmp_path / "photos"
    )

    assert result.image_paths[0].name == "vacances_0001.jpg"
    assert runner.calls[0][1] == 12.3
    assert runner.calls[0][3] == 2
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["images"][0]["source_video"] == "vacances.mp4"
    assert manifest["images"][0]["timestamp"] == 12.3
    assert manifest["images"][0]["frame"] == 123
    assert manifest["images"][0]["score_final"] == 0.91
    assert manifest["images"][0]["scene_source"] == 2


def test_export_supports_png(tmp_path: Path) -> None:
    result = FinalExporter(FakeFrameExporter(), ExportSettings(2)).export(
        Path("vacances.mp4"), _selection(), tmp_path, image_format="png"
    )

    assert result.image_paths[0].suffix == ".png"


def test_export_stops_between_frames_and_writes_a_partial_manifest(tmp_path: Path) -> None:
    frame_exporter = FakeFrameExporter()
    stop_requested = False
    exported: list[Path] = []

    def on_image_exported(image_path: Path) -> None:
        nonlocal stop_requested
        exported.append(image_path)
        stop_requested = True

    result = FinalExporter(frame_exporter, ExportSettings(2)).export(
        Path("vacances.mp4"),
        _selection((12.3, 12.4)),
        tmp_path / "photos",
        on_image_exported=on_image_exported,
        should_stop=lambda: stop_requested,
    )

    assert result.cancelled
    assert len(frame_exporter.calls) == 1
    assert result.image_paths == tuple(exported)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["images"]) == 1
