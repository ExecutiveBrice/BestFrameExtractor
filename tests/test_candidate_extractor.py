"""Tests unitaires de l'échantillonnage séquentiel des candidates."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Self

from pytest import MonkeyPatch

from bestshot.domain.candidate_frame import PreviewImage
from bestshot.domain.scene import Scene
from bestshot.video.candidate_extractor import (
    CandidateExtractionSettings,
    CandidateExtractor,
    DecodedFrame,
    PyAVCandidateFrameBackend,
)


class FakeCandidateBackend:
    def __init__(self, frames: list[DecodedFrame]) -> None:
        self.frames = frames
        self.calls: list[Path] = []

    def decode(
        self, video_path: Path, settings: CandidateExtractionSettings
    ) -> Iterator[DecodedFrame]:
        self.calls.append(video_path)
        yield from self.frames


def _frame(timestamp: float, index: int) -> DecodedFrame:
    return DecodedFrame(
        timestamp=timestamp,
        frame_index=index,
        source_width=1920,
        source_height=1080,
        preview=PreviewImage(width=960, height=540, rgb_bytes=b"preview"),
    )


def test_extract_samples_each_scene_and_preserves_source_metadata() -> None:
    backend = FakeCandidateBackend(
        [_frame(0.0, 0), _frame(0.34, 10), _frame(0.67, 20), _frame(1.0, 30), _frame(1.34, 40)]
    )
    extractor = CandidateExtractor(backend, CandidateExtractionSettings(fps=3.0, analysis_max_width=960))
    scenes = [Scene(1, 0.0, 1.0, 1.0), Scene(2, 1.0, 2.0, 1.0)]

    candidates = list(extractor.extract(Path("family.mp4"), scenes))

    assert [(candidate.scene_id, candidate.timestamp) for candidate in candidates] == [
        (1, 0.0),
        (1, 0.34),
        (1, 0.67),
        (2, 1.0),
        (2, 1.34),
    ]
    assert candidates[0].frame_index == 0
    assert (candidates[0].source_width, candidates[0].source_height) == (1920, 1080)
    assert candidates[0].preview.width == 960
    assert backend.calls == [Path("family.mp4")]


def test_extract_returns_an_iterator_without_preloading_candidates() -> None:
    extractor = CandidateExtractor(
        FakeCandidateBackend([_frame(0.0, 0)]),
        CandidateExtractionSettings(fps=3.0, analysis_max_width=960),
    )

    candidates = extractor.extract(Path("family.mp4"), [Scene(1, 0.0, 1.0, 1.0)])

    assert isinstance(candidates, Iterator)
    assert next(candidates).timestamp == 0.0


def test_pyav_backend_resizes_preview_without_exporting_a_file(monkeypatch: MonkeyPatch) -> None:
    class FakeImage:
        def __init__(self) -> None:
            self.width = 1920
            self.height = 1080

        @property
        def size(self) -> tuple[int, int]:
            return self.width, self.height

        def convert(self, mode: str) -> FakeImage:
            assert mode == "RGB"
            return self

        def resize(self, size: tuple[int, int], resample: object) -> FakeImage:
            assert resample == "lanczos"
            self.width, self.height = size
            return self

        def tobytes(self) -> bytes:
            return b"resized-preview"

    class FakeFrame:
        time = 1.5
        width = 3840
        height = 2160

        def to_image(self) -> FakeImage:
            return FakeImage()

    class FakeContainer:
        streams = type("Streams", (), {"video": [object()]})()

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def decode(self, stream: object) -> list[FakeFrame]:
            return [FakeFrame()]

    fake_av = ModuleType("av")
    fake_av.open = lambda path: FakeContainer()
    fake_pil = ModuleType("PIL")
    fake_pil.Image = type("Image", (), {"Resampling": type("Resampling", (), {"LANCZOS": "lanczos"})})
    monkeypatch.setitem(sys.modules, "av", fake_av)
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)

    frames = list(
        PyAVCandidateFrameBackend().decode(
            Path("family.mp4"), CandidateExtractionSettings(fps=3.0, analysis_max_width=960)
        )
    )

    assert len(frames) == 1
    assert frames[0].timestamp == 1.5
    assert frames[0].frame_index == 0
    assert (frames[0].source_width, frames[0].source_height) == (3840, 2160)
    assert (frames[0].preview.width, frames[0].preview.height) == (960, 540)
    assert frames[0].preview.rgb_bytes == b"resized-preview"
