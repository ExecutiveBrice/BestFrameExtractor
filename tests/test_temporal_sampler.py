"""Tests du pré-échantillonnage V2 avant conversion des pixels."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Self

import numpy as np
from pytest import MonkeyPatch

from bestshot.infrastructure.temporal_sampling import PyAVTemporalSamplingBackend
from bestshot.sampling.temporal_sampler import (
    DecodedVideoFrame,
    GrayscaleImage,
    PresamplingSettings,
    TemporalSampler,
    TemporalSamplingStatistics,
)


class FakeTemporalBackend:
    def __init__(self, frames: list[DecodedVideoFrame]) -> None:
        self.frames = frames
        self.converted_indices: list[int] = []

    def decode(self, video_path: Path) -> Iterator[DecodedVideoFrame]:
        del video_path
        yield from self.frames

    def to_grayscale(self, frame: DecodedVideoFrame, max_width: int) -> GrayscaleImage:
        assert max_width == 640
        self.converted_indices.append(frame.frame_index)
        return GrayscaleImage(width=2, height=2, gray_bytes=b"\x00\x00\x00\x00")


def _decoded(timestamp: float | None, index: int) -> DecodedVideoFrame:
    return DecodedVideoFrame(
        timestamp=timestamp,
        frame_index=index,
        source_width=1920,
        source_height=1080,
        payload=object(),
    )


def test_temporal_sampler_converts_only_frames_at_the_analysis_cadence() -> None:
    backend = FakeTemporalBackend(
        [_decoded(0.0, 0), _decoded(0.1, 1), _decoded(0.5, 2), _decoded(0.6, 3), _decoded(1.0, 4)]
    )
    sampler = TemporalSampler(
        backend,
        PresamplingSettings(
            analysis_fps=2.0,
            bucket_seconds=1.0,
            keep_per_bucket=2,
            analysis_max_width=640,
        ),
    )

    sampled = sampler.sample(Path("family.mp4"))

    assert [frame.frame_index for frame in sampled.samples] == [0, 2, 4]
    assert backend.converted_indices == [0, 2, 4]
    assert sampled.statistics == TemporalSamplingStatistics(video_frame_count=5, analyzed_frame_count=3)


def test_pyav_backend_uses_gray_reformat_only_for_sampled_frames(monkeypatch: MonkeyPatch) -> None:
    converted_indices: list[int] = []

    class FakeGrayFrame:
        def __init__(self, width: int, height: int) -> None:
            self.width = width
            self.height = height

        def to_ndarray(self) -> np.ndarray[tuple[int, int], np.dtype[np.uint8]]:
            return np.zeros((self.height, self.width), dtype=np.uint8)

    class FakeFrame:
        width = 1920
        height = 1080

        def __init__(self, timestamp: float, index: int) -> None:
            self.time = timestamp
            self.index = index

        def reformat(self, *, width: int, height: int, format: str) -> FakeGrayFrame:
            assert format == "gray"
            converted_indices.append(self.index)
            return FakeGrayFrame(width, height)

    class FakeContainer:
        streams = type("Streams", (), {"video": [object()]})()

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def decode(self, stream: object) -> list[FakeFrame]:
            del stream
            return [
                FakeFrame(0.0, 0),
                FakeFrame(0.1, 1),
                FakeFrame(0.5, 2),
                FakeFrame(0.6, 3),
                FakeFrame(1.0, 4),
            ]

    fake_av = ModuleType("av")
    fake_av.open = lambda path: FakeContainer()
    monkeypatch.setitem(sys.modules, "av", fake_av)

    sampler = TemporalSampler(
        PyAVTemporalSamplingBackend(),
        PresamplingSettings(
            analysis_fps=2.0,
            bucket_seconds=1.0,
            keep_per_bucket=2,
            analysis_max_width=640,
        ),
    )

    sampled = sampler.sample(Path("family.mp4"))

    assert [frame.frame_index for frame in sampled.samples] == [0, 2, 4]
    assert converted_indices == [0, 2, 4]
