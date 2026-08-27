"""Tests du décodage RGB limité aux seules candidates sans cache."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Self

import numpy as np
from pytest import MonkeyPatch

from bestshot.infrastructure.embedding_frames import PyAVCandidatePreviewReader
from bestshot.sampling.candidate_generator import PresampledCandidate


def _candidate(frame_index: int) -> PresampledCandidate:
    return PresampledCandidate(
        timestamp=frame_index / 10.0,
        frame_index=frame_index,
        source_width=1920,
        source_height=1080,
        bucket_index=0,
    )


def test_reader_converts_only_requested_candidate_frames(monkeypatch: MonkeyPatch) -> None:
    converted_indices: list[int] = []

    class FakeRGBFrame:
        def __init__(self, width: int, height: int) -> None:
            self.width = width
            self.height = height

        def to_ndarray(self) -> np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    class FakeFrame:
        width = 1920
        height = 1080

        def __init__(self, index: int) -> None:
            self.index = index

        def reformat(self, *, width: int, height: int, format: str) -> FakeRGBFrame:
            assert format == "rgb24"
            converted_indices.append(self.index)
            return FakeRGBFrame(width, height)

    class FakeContainer:
        streams = type("Streams", (), {"video": [object()]})()

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def decode(self, stream: object) -> list[FakeFrame]:
            del stream
            return [FakeFrame(0), FakeFrame(1), FakeFrame(2)]

    fake_av = ModuleType("av")
    fake_av.open = lambda path: FakeContainer()
    monkeypatch.setitem(sys.modules, "av", fake_av)

    previews = list(PyAVCandidatePreviewReader().read(Path("family.mp4"), [_candidate(1)], 640))

    assert converted_indices == [1]
    assert previews[0].candidate.frame_index == 1
    assert (previews[0].preview.width, previews[0].preview.height) == (640, 360)
