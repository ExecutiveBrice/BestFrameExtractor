"""Tests du classement local de netteté V2."""

from __future__ import annotations

from bestshot.sampling.sharpness_ranker import SharpnessRanker
from bestshot.sampling.temporal_sampler import AnalysisFrame, GrayscaleImage


def _frame(timestamp: float, index: int, pixels: bytes) -> AnalysisFrame:
    return AnalysisFrame(
        timestamp=timestamp,
        frame_index=index,
        source_width=5,
        source_height=5,
        grayscale=GrayscaleImage(width=5, height=5, gray_bytes=pixels),
    )


def test_ranker_keeps_the_sharpest_frames_in_one_bucket() -> None:
    constant = bytes(25)
    checkerboard = bytes(255 if (index // 5 + index % 5) % 2 else 0 for index in range(25))
    ranker = SharpnessRanker()

    ranked = ranker.rank(
        [_frame(0.0, 0, constant), _frame(0.1, 1, checkerboard), _frame(0.2, 2, constant)],
        keep_per_bucket=2,
    )

    assert [item.frame.frame_index for item in ranked] == [1, 0]
    assert ranked[0].sharpness > ranked[1].sharpness


def test_ranker_applies_no_minimum_sharpness_threshold() -> None:
    ranker = SharpnessRanker()
    constant = bytes(25)

    ranked = ranker.rank(
        [_frame(0.0, 0, constant), _frame(0.1, 1, constant), _frame(0.2, 2, constant)],
        keep_per_bucket=2,
    )

    assert [item.frame.frame_index for item in ranked] == [0, 1]
    assert [item.sharpness for item in ranked] == [0.0, 0.0]


def test_ranker_handles_a_single_pixel_frame_without_external_cv_backend() -> None:
    frame = AnalysisFrame(
        timestamp=0.0,
        frame_index=0,
        source_width=1,
        source_height=1,
        grayscale=GrayscaleImage(width=1, height=1, gray_bytes=b"\xff"),
    )

    assert SharpnessRanker().measure(frame) == 0.0
