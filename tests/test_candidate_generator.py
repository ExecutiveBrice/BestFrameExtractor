"""Tests de génération des candidates par fenêtres temporelles V2."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from bestshot.sampling.candidate_generator import CandidateGenerator
from bestshot.sampling.sharpness_ranker import RankedAnalysisFrame
from bestshot.sampling.temporal_sampler import (
    AnalysisFrame,
    GrayscaleImage,
    PresamplingSettings,
    TemporalSampleStream,
    TemporalSamplingStatistics,
)


class FakeTemporalSampler:
    def __init__(self, frames: list[AnalysisFrame]) -> None:
        self.frames = frames
        self.paths: list[Path] = []

    def sample(self, video_path: Path) -> TemporalSampleStream:
        self.paths.append(video_path)
        return TemporalSampleStream(
            samples=self._frames(),
            statistics=TemporalSamplingStatistics(
                video_frame_count=48,
                analyzed_frame_count=len(self.frames),
            ),
        )

    def _frames(self) -> Iterator[AnalysisFrame]:
        yield from self.frames


class FixedSharpnessRanker:
    def __init__(self, scores: dict[int, float]) -> None:
        self.scores = scores

    def rank(
        self,
        frames: list[AnalysisFrame],
        keep_per_bucket: int,
    ) -> list[RankedAnalysisFrame]:
        return sorted(
            (RankedAnalysisFrame(frame, self.scores[frame.frame_index]) for frame in frames),
            key=lambda item: -item.sharpness,
        )[:keep_per_bucket]


def _frame(timestamp: float, index: int) -> AnalysisFrame:
    return AnalysisFrame(
        timestamp=timestamp,
        frame_index=index,
        source_width=1920,
        source_height=1080,
        grayscale=GrayscaleImage(width=1, height=1, gray_bytes=b"\x00"),
    )


def test_generator_keeps_two_best_frames_per_window_without_global_threshold() -> None:
    frames = [_frame(0.0, 0), _frame(0.1, 1), _frame(0.2, 2), _frame(1.1, 3)]
    sampler = FakeTemporalSampler(frames)
    generator = CandidateGenerator(
        sampler,  # type: ignore[arg-type]
        FixedSharpnessRanker({0: 10.0, 1: 40.0, 2: 20.0, 3: 0.0}),  # type: ignore[arg-type]
        PresamplingSettings(
            analysis_fps=8.0,
            bucket_seconds=1.0,
            keep_per_bucket=2,
            analysis_max_width=640,
        ),
    )

    result = generator.generate(Path("family.mp4"))

    assert sampler.paths == [Path("family.mp4")]
    assert [(candidate.bucket_index, candidate.frame_index) for candidate in result.candidates] == [
        (0, 1),
        (0, 2),
        (1, 3),
    ]
    assert [candidate.sharpness for candidate in result.candidates] == [40.0, 20.0, 0.0]
    assert result.video_frame_count == 48
    assert result.analyzed_frame_count == 4
    assert result.candidate_count == 3
