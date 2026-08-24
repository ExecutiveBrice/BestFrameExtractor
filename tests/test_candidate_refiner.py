"""Tests du raffinement autour d'un maximum de score artificiel."""

from collections.abc import Iterator, Sequence
from pathlib import Path

from bestshot.domain.candidate_frame import CandidateFrame, PreviewImage
from bestshot.domain.composite_score import AestheticScore, CompositeScore, CompositionScore
from bestshot.domain.face_analysis import FaceScore
from bestshot.domain.refinement import RankedCandidate
from bestshot.domain.technical_score import TechnicalScore
from bestshot.video.candidate_extractor import CandidateExtractionSettings, DecodedFrame
from bestshot.video.candidate_refiner import CandidateRefiner, RefinementSettings


class FakeDecoder:
    def __init__(self, frames: Sequence[DecodedFrame]) -> None:
        self.frames = frames

    def decode(
        self, video_path: Path, settings: CandidateExtractionSettings
    ) -> Iterator[DecodedFrame]:
        yield from self.frames


class FakeTechnicalScorer:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def score(self, preview: PreviewImage) -> TechnicalScore:
        value = preview.rgb_bytes[0] / 255.0
        self.calls.append(preview.rgb_bytes[0])
        return TechnicalScore(value, value, value, value, value, value, value)


class FakeFaceScorer:
    def score(self, preview: PreviewImage) -> FaceScore:
        return FaceScore((), None, None, None, None, None, None, None, None)


class FakeCompositeScorer:
    def score(self, technical: TechnicalScore, face: FaceScore) -> CompositeScore:
        return _composite(technical.global_score)


def _composite(score: float) -> CompositeScore:
    technical = TechnicalScore(score, score, score, score, score, score, score)
    face = FaceScore((), None, None, None, None, None, None, None, None)
    return CompositeScore(
        score,
        "no_people",
        technical,
        face,
        AestheticScore(0.5, is_neutral=True),
        CompositionScore(0.5, is_neutral=True),
        (),
    )


def _frame(timestamp: float, index: int, score: int) -> DecodedFrame:
    return DecodedFrame(
        timestamp=timestamp,
        frame_index=index,
        source_width=1920,
        source_height=1080,
        preview=PreviewImage(1, 1, bytes((score, score, score))),
    )


def _ranked(timestamp: float, ranking_score: float) -> RankedCandidate:
    candidate = CandidateFrame(1, timestamp, round(timestamp * 10), 1920, 1080, PreviewImage(1, 1, b"\0\0\0"))
    return RankedCandidate(candidate, _composite(ranking_score))


def test_refiner_selects_artificial_maximum_and_reuses_overlapping_frames() -> None:
    frames = [_frame(0.8, 8, 50), _frame(1.0, 10, 230), _frame(1.2, 12, 100), _frame(1.2, 12, 100)]
    technical_scorer = FakeTechnicalScorer()
    refiner = CandidateRefiner(
        FakeDecoder(frames),
        CandidateExtractionSettings(3.0, 960),
        technical_scorer,
        FakeFaceScorer(),
        FakeCompositeScorer(),
        RefinementSettings(True, 500, 2),
    )

    refined = refiner.refine(Path("family.mp4"), [_ranked(1.0, 0.9), _ranked(1.2, 0.8)])

    assert [result.selected_frame.timestamp for result in refined] == [1.0, 1.0]
    assert technical_scorer.calls == [50, 230, 100]


def test_refiner_limits_targets_to_top_candidates_per_scene() -> None:
    refiner = CandidateRefiner(
        FakeDecoder([_frame(1.0, 10, 230), _frame(2.0, 20, 100)]),
        CandidateExtractionSettings(3.0, 960),
        FakeTechnicalScorer(),
        FakeFaceScorer(),
        FakeCompositeScorer(),
        RefinementSettings(True, 500, 1),
    )

    refined = refiner.refine(Path("family.mp4"), [_ranked(1.0, 0.9), _ranked(2.0, 0.8)])

    assert len(refined) == 1
    assert refined[0].source_candidate.candidate.timestamp == 1.0
