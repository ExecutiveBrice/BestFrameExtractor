"""Tests du dédoublonnage local de candidates proches et similaires."""

from collections.abc import Mapping

import numpy as np

from bestshot.domain.candidate_frame import CandidateFrame, PreviewImage
from bestshot.domain.composite_score import AestheticScore, CompositeScore, CompositionScore
from bestshot.domain.face_analysis import FaceScore
from bestshot.domain.refinement import RankedCandidate
from bestshot.domain.technical_score import TechnicalScore
from bestshot.selection.deduplicate import (
    DeduplicationSettings,
    Deduplicator,
    PerceptualHashSimilarityScorer,
)


class FakeSimilarityScorer:
    def __init__(self, similarities: Mapping[tuple[bytes, bytes], float]) -> None:
        self.similarities = similarities

    def similarity(self, first: PreviewImage, second: PreviewImage) -> float:
        if first.rgb_bytes == second.rgb_bytes:
            return 1.0
        return self.similarities.get((first.rgb_bytes, second.rgb_bytes), 0.0)


def _ranked(timestamp: float, score: float, preview_value: bytes) -> RankedCandidate:
    preview = PreviewImage(1, 1, preview_value * 3)
    technical = TechnicalScore(score, score, score, score, score, score, score)
    face = FaceScore((), None, None, None, None, None, None, None, None)
    composite = CompositeScore(
        score,
        "no_people",
        technical,
        face,
        AestheticScore(0.5, is_neutral=True),
        CompositionScore(0.5, is_neutral=True),
        (),
    )
    candidate = CandidateFrame(1, timestamp, round(timestamp * 1_000), 1, 1, preview)
    return RankedCandidate(candidate, composite)


def test_nearby_similar_candidates_keep_only_the_highest_ranked() -> None:
    first = _ranked(12.300, 0.91, b"a")
    second = _ranked(12.333, 0.90, b"a")
    third = _ranked(12.366, 0.89, b"a")
    deduplicator = Deduplicator(FakeSimilarityScorer({}), DeduplicationSettings(0.9, 1_000, 8))

    result = deduplicator.deduplicate([third, second, first])

    assert result.kept == (first,)
    assert [item.discarded for item in result.duplicates] == [second, third]
    assert all(item.retained == first for item in result.duplicates)


def test_visually_different_candidates_in_temporal_window_are_kept() -> None:
    first = _ranked(12.300, 0.91, b"a")
    different = _ranked(12.333, 0.90, b"b")
    deduplicator = Deduplicator(FakeSimilarityScorer({}), DeduplicationSettings(0.9, 1_000, 8))

    result = deduplicator.deduplicate([first, different])

    assert result.kept == (first, different)
    assert result.duplicates == ()


def test_perceptual_hash_scores_identical_previews_as_identical() -> None:
    image = np.indices((32, 32)).sum(axis=0) % 2 * 255
    rgb = np.repeat(image[:, :, np.newaxis].astype(np.uint8), 3, axis=2)
    preview = PreviewImage(32, 32, rgb.tobytes())

    similarity = PerceptualHashSimilarityScorer(8).similarity(preview, preview)

    assert similarity == 1.0
