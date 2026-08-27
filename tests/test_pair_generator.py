"""Tests des stratégies de propositions pairwise locales."""

from __future__ import annotations

from bestshot.dataset.repository import FrameRecord
from bestshot.learning.pair_generator import (
    FrameEmbedding,
    MixedPairSelectionStrategy,
    NearbyPairSelectionStrategy,
    PairGenerationSettings,
    SimilarityPairSelectionStrategy,
    generate_pairs,
)


def test_nearby_strategy_keeps_only_pairs_in_temporal_window() -> None:
    frames = _frames((0.0, (1.0, 0.0)), (1.0, (0.9, 0.1)), (8.0, (0.0, 1.0)))

    pairs = NearbyPairSelectionStrategy().select(frames, PairGenerationSettings(2.0, 10))

    assert [(item.first_frame_id, item.second_frame_id, item.reason) for item in pairs] == [(1, 2, "nearby")]


def test_similarity_and_mixed_strategies_are_deduplicated_and_skip_reviewed_pairs() -> None:
    frames = _frames((0.0, (1.0, 0.0)), (1.0, (0.99, 0.01)), (2.0, (0.0, 1.0)))
    settings = PairGenerationSettings(5.0, 3)

    similar = SimilarityPairSelectionStrategy().select(frames, settings)
    mixed = generate_pairs(
        frames,
        MixedPairSelectionStrategy(),
        settings,
        existing_pairs={(1, 2)},
    )

    assert (similar[0].first_frame_id, similar[0].second_frame_id) == (1, 2)
    assert (1, 2) not in {(pair.first_frame_id, pair.second_frame_id) for pair in mixed}
    assert len({(pair.first_frame_id, pair.second_frame_id) for pair in mixed}) == len(mixed)


def _frames(*values: tuple[float, tuple[float, float]]) -> list[FrameEmbedding]:
    return [
        FrameEmbedding(
            FrameRecord(
                video_id=1,
                timestamp=timestamp,
                frame_index=index,
                preview_reference=f"preview-{index}.jpg",
                sharpness=0.0,
                embedding_reference=f"embedding-{index}.json",
                id=index + 1,
            ),
            embedding,
        )
        for index, (timestamp, embedding) in enumerate(values)
    ]
