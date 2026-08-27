"""Tests du head pairwise entraînable, séparé de DINOv2 frozen."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from bestshot.domain.preferences import PreferenceChoice
from bestshot.learning.ranking_trainer import (
    PersonalRankingSettings,
    RankingExample,
    RankingTrainer,
    load_current_ranking_model,
    pairwise_loss,
    save_ranking_artifact,
    split_by_video,
)


def test_video_split_excludes_cross_boundary_pairs() -> None:
    examples = [
        _example((1.0, 0.0), (0.0, 1.0), PreferenceChoice.FIRST, 1, 1),
        _example((1.0, 0.0), (0.0, 1.0), PreferenceChoice.FIRST, 2, 2),
        _example((1.0, 0.0), (0.0, 1.0), PreferenceChoice.FIRST, 1, 2),
    ]

    split = split_by_video(examples, PersonalRankingSettings(validation_ratio=0.5, seed=1))

    assert len(split.train_video_ids) == 1
    assert len(split.validation_video_ids) == 1
    assert split.excluded_cross_split_count == 1
    assert len(split.train_examples) + len(split.validation_examples) == 2


def test_pairwise_losses_handle_first_second_equal_and_reject_skip() -> None:
    first = torch.tensor([0.0, 0.0, 1.0])
    second = torch.tensor([0.0, 0.0, 0.0])

    loss = pairwise_loss(
        first,
        second,
        (PreferenceChoice.FIRST, PreferenceChoice.SECOND, PreferenceChoice.EQUAL),
        equal_loss_weight=0.5,
    )

    assert loss.item() == pytest.approx((0.693147 + 0.693147 + 0.5) / 3, rel=1e-4)
    with pytest.raises(ValueError, match="SKIP"):
        pairwise_loss(first[:1], second[:1], (PreferenceChoice.SKIP,), equal_loss_weight=0.5)


def test_linear_ranknet_learns_synthetic_a_b_c_order_and_saves_versioned_artifact(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = PersonalRankingSettings(
        learning_rate=0.05,
        epochs=300,
        early_stopping_patience=50,
        validation_ratio=0.0,
        seed=7,
    )
    result = RankingTrainer(settings).train(
        [
            _example((1.0, 0.0), (0.0, 1.0), PreferenceChoice.FIRST, 1, 1),
            _example((0.0, 1.0), (-1.0, 0.0), PreferenceChoice.FIRST, 1, 1),
            _example((1.0, 0.0), (-1.0, 0.0), PreferenceChoice.FIRST, 1, 1),
        ]
    )

    artifact = save_ranking_artifact(
        result,
        tmp_path / "personal",
        embedding_model_version="dino-test-1",
        settings=settings,
    )

    assert result.model.score((1.0, 0.0)) > result.model.score((0.0, 1.0))
    assert result.model.score((0.0, 1.0)) > result.model.score((-1.0, 0.0))
    assert artifact.version == "model-0001"
    assert artifact.model_path.is_file()
    assert json.loads((tmp_path / "personal" / "current.json").read_text())["version"] == "model-0001"
    reloaded = load_current_ranking_model(tmp_path / "personal")
    assert reloaded.score((1.0, 0.0)) > reloaded.score((0.0, 1.0))


def _example(
    first: tuple[float, float],
    second: tuple[float, float],
    preference: PreferenceChoice,
    first_video: int,
    second_video: int,
) -> RankingExample:
    return RankingExample(first, second, preference, first_video, second_video)
