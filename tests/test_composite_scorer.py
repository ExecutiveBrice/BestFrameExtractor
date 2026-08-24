"""Tests du score composite expliqué pour les profils avec et sans personnes."""

import pytest

from bestshot.domain.composite_score import AestheticScore, CompositionScore
from bestshot.domain.face_analysis import FaceAnalysis, FaceScore
from bestshot.domain.technical_score import TechnicalScore
from bestshot.scoring.composite import CompositeScorer, CompositeScoringSettings, CompositeWeights


def _settings() -> CompositeScoringSettings:
    return CompositeScoringSettings(
        people=CompositeWeights(technical=0.30, face=0.40, aesthetic=0.20, composition=0.10),
        no_people=CompositeWeights(technical=0.40, face=0.0, aesthetic=0.35, composition=0.25),
        neutral_score=0.5,
    )


def _technical(score: float) -> TechnicalScore:
    return TechnicalScore(score, score, score, score, score, score, score)


def _face(score: float | None) -> FaceScore:
    analyses = () if score is None else (FaceAnalysis(None, 0.2, 0.0, 1.0, 1.0, 200.0, False),)
    return FaceScore(analyses, score, score, score, score, score, score, score, score)


def test_no_people_profile_uses_neutral_future_scores() -> None:
    result = CompositeScorer(_settings()).score(_technical(0.8), _face(None))

    assert result.profile == "no_people"
    assert result.final_score == pytest.approx(0.62)
    assert result.aesthetic.is_neutral is True
    assert result.composition.is_neutral is True
    assert {reason.criterion for reason in result.reasons} == {
        "technical",
        "aesthetic",
        "composition",
    }


def test_people_profile_adds_face_weight_and_explains_the_ranking() -> None:
    result = CompositeScorer(_settings()).score(_technical(0.8), _face(0.9))

    assert result.profile == "people"
    assert result.final_score == pytest.approx(0.75)
    assert result.reasons[0].criterion == "face"
    assert result.reasons[0].detail == "score calculé"


def test_future_scores_replace_the_neutral_placeholders() -> None:
    result = CompositeScorer(_settings()).score(
        _technical(0.8),
        _face(None),
        AestheticScore(0.9),
        CompositionScore(0.1),
    )

    assert result.final_score == pytest.approx(0.66)
    assert result.aesthetic.is_neutral is False
    assert result.composition.is_neutral is False
