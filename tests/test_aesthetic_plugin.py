"""Tests du fallback esthétique optionnel sans téléchargement de modèle."""

from bestshot.domain.candidate_frame import PreviewImage
from bestshot.plugins.aesthetic import UnavailableAestheticScorer


def test_unavailable_aesthetic_scorer_is_neutral() -> None:
    score = UnavailableAestheticScorer("non installé").score(PreviewImage(1, 1, b"\0\0\0"))

    assert score.global_score == 0.5
    assert score.is_neutral is True
    assert score.inference_ms is None
    assert score.status == "non installé"
