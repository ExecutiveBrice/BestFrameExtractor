"""Tests de l'agrégation technique des candidates en flux."""

from bestshot.domain.candidate_frame import CandidateFrame, PreviewImage
from bestshot.domain.scene import Scene
from bestshot.scoring.technical import TechnicalScorer, TechnicalScoringSettings
from bestshot.services.technical_analysis import format_technical_analysis


def test_format_technical_analysis_aggregates_candidates_by_scene() -> None:
    scorer = TechnicalScorer(
        TechnicalScoringSettings(
            0.001, 0.05, 0.5, 0.25, 0.95, 0.05, 0.05, 0.05, 0.01, 0.2, 0.001, 0.8,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        )
    )
    preview = PreviewImage(1, 1, bytes((128, 128, 128)))
    candidates = [CandidateFrame(1, 0.0, 0, 1, 1, preview)]

    output = format_technical_analysis([Scene(1, 0.0, 1.0, 1.0)], candidates, scorer)

    assert "Scène 1: 1 candidate(s)" in output
    assert "score technique moyen" in output
