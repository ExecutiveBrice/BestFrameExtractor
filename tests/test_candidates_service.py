"""Tests de synthèse des candidates pour la ligne de commande."""

from bestshot.domain.candidate_frame import CandidateFrame, PreviewImage
from bestshot.domain.scene import Scene
from bestshot.services.candidates import format_candidate_counts


def test_format_candidate_counts_includes_scenes_without_candidates() -> None:
    scenes = [Scene(1, 0.0, 1.0, 1.0), Scene(2, 1.0, 2.0, 1.0)]
    candidate = CandidateFrame(1, 0.0, 0, 1920, 1080, PreviewImage(1, 1, b""))

    assert format_candidate_counts(scenes, [candidate]) == "Scène 1: 1 candidate(s)\nScène 2: 0 candidate(s)"
