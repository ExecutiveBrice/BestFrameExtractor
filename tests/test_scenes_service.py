"""Tests de l'affichage textuel des scènes."""

from bestshot.domain.scene import Scene
from bestshot.services.scenes import format_scenes


def test_format_scenes_displays_each_scene() -> None:
    output = format_scenes([Scene(index=1, start_time=0.0, end_time=2.5, duration=2.5)])

    assert output == "Scène 1: 0.000s → 2.500s (durée : 2.500s)"


def test_format_scenes_handles_empty_results() -> None:
    assert format_scenes([]) == "Aucune scène détectée."
