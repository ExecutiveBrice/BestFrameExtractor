"""Cas d'usage d'analyse technique en flux des candidates."""

from collections.abc import Iterable, Sequence

from bestshot.domain.candidate_frame import CandidateFrame
from bestshot.domain.scene import Scene
from bestshot.scoring.technical import TechnicalScorer


def format_technical_analysis(
    scenes: Sequence[Scene], candidates: Iterable[CandidateFrame], scorer: TechnicalScorer
) -> str:
    """Calcule une moyenne technique par scène sans accumuler les aperçus."""
    totals = {scene.index: 0.0 for scene in scenes}
    counts = {scene.index: 0 for scene in scenes}
    for candidate in candidates:
        if candidate.scene_id not in totals:
            continue
        totals[candidate.scene_id] += scorer.score(candidate.preview).global_score
        counts[candidate.scene_id] += 1
    if not totals:
        return "Aucune scène détectée."
    return "\n".join(
        f"Scène {scene.index}: {counts[scene.index]} candidate(s), "
        f"score technique moyen : {_average(totals[scene.index], counts[scene.index]):.3f}"
        for scene in scenes
    )


def _average(total: float, count: int) -> float:
    return total / count if count else 0.0
