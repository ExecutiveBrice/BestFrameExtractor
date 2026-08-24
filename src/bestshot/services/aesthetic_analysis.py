"""Rapport esthétique optionnel, fonctionnel même sans modèle installé."""

from collections.abc import Iterable

from bestshot.domain.candidate_frame import CandidateFrame
from bestshot.plugins.aesthetic import ClipAestheticScorer, UnavailableAestheticScorer


def format_aesthetic_analysis(
    candidates: Iterable[CandidateFrame], scorer: ClipAestheticScorer | UnavailableAestheticScorer
) -> str:
    scores = [scorer.score(candidate.preview) for candidate in candidates]
    if not scores:
        return "Aucune candidate à analyser."
    average = sum(score.global_score for score in scores) / len(scores)
    inference_ms = sum(score.inference_ms or 0.0 for score in scores)
    return (
        f"Score esthétique moyen : {average:.3f}\n"
        f"Inférence : {inference_ms:.1f} ms pour {len(scores)} candidate(s)\n"
        f"Statut modèle : {scores[0].status}"
    )
