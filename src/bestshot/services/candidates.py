"""Cas d'usage de génération et de synthèse des candidates."""

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

from bestshot.domain.candidate_frame import CandidateFrame
from bestshot.domain.scene import Scene
from bestshot.video.candidate_extractor import CandidateExtractor


def extract_candidates(
    video_path: Path, scenes: Sequence[Scene], extractor: CandidateExtractor
) -> Iterator[CandidateFrame]:
    """Produit les candidates en flux, au rythme du décodeur."""
    yield from extractor.extract(video_path, scenes)


def format_candidate_counts(scenes: Sequence[Scene], candidates: Iterable[CandidateFrame]) -> str:
    """Compte les candidates par scène sans conserver les aperçus en mémoire."""
    counts = {scene.index: 0 for scene in scenes}
    for candidate in candidates:
        counts[candidate.scene_id] = counts.get(candidate.scene_id, 0) + 1
    if not counts:
        return "Aucune scène détectée."
    return "\n".join(
        f"Scène {scene.index}: {counts[scene.index]} candidate(s)" for scene in scenes
    )
