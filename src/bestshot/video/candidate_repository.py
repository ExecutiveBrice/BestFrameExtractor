"""Port de persistance locale des aperçus de candidates."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from bestshot.domain.candidate_frame import CandidateFrame


@dataclass(frozen=True, slots=True)
class CandidateRepositoryResult:
    """Résumé d'un dépôt de candidates créé localement."""

    output_directory: Path
    candidate_count: int
    scene_counts: Mapping[int, int]
    manifest_path: Path


class CandidatePreviewRepository(Protocol):
    """Port pour persister uniquement les aperçus réduits des candidates."""

    def store(
        self, video_path: Path, candidates: Iterable[CandidateFrame]
    ) -> CandidateRepositoryResult:
        """Écrit un flux de candidates sans en conserver la collection en mémoire."""
