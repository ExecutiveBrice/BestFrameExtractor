"""Résultat détaillé de la sélection finale des meilleures candidates."""

from dataclasses import dataclass

from bestshot.domain.deduplication import DeduplicationResult
from bestshot.domain.refinement import RankedCandidate


@dataclass(frozen=True, slots=True)
class SelectionRejection:
    candidate: RankedCandidate
    reason: str


@dataclass(frozen=True, slots=True)
class SelectionResult:
    requested_count: int | None
    selected: tuple[RankedCandidate, ...]
    rejections: tuple[SelectionRejection, ...]
    deduplication: DeduplicationResult
