"""Résultat expliqué de l'élimination de candidates visuellement similaires."""

from dataclasses import dataclass

from bestshot.domain.refinement import RankedCandidate


@dataclass(frozen=True, slots=True)
class DuplicateCandidate:
    """Candidate écartée au profit d'une candidate mieux notée."""

    discarded: RankedCandidate
    retained: RankedCandidate
    similarity: float
    time_delta_seconds: float


@dataclass(frozen=True, slots=True)
class DeduplicationResult:
    """Candidates conservées et doublons justifiés, sans décision opaque."""

    kept: tuple[RankedCandidate, ...]
    duplicates: tuple[DuplicateCandidate, ...]
