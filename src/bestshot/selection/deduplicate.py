"""Dédoublonnage local de candidates par perceptual hash et proximité temporelle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from bestshot.domain.candidate_frame import PreviewImage
from bestshot.domain.deduplication import DeduplicationResult, DuplicateCandidate
from bestshot.domain.refinement import RankedCandidate


@dataclass(frozen=True, slots=True)
class DeduplicationSettings:
    """Seuils de similarité visuelle et de proximité temporelle."""

    similarity_threshold: float
    temporal_window_ms: int
    hash_size: int


class SimilarityScorer(Protocol):
    """Interface extensible pour pHash aujourd'hui et embeddings demain."""

    def similarity(self, first: PreviewImage, second: PreviewImage) -> float:
        """Retourne une similarité normalisée entre 0 et 1."""


class PerceptualHashSimilarityScorer:
    """Compare des aperçus RGB en utilisant un perceptual hash DCT local."""

    def __init__(self, hash_size: int) -> None:
        if hash_size <= 0:
            raise ValueError("La taille du perceptual hash doit être positive.")
        self._hash_size = hash_size

    def similarity(self, first: PreviewImage, second: PreviewImage) -> float:
        """Calcule la similarité par distance de Hamming entre les hashes DCT."""
        first_hash = self._hash(first)
        second_hash = self._hash(second)
        hamming_distance = int(np.count_nonzero(first_hash != second_hash))
        return 1.0 - hamming_distance / first_hash.size

    def _hash(self, preview: PreviewImage) -> np.ndarray[tuple[int, int], np.dtype[np.bool_]]:
        expected_size = preview.width * preview.height * 3
        if preview.width <= 0 or preview.height <= 0 or len(preview.rgb_bytes) != expected_size:
            raise ValueError("Les données de l'aperçu RGB sont invalides.")
        rgb = np.frombuffer(preview.rgb_bytes, dtype=np.uint8).reshape(
            (preview.height, preview.width, 3)
        )
        grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        resize_size = self._hash_size * 4
        reduced = cv2.resize(grayscale, (resize_size, resize_size), interpolation=cv2.INTER_AREA)
        coefficients = cv2.dct(reduced.astype(np.float32))[: self._hash_size, : self._hash_size]
        median = float(np.median(coefficients.ravel()[1:])) if coefficients.size > 1 else 0.0
        return coefficients > median


class Deduplicator:
    """Conserve la candidate la mieux classée de chaque groupe de doublons proches."""

    def __init__(self, similarity_scorer: SimilarityScorer, settings: DeduplicationSettings) -> None:
        self._similarity_scorer = similarity_scorer
        self._settings = settings

    def deduplicate(self, candidates: list[RankedCandidate]) -> DeduplicationResult:
        """Écarte uniquement les candidates similaires dans la fenêtre temporelle définie."""
        if not 0.0 <= self._settings.similarity_threshold <= 1.0:
            raise ValueError("Le seuil de similarité doit être compris entre 0 et 1.")
        if self._settings.temporal_window_ms <= 0:
            raise ValueError("La fenêtre temporelle de dédoublonnage doit être positive.")

        temporal_window_seconds = self._settings.temporal_window_ms / 1_000.0
        kept: list[RankedCandidate] = []
        duplicates: list[DuplicateCandidate] = []
        for candidate in sorted(
            candidates,
            key=lambda item: item.composite_score.final_score,
            reverse=True,
        ):
            duplicate = self._find_duplicate(candidate, kept, temporal_window_seconds)
            if duplicate is None:
                kept.append(candidate)
            else:
                duplicates.append(duplicate)
        return DeduplicationResult(tuple(kept), tuple(duplicates))

    def _find_duplicate(
        self,
        candidate: RankedCandidate,
        kept: list[RankedCandidate],
        temporal_window_seconds: float,
    ) -> DuplicateCandidate | None:
        for retained in kept:
            time_delta_seconds = abs(candidate.candidate.timestamp - retained.candidate.timestamp)
            if time_delta_seconds > temporal_window_seconds:
                continue
            similarity = self._similarity_scorer.similarity(
                candidate.candidate.preview, retained.candidate.preview
            )
            if similarity >= self._settings.similarity_threshold:
                return DuplicateCandidate(candidate, retained, similarity, time_delta_seconds)
        return None
