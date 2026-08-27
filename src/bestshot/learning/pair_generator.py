"""Sélection locale de paires informatives à comparer par l'utilisateur."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from bestshot.dataset.repository import FrameRecord
from bestshot.embedding.provider import EmbeddingVector, normalize_embedding


@dataclass(frozen=True, slots=True)
class PairGenerationSettings:
    """Réglages de propositions, sans lien avec les scores de qualité V2."""

    temporal_window_seconds: float = 5.0
    max_pairs_per_group: int = 20
    seed: int = 42
    photo_pool_coverage_segment_count: int = 20
    photo_pool_maximum_cosine_similarity: float = 0.92
    photo_pool_minimum_frame_gap: int = 4

    def __post_init__(self) -> None:
        if self.temporal_window_seconds <= 0:
            raise ValueError("La fenêtre temporelle doit être positive.")
        if self.max_pairs_per_group <= 0:
            raise ValueError("Le nombre maximal de paires doit être positif.")
        if self.photo_pool_coverage_segment_count <= 0:
            raise ValueError("Le nombre de segments de couverture doit être positif.")
        if not -1.0 <= self.photo_pool_maximum_cosine_similarity <= 1.0:
            raise ValueError("La similarité maximale du pool photo doit être comprise entre -1 et 1.")
        if self.photo_pool_minimum_frame_gap < 0:
            raise ValueError("L'écart minimal entre candidates doit être positif ou nul.")


@dataclass(frozen=True, slots=True)
class FrameEmbedding:
    """Une candidate du dataset accompagnée de son vecteur local normalisé."""

    frame: FrameRecord
    embedding: EmbeddingVector

    def __post_init__(self) -> None:
        if self.frame.id is None:
            raise ValueError("La frame doit être persistée avant de générer une paire.")
        object.__setattr__(self, "embedding", normalize_embedding(self.embedding))


@dataclass(frozen=True, slots=True)
class GeneratedPair:
    """Paire proposée à l'UI ; les identifiants sont canoniques."""

    first_frame_id: int
    second_frame_id: int
    reason: str

    def __post_init__(self) -> None:
        if self.first_frame_id >= self.second_frame_id:
            raise ValueError("Les paires générées doivent être canoniques.")


class PairSelectionStrategy(Protocol):
    """Port extensible pour de futures stratégies fondées sur l'incertitude."""

    def select(
        self,
        frames: Sequence[FrameEmbedding],
        settings: PairGenerationSettings,
    ) -> list[GeneratedPair]:
        """Retourne des paires canoniques ordonnées par intérêt."""


class RandomPairSelectionStrategy:
    """Échantillonnage uniforme reproductible, utile lorsque le dataset est petit."""

    def select(
        self, frames: Sequence[FrameEmbedding], settings: PairGenerationSettings
    ) -> list[GeneratedPair]:
        candidates = _all_pairs(frames, "random")
        random.Random(settings.seed).shuffle(candidates)
        return candidates


class NearbyPairSelectionStrategy:
    """Propose des candidates proches dans le temps, donc comparables visuellement."""

    def select(
        self, frames: Sequence[FrameEmbedding], settings: PairGenerationSettings
    ) -> list[GeneratedPair]:
        sorted_frames = sorted(frames, key=lambda item: (item.frame.timestamp, item.frame.frame_index))
        distances: list[tuple[float, GeneratedPair]] = []
        for index, current in enumerate(sorted_frames):
            for other in sorted_frames[index + 1 :]:
                distance = other.frame.timestamp - current.frame.timestamp
                if distance > settings.temporal_window_seconds:
                    break
                distances.append((distance, _pair_for(current, other, "nearby")))
        distances.sort(key=lambda item: (item[0], item[1].first_frame_id, item[1].second_frame_id))
        return [pair for _, pair in distances]


class SimilarityPairSelectionStrategy:
    """Propose les images DINOv2 les plus similaires à comparer localement."""

    def select(
        self, frames: Sequence[FrameEmbedding], settings: PairGenerationSettings
    ) -> list[GeneratedPair]:
        similarities: list[tuple[float, GeneratedPair]] = []
        for first_index, first in enumerate(frames):
            for second in frames[first_index + 1 :]:
                similarity = _cosine_similarity(first.embedding, second.embedding)
                similarities.append((similarity, _pair_for(first, second, "similar")))
        similarities.sort(key=lambda item: (-item[0], item[1].first_frame_id, item[1].second_frame_id))
        return [pair for _, pair in similarities]


class MixedPairSelectionStrategy:
    """Entrelace proximité temporelle et ressemblance sémantique, sans doublon."""

    def __init__(self) -> None:
        self._nearby = NearbyPairSelectionStrategy()
        self._similarity = SimilarityPairSelectionStrategy()

    def select(
        self, frames: Sequence[FrameEmbedding], settings: PairGenerationSettings
    ) -> list[GeneratedPair]:
        # La fusion entière est ensuite plafonnée seulement après exclusion des paires revues.
        nearby = self._nearby.select(frames, settings)
        similar = self._similarity.select(frames, settings)
        merged: list[GeneratedPair] = []
        seen: set[tuple[int, int]] = set()
        for index in range(max(len(nearby), len(similar))):
            for source in (nearby, similar):
                if index >= len(source):
                    continue
                pair = source[index]
                key = (pair.first_frame_id, pair.second_frame_id)
                if key not in seen:
                    merged.append(pair)
                    seen.add(key)
        return merged


class PhotoPoolCoveragePairSelectionStrategy:
    """Couvre les portions temporelles d'un pool sans analyser ni détecter les scènes.

    Les candidates d'un film exportées dans le pool sont rangées par ordre temporel.
    Cette stratégie découpe donc cet ordre en segments réguliers, choisit dans chacun
    des images encore apparentées mais pas quasi identiques, puis entrelace les listes.
    """

    def select(
        self, frames: Sequence[FrameEmbedding], settings: PairGenerationSettings
    ) -> list[GeneratedPair]:
        ordered = sorted(frames, key=lambda item: (item.frame.timestamp, item.frame.frame_index))
        segment_count = min(
            settings.photo_pool_coverage_segment_count,
            len(ordered) // (settings.photo_pool_minimum_frame_gap + 1),
        )
        if segment_count == 0:
            return []
        ranked_segments: list[list[GeneratedPair]] = []
        for segment in _contiguous_segments(ordered, segment_count):
            similarities: list[tuple[float, GeneratedPair]] = []
            for first_index, first in enumerate(segment):
                for second in segment[first_index + 1 :]:
                    if (
                        second.frame.frame_index - first.frame.frame_index
                        < settings.photo_pool_minimum_frame_gap
                    ):
                        continue
                    similarity = _cosine_similarity(first.embedding, second.embedding)
                    if similarity >= settings.photo_pool_maximum_cosine_similarity:
                        continue
                    similarities.append(
                        (similarity, _pair_for(first, second, "coverage-similar"))
                    )
            similarities.sort(key=lambda item: (-item[0], item[1].first_frame_id, item[1].second_frame_id))
            ranked_segments.append([pair for _, pair in similarities])
        return _interleave(ranked_segments)


def generate_pairs(
    frames: Sequence[FrameEmbedding],
    strategy: PairSelectionStrategy,
    settings: PairGenerationSettings,
    existing_pairs: set[tuple[int, int]] | None = None,
    *,
    include_reviewed: bool = False,
    return_all: bool = False,
) -> list[GeneratedPair]:
    """Génère des paires sans reproposer les comparaisons déjà enregistrées."""
    normalized_existing = existing_pairs or set()
    pairs = strategy.select(frames, settings)
    available = pairs if include_reviewed else [
        pair
        for pair in pairs
        if (pair.first_frame_id, pair.second_frame_id) not in normalized_existing
    ]
    return available if return_all else available[: settings.max_pairs_per_group]


def _all_pairs(frames: Sequence[FrameEmbedding], reason: str) -> list[GeneratedPair]:
    pairs: list[GeneratedPair] = []
    for first_index, first in enumerate(frames):
        for second in frames[first_index + 1 :]:
            pairs.append(_pair_for(first, second, reason))
    return pairs


def _contiguous_segments(
    frames: Sequence[FrameEmbedding], segment_count: int
) -> list[Sequence[FrameEmbedding]]:
    """Découpe l'ordre temporel sans déduire de frontière de scène."""
    base_size, remainder = divmod(len(frames), segment_count)
    segments: list[Sequence[FrameEmbedding]] = []
    start = 0
    for index in range(segment_count):
        size = base_size + (1 if index < remainder else 0)
        segments.append(frames[start : start + size])
        start += size
    return segments


def _interleave(groups: Sequence[Sequence[GeneratedPair]]) -> list[GeneratedPair]:
    """Entrelace les groupes pour parcourir l'ensemble de la chronologie."""
    pairs: list[GeneratedPair] = []
    for index in range(max((len(group) for group in groups), default=0)):
        for group in groups:
            if index < len(group):
                pairs.append(group[index])
    return pairs


def _pair_for(first: FrameEmbedding, second: FrameEmbedding, reason: str) -> GeneratedPair:
    first_id = first.frame.id
    second_id = second.frame.id
    assert first_id is not None and second_id is not None
    return GeneratedPair(min(first_id, second_id), max(first_id, second_id), reason)


def _cosine_similarity(first: EmbeddingVector, second: EmbeddingVector) -> float:
    if len(first) != len(second):
        raise ValueError("Les embeddings comparés doivent avoir la même dimension.")
    similarity = sum(a * b for a, b in zip(first, second, strict=True))
    if not math.isfinite(similarity):
        raise ValueError("La similarité d'embeddings doit être finie.")
    return similarity
