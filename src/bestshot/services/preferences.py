"""Cas d'usage des préférences pairwise, indépendants de l'interface PySide."""

from __future__ import annotations

from pathlib import Path

from bestshot.dataset.repository import DatasetRepository, PreferenceStats
from bestshot.domain.preferences import PairwisePreference, PreferenceChoice
from bestshot.embedding.cache import EmbeddingCache
from bestshot.learning.pair_generator import (
    FrameEmbedding,
    GeneratedPair,
    MixedPairSelectionStrategy,
    PairGenerationSettings,
    PairSelectionStrategy,
    generate_pairs,
)


class PreferenceServiceError(RuntimeError):
    """Le dataset ne contient pas les candidates nécessaires à une comparaison."""


def generate_video_preferences(
    repository: DatasetRepository,
    video_path: Path,
    settings: PairGenerationSettings,
    strategy: PairSelectionStrategy | None = None,
    *,
    include_reviewed: bool = False,
) -> list[GeneratedPair]:
    """Propose des paires pour une vidéo déjà analysée par ``bestshot embeddings``."""
    video = repository.get_video_by_source_path(video_path)
    if video is None or video.id is None:
        raise PreferenceServiceError(
            "Vidéo absente du dataset : exécutez d'abord `bestshot embeddings VIDEO`."
        )
    records = repository.list_frames_for_video(video.id)
    if len(records) < 2:
        raise PreferenceServiceError("Au moins deux candidates avec embedding sont nécessaires.")
    try:
        frames = [
            FrameEmbedding(frame, EmbeddingCache.load_reference(frame.embedding_reference))
            for frame in records
        ]
    except RuntimeError as error:
        raise PreferenceServiceError(str(error)) from error
    existing_pairs = {
        (preference.first_frame_id, preference.second_frame_id)
        for preference in repository.list_preferences_for_video(video.id)
    }
    return generate_pairs(
        frames,
        strategy or MixedPairSelectionStrategy(),
        settings,
        existing_pairs,
        include_reviewed=include_reviewed,
    )


def record_preference(
    repository: DatasetRepository,
    first_frame_id: int,
    second_frame_id: int,
    choice: PreferenceChoice,
) -> PairwisePreference:
    """Enregistre immédiatement une réponse de l'UI, y compris SKIP explicite."""
    # La canonisation est effectuée dans le repository afin que tous les adaptateurs
    # respectent la même contrainte d'unicité.
    return repository.save_preference(PairwisePreference(first_frame_id, second_frame_id, choice))


def format_preference_stats(stats: PreferenceStats) -> str:
    """Formate les compteurs demandés par la commande de diagnostic."""
    return "\n".join(
        (
            f"Comparaisons : {stats.total_count}",
            f"FIRST : {stats.first_count}",
            f"SECOND : {stats.second_count}",
            f"EQUAL : {stats.equal_count}",
            f"SKIP : {stats.skip_count}",
            f"Vidéos : {stats.video_count}",
            f"Frames distinctes : {stats.distinct_frame_count}",
        )
    )
