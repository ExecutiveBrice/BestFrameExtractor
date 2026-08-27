"""Orchestration locale du training pairwise et de la persistance versionnée."""

from __future__ import annotations

from pathlib import Path

from bestshot.dataset.repository import DatasetRepository, TrainingModel
from bestshot.embedding.cache import EmbeddingCache
from bestshot.learning.ranking_trainer import (
    PersonalRankingSettings,
    RankingArtifact,
    RankingExample,
    RankingTrainer,
    RankingTrainingError,
    RankingTrainingResult,
    save_ranking_artifact,
)


def build_ranking_examples(repository: DatasetRepository) -> list[RankingExample]:
    """Enrichit les préférences non-SKIP avec les embeddings locaux référencés."""
    preferences = repository.list_usable_preferences()
    frame_ids = {
        frame_id
        for preference in preferences
        for frame_id in (preference.first_frame_id, preference.second_frame_id)
    }
    frames = repository.get_frames_by_ids(frame_ids)
    examples: list[RankingExample] = []
    for preference in preferences:
        first = frames.get(preference.first_frame_id)
        second = frames.get(preference.second_frame_id)
        if first is None or second is None:
            raise RankingTrainingError("Une préférence référence une frame absente du dataset.")
        examples.append(
            RankingExample(
                EmbeddingCache.load_reference(first.embedding_reference),
                EmbeddingCache.load_reference(second.embedding_reference),
                preference.preference,
                first.video_id,
                second.video_id,
            )
        )
    return examples


def train_personal_ranking(
    repository: DatasetRepository,
    settings: PersonalRankingSettings,
    models_directory: Path,
    embedding_model_version: str,
) -> tuple[RankingTrainingResult, RankingArtifact]:
    """Entraîne le head personnel et enregistre une version immuable du modèle."""
    result = RankingTrainer(settings).train(build_ranking_examples(repository))
    artifact = save_ranking_artifact(
        result,
        models_directory,
        embedding_model_version=embedding_model_version,
        settings=settings,
    )
    repository.upsert_training_model(
        TrainingModel(
            name="personal-ranking",
            version=artifact.version,
            metadata_json=artifact.metadata_path.read_text(encoding="utf-8"),
        )
    )
    return result, artifact


def format_ranking_training_result(result: RankingTrainingResult, artifact: RankingArtifact) -> str:
    """Affiche les compteurs et métriques essentiels après l'entraînement local."""
    return "\n".join(
        (
            f"Modèle : {artifact.version}",
            f"Comparaisons entraînement : {result.train_metrics.comparison_count}",
            f"Comparaisons validation : {result.validation_metrics.comparison_count}",
            f"Vidéos entraînement : {result.train_metrics.video_count}",
            f"Vidéos validation : {result.validation_metrics.video_count}",
            f"PairwiseAccuracy entraînement : {_format_metric(result.train_metrics.pairwise_accuracy)}",
            f"PairwiseAccuracy validation : {_format_metric(result.validation_metrics.pairwise_accuracy)}",
            f"Meilleure époque : {result.best_epoch}",
            f"Artefact : {artifact.directory}",
        )
    )


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"
