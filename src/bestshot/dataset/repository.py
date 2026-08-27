"""Port et modèles de données du dataset local de préférences."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from bestshot.dataset.labels import FrameLabel
from bestshot.domain.preferences import PairwisePreference


@dataclass(frozen=True, slots=True)
class DatasetSettings:
    """Emplacements locaux de la base SQLite et du cache d'aperçus réduit."""

    database_path: Path
    preview_cache_dir: Path


@dataclass(frozen=True, slots=True)
class VideoRecord:
    """Identité locale et immuable d'une vidéo source du dataset."""

    source_path: Path
    video_hash: str
    source_size: int
    source_mtime_ns: int
    id: int | None = None
    created_at: str | None = None


@dataclass(frozen=True, slots=True)
class FrameRecord:
    """Candidate légère : aucun pixel 4K n'est stocké dans SQLite."""

    video_id: int
    timestamp: float
    frame_index: int
    preview_reference: str
    sharpness: float
    embedding_reference: str
    id: int | None = None
    label: FrameLabel = FrameLabel.SKIP
    created_at: str | None = None


@dataclass(frozen=True, slots=True)
class TrainingModel:
    """Métadonnées réservées aux futurs modèles entraînés sur les labels personnels."""

    name: str
    version: str
    metadata_json: str = "{}"
    id: int | None = None
    created_at: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetStats:
    """Compteurs utiles pour suivre la couverture des labels locaux."""

    video_count: int
    frame_count: int
    keep_count: int
    reject_count: int
    skip_count: int
    training_model_count: int


@dataclass(frozen=True, slots=True)
class VideoDatasetSummary:
    """Vidéo avec ses compteurs de candidates et labels."""

    video: VideoRecord
    frame_count: int
    keep_count: int
    reject_count: int
    skip_count: int


@dataclass(frozen=True, slots=True)
class PreferenceStats:
    """Couverture du dataset de comparaisons relatives."""

    total_count: int
    first_count: int
    second_count: int
    equal_count: int
    skip_count: int
    video_count: int
    distinct_frame_count: int


class DatasetRepository(Protocol):
    """Port de persistance local, remplaçable sans dépendance à SQLite côté métier."""

    def upsert_video(self, record: VideoRecord) -> VideoRecord:
        """Crée ou actualise l'identité d'une vidéo locale."""

    def upsert_frame(self, record: FrameRecord) -> FrameRecord:
        """Crée ou actualise une candidate et ses références de cache externes."""

    def set_frame_label(self, frame_id: int, label: FrameLabel) -> None:
        """Associe un label ; SKIP supprime le label persistant."""

    def reset_labels(self) -> int:
        """Supprime tous les labels utilisateur sans supprimer les candidates."""

    def stats(self) -> DatasetStats:
        """Retourne les compteurs globaux du dataset."""

    def list_videos(self) -> list[VideoDatasetSummary]:
        """Retourne les vidéos et leurs compteurs de labels."""

    def upsert_training_model(self, model: TrainingModel) -> TrainingModel:
        """Réserve les métadonnées d'un futur modèle, sans l'entraîner."""

    def get_video_by_source_path(self, source_path: Path) -> VideoRecord | None:
        """Retourne une vidéo précédemment ingérée, si elle existe."""

    def list_frames_for_video(self, video_id: int) -> list[FrameRecord]:
        """Retourne les candidates d'une vidéo, dans l'ordre temporel."""

    def get_frames_by_ids(self, frame_ids: set[int]) -> dict[int, FrameRecord]:
        """Retourne les candidates demandées pour enrichir un lot de préférences."""

    def save_preference(self, preference: PairwisePreference) -> PairwisePreference:
        """Crée ou met à jour une préférence canonique."""

    def delete_preference(self, first_frame_id: int, second_frame_id: int) -> bool:
        """Supprime une préférence, si elle existe."""

    def get_preference(self, first_frame_id: int, second_frame_id: int) -> PairwisePreference | None:
        """Lit une préférence, quelle que soit l'orientation de la demande."""

    def list_usable_preferences(self) -> list[PairwisePreference]:
        """Retourne exclusivement les choix utilisables pour l'entraînement."""

    def list_preferences_for_video(self, video_id: int) -> list[PairwisePreference]:
        """Retourne les préférences impliquant une frame de cette vidéo."""

    def preference_stats(self) -> PreferenceStats:
        """Retourne les compteurs de comparaisons personnelles."""

    def reset_preferences(self) -> int:
        """Supprime les comparaisons pairwise sans supprimer les candidates."""

    def set_active_learning_pool(self, directory: Path) -> None:
        """Mémorise localement le dernier pool photo importé."""

    def get_active_learning_pool(self) -> Path | None:
        """Retourne le dernier pool photo importé, si son état est disponible."""


def default_dataset_settings() -> DatasetSettings:
    """Fournit les chemins locaux par défaut hors de toute configuration utilisateur."""
    return DatasetSettings(
        database_path=Path(".bestshot/dataset/bestshot.db"),
        preview_cache_dir=Path(".bestshot/dataset/previews"),
    )
