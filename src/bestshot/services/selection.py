"""Sélection finale relative par vidéo à partir du modèle personnel local."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Protocol

from bestshot.dataset.repository import DatasetRepository, FrameRecord
from bestshot.embedding.cache import EmbeddingCache
from bestshot.embedding.provider import EmbeddingVector

SELECTION_DIRECTORY_NAME = "bestshot-selection"


class PersonalSelectionError(RuntimeError):
    """Les candidates locales ne permettent pas une sélection personnelle."""


@dataclass(frozen=True, slots=True)
class SelectionSettings:
    """Règles locales de diversité des photos finales d'une même vidéo."""

    minimum_time_separation_seconds: float = 1.0
    maximum_cosine_similarity: float = 0.985

    def __post_init__(self) -> None:
        if self.minimum_time_separation_seconds < 0:
            raise ValueError("L'écart temporel minimal doit être positif ou nul.")
        if not -1.0 <= self.maximum_cosine_similarity <= 1.0:
            raise ValueError("La similarité cosinus maximale doit être comprise entre -1 et 1.")


class RankingScorer(Protocol):
    """Port du seul head personnel utilisé pour classer les candidates."""

    def score(self, embedding: EmbeddingVector) -> float:
        """Retourne un score relatif au modèle courant."""


class SelectedFrameExporter(Protocol):
    """Port d'export local des frames finales en dehors de SQLite."""

    def export(
        self,
        video_path: Path,
        frames: tuple[FrameRecord, ...],
        destination_directory: Path,
    ) -> tuple[Path, ...]:
        """Écrit les frames demandées et retourne leurs chemins locaux."""


@dataclass(frozen=True, slots=True)
class VideoSelectionResult:
    """Export d'une vidéo, sans rendre les scores comparables entre vidéos."""

    video_path: Path
    selected_frame_indexes: tuple[int, ...]
    exported_paths: tuple[Path, ...]
    requested_count: int = 0
    duplicate_count: int = 0


class PersonalSelectionService:
    """Classe les candidates d'une même vidéo puis demande leur export explicite.

    Le modèle personnel ne fournit qu'un ordre relatif local. Le nombre demandé
    est donc appliqué indépendamment à chaque vidéo : aucun seuil de score et
    aucune comparaison de score entre vidéos ne sont utilisés.
    """

    def __init__(
        self,
        repository: DatasetRepository,
        scorer: RankingScorer,
        exporter: SelectedFrameExporter,
        settings: SelectionSettings | None = None,
    ) -> None:
        self._repository = repository
        self._scorer = scorer
        self._exporter = exporter
        self._settings = settings or SelectionSettings()

    def select_video(
        self,
        video_path: Path,
        keep_count: int,
        settings: SelectionSettings | None = None,
    ) -> VideoSelectionResult:
        """Exporte les candidates les mieux classées, avec une diversité locale réglable."""
        if keep_count <= 0:
            raise ValueError("Le nombre de photos à exporter doit être positif.")
        video = self._repository.get_video_by_source_path(video_path)
        if video is None or video.id is None:
            raise PersonalSelectionError(
                "Vidéo absente du dataset : lancez d'abord l'analyse locale dans l'onglet précédent."
            )
        frames = self._repository.list_frames_for_video(video.id)
        if not frames:
            raise PersonalSelectionError("La vidéo ne contient aucune candidate analysée.")
        effective_settings = settings or self._settings
        ranked = sorted(
            (
                (self._scorer.score(embedding), frame, embedding)
                for frame in frames
                for embedding in (EmbeddingCache.load_reference(frame.embedding_reference),)
            ),
            key=lambda item: (-item[0], item[1].timestamp, item[1].frame_index),
        )
        selected: list[tuple[FrameRecord, EmbeddingVector]] = []
        duplicate_count = 0
        for _, frame, embedding in ranked:
            if self._is_duplicate(frame, embedding, selected, effective_settings):
                duplicate_count += 1
                continue
            selected.append((frame, embedding))
            if len(selected) == keep_count:
                break

        selected_frames = tuple(frame for frame, _ in selected)
        destination = video_path.resolve().parent / SELECTION_DIRECTORY_NAME
        exported = self._exporter.export(video_path, selected_frames, destination)
        if len(exported) != len(selected_frames):
            raise PersonalSelectionError("L'export ne contient pas toutes les frames sélectionnées.")
        return VideoSelectionResult(
            video_path=video_path,
            selected_frame_indexes=tuple(frame.frame_index for frame in selected_frames),
            exported_paths=exported,
            requested_count=keep_count,
            duplicate_count=duplicate_count,
        )

    def _is_duplicate(
        self,
        candidate: FrameRecord,
        candidate_embedding: EmbeddingVector,
        selected: list[tuple[FrameRecord, EmbeddingVector]],
        settings: SelectionSettings,
    ) -> bool:
        for selected_frame, selected_embedding in selected:
            if (
                abs(candidate.timestamp - selected_frame.timestamp)
                < settings.minimum_time_separation_seconds
            ):
                return True
            if _cosine_similarity(candidate_embedding, selected_embedding) >= (
                settings.maximum_cosine_similarity
            ):
                return True
        return False


def _cosine_similarity(first: EmbeddingVector, second: EmbeddingVector) -> float:
    """Retourne la similarité cosinus de deux embeddings locaux compatibles."""
    if len(first) != len(second) or not first:
        raise PersonalSelectionError("Embeddings incompatibles pour la déduplication locale.")
    value = sum(left * right for left, right in zip(first, second, strict=True))
    if not isfinite(value):
        raise PersonalSelectionError("Embedding non fini rencontré pendant la déduplication locale.")
    return value
