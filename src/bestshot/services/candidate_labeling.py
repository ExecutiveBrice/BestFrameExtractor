"""Collecte locale de labels personnels sur les candidates déjà analysées."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bestshot.dataset.labels import FrameLabel
from bestshot.dataset.repository import DatasetRepository, FrameRecord
from bestshot.services.embeddings import CANDIDATE_EXPORT_DIRECTORY_NAME


class CandidateLabelingError(ValueError):
    """Le dossier de candidates ne correspond pas aux données locales connues."""


@dataclass(frozen=True, slots=True)
class CandidateLabelingItem:
    """Une candidate affichable, avec sa source et son aperçu externe."""

    video_path: Path
    frame: FrameRecord


class CandidateLabelingService:
    """Lit et étiquette les candidates sans charger de pixels depuis SQLite."""

    def __init__(self, repository: DatasetRepository) -> None:
        self._repository = repository

    def list_candidates(self, directory: Path) -> tuple[CandidateLabelingItem, ...]:
        """Retourne les candidates reliées au dossier ``bestshot-candidates`` choisi."""
        try:
            normalized_directory = directory.resolve()
        except OSError as error:
            raise CandidateLabelingError(f"Dossier de candidates inaccessible : {directory}") from error
        if not normalized_directory.is_dir():
            raise CandidateLabelingError(f"Dossier de candidates introuvable : {normalized_directory}")
        if normalized_directory.name != CANDIDATE_EXPORT_DIRECTORY_NAME:
            raise CandidateLabelingError(
                f"Choisissez le dossier « {CANDIDATE_EXPORT_DIRECTORY_NAME} » créé par l'analyse."
            )

        items: list[CandidateLabelingItem] = []
        for summary in self._repository.list_videos():
            video = summary.video
            if video.id is None:
                continue
            if video.source_path.resolve().parent / CANDIDATE_EXPORT_DIRECTORY_NAME != normalized_directory:
                continue
            items.extend(
                CandidateLabelingItem(video.source_path, frame)
                for frame in self._repository.list_frames_for_video(video.id)
            )
        if not items:
            raise CandidateLabelingError(
                "Aucune candidate indexée pour ce dossier. Lancez d'abord l'analyse des vidéos."
            )
        return tuple(
            sorted(
                items,
                key=lambda item: (
                    item.video_path.name.casefold(),
                    item.frame.timestamp,
                    item.frame.frame_index,
                ),
            )
        )

    def set_label(self, frame_id: int, label: FrameLabel) -> None:
        """Persiste un choix ; ``SKIP`` est délégué au repository comme valeur NULL."""
        self._repository.set_frame_label(frame_id, label)
