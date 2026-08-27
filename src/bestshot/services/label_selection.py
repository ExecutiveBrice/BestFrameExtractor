"""Export des candidates classées KEEP par la tête personnelle locale."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from bestshot.dataset.repository import DatasetRepository, FrameRecord
from bestshot.embedding.cache import EmbeddingCache
from bestshot.infrastructure.selection_export import PyAVSelectedFrameExporter
from bestshot.services.personal_label_model import (
    LabelModelTrainingReport,
    PersonalLabelModel,
    PersonalLabelModelTrainer,
)
from bestshot.services.selection import SELECTION_DIRECTORY_NAME


class LabelSelectionError(RuntimeError):
    """Une sélection IA à partir des labels locaux ne peut pas être produite."""


@dataclass(frozen=True, slots=True)
class LabelSelectionResult:
    """Photos KEEP exportées pour une vidéo source."""

    video_path: Path
    exported_paths: tuple[Path, ...]


class FrameExporter(Protocol):
    def export(
        self, video_path: Path, frames: tuple[FrameRecord, ...], destination_directory: Path
    ) -> tuple[Path, ...]: ...


class LabelDrivenSelectionService:
    """Entraîne la tête personnelle et exporte toute candidate prédite KEEP."""

    def __init__(
        self,
        repository: DatasetRepository,
        exporter: FrameExporter | None = None,
        trainer: PersonalLabelModelTrainer | None = None,
    ) -> None:
        self._repository = repository
        self._exporter = exporter or PyAVSelectedFrameExporter()
        self._trainer = trainer or PersonalLabelModelTrainer(repository)

    def train(self) -> LabelModelTrainingReport:
        self._model, report = self._trainer.train_and_save()
        return report

    def select_video(self, video_path: Path) -> LabelSelectionResult:
        model = getattr(self, "_model", None)
        if not isinstance(model, PersonalLabelModel):
            raise LabelSelectionError("Le modèle de labels doit être entraîné avant la sélection.")
        video = self._repository.get_video_by_source_path(video_path)
        if video is None or video.id is None:
            raise LabelSelectionError("Vidéo absente du dataset : lancez d'abord l'analyse locale.")
        frames = self._repository.list_frames_for_video(video.id)
        selected = tuple(
            frame
            for frame in frames
            if model.predict_keep(EmbeddingCache.load_reference(frame.embedding_reference))
        )
        exported = self._exporter.export(
            video_path, selected, video_path.resolve().parent / SELECTION_DIRECTORY_NAME
        )
        return LabelSelectionResult(video_path, exported)
