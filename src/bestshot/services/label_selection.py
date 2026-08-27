"""Export des candidates classées KEEP par la tête personnelle locale."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from bestshot.dataset.repository import DatasetRepository
from bestshot.infrastructure.selection_export import ExportableFrame, PyAVSelectedFrameExporter
from bestshot.services.embeddings import CandidateEmbeddingResult
from bestshot.services.personal_label_model import (
    LabelModelTrainingReport,
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
        self, video_path: Path, frames: Sequence[ExportableFrame], destination_directory: Path
    ) -> tuple[Path, ...]: ...


class CandidateEmbedder(Protocol):
    """Prépare une vidéo pour l'inférence sans l'ajouter au dataset d'apprentissage."""

    def embed_candidates(self, video_path: Path) -> CandidateEmbeddingResult: ...


class KeepPredictor(Protocol):
    def predict_keep(self, embedding: tuple[float, ...]) -> bool: ...


class LabelDrivenSelectionService:
    """Entraîne la tête globale puis sélectionne des vidéos sans les ingérer au dataset."""

    def __init__(
        self,
        repository: DatasetRepository,
        candidate_embedder: CandidateEmbedder,
        exporter: FrameExporter | None = None,
        trainer: PersonalLabelModelTrainer | None = None,
    ) -> None:
        self._repository = repository
        self._candidate_embedder = candidate_embedder
        self._exporter = exporter or PyAVSelectedFrameExporter()
        self._trainer = trainer or PersonalLabelModelTrainer(repository)

    def train(self) -> LabelModelTrainingReport:
        self._model, report = self._trainer.train_and_save()
        return report

    def select_video(self, video_path: Path) -> LabelSelectionResult:
        model: KeepPredictor | None = getattr(self, "_model", None)
        if model is None:
            raise LabelSelectionError("Le modèle de labels doit être entraîné avant la sélection.")
        prepared = self._candidate_embedder.embed_candidates(video_path)
        selected = tuple(
            candidate
            for candidate in prepared.candidates
            if model.predict_keep(candidate.embedding)
        )
        exported = self._exporter.export(
            video_path, selected, video_path.resolve().parent / SELECTION_DIRECTORY_NAME
        )
        return LabelSelectionResult(video_path, exported)
