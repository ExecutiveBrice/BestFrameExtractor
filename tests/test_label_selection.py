"""Tests de la sélection IA découplée du dataset de préférences."""

from pathlib import Path

from bestshot.dataset.repository import DatasetStats
from bestshot.dataset.sqlite_repository import SQLiteDatasetRepository
from bestshot.services.embeddings import (
    CandidateEmbeddingResult,
    EmbeddedCandidate,
    EmbeddingReport,
)
from bestshot.services.label_selection import LabelDrivenSelectionService
from bestshot.services.personal_label_model import LabelModelTrainingReport


class FakeEmbedder:
    def embed_candidates(self, video_path: Path) -> CandidateEmbeddingResult:
        del video_path
        return CandidateEmbeddingResult(
            (
                EmbeddedCandidate(0.0, 5, 1.0, (1.0, 0.0)),
                EmbeddedCandidate(1.0, 10, 1.0, (0.0, 1.0)),
            ),
            EmbeddingReport("cpu", "Fake DINO", 2, 0, 0.0),
        )


class FakeModel:
    def predict_keep(self, embedding: tuple[float, ...]) -> bool:
        return embedding[0] > 0.0


class FakeTrainer:
    def train_and_save(self) -> tuple[FakeModel, LabelModelTrainingReport]:
        return FakeModel(), LabelModelTrainingReport(2, 2, 2, "cpu")


class FakeExporter:
    def __init__(self) -> None:
        self.frame_indexes: tuple[int, ...] = ()

    def export(self, video_path: Path, frames: tuple[object, ...], destination_directory: Path) -> tuple[Path, ...]:
        self.frame_indexes = tuple(frame.frame_index for frame in frames)  # type: ignore[attr-defined]
        return tuple(destination_directory / f"{index}.jpg" for index in self.frame_indexes)


def test_selection_processes_an_unregistered_video_without_changing_the_dataset(tmp_path: Path) -> None:
    repository = SQLiteDatasetRepository(tmp_path / "dataset.sqlite")
    exporter = FakeExporter()
    service = LabelDrivenSelectionService(
        repository,
        FakeEmbedder(),
        exporter,
        FakeTrainer(),  # type: ignore[arg-type]
    )
    selection_video = tmp_path / "not-in-dataset.mp4"
    before = repository.stats()

    service.train()
    result = service.select_video(selection_video)

    assert exporter.frame_indexes == (5,)
    assert result.exported_paths == (tmp_path / "bestshot-selection" / "5.jpg",)
    assert repository.stats() == before == DatasetStats(0, 0, 0, 0, 0, 0)
