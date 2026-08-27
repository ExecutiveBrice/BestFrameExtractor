"""Tests de la tête binaire locale entraînée sur KEEP/REJECT uniquement."""

from pathlib import Path

import pytest

from bestshot.dataset.labels import FrameLabel
from bestshot.dataset.repository import FrameRecord
from bestshot.dataset.sqlite_repository import SQLiteDatasetRepository, video_record_from_path
from bestshot.embedding.cache import EmbeddingCache, EmbeddingCacheKey
from bestshot.services.personal_label_model import (
    PersonalLabelModelError,
    PersonalLabelModelSettings,
    PersonalLabelModelTrainer,
)


def test_label_model_trains_only_on_keep_and_reject_embeddings(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    video_path = tmp_path / "family.mp4"
    video_path.write_bytes(b"video")
    repository = SQLiteDatasetRepository(tmp_path / "dataset.sqlite")
    video = repository.upsert_video(video_record_from_path(video_path))
    assert video.id is not None
    cache = EmbeddingCache(tmp_path / "embeddings")
    keep = _store_frame(repository, cache, video_path, video.id, 0, (1.0, 0.0), FrameLabel.KEEP)
    reject = _store_frame(repository, cache, video_path, video.id, 1, (0.0, 1.0), FrameLabel.REJECT)
    _store_frame(repository, cache, video_path, video.id, 2, (0.5, 0.5), FrameLabel.SKIP)

    trainer = PersonalLabelModelTrainer(
        repository,
        PersonalLabelModelSettings(epochs=80, learning_rate=0.05),
        tmp_path / "models",
    )
    model, report = trainer.train_and_save()

    assert (report.keep_count, report.reject_count, report.embedding_dimension) == (1, 1, 2)
    assert model.predict_keep((1.0, 0.0)) is True
    assert model.predict_keep((0.0, 1.0)) is False
    assert (tmp_path / "models" / "model.pt").is_file()
    assert keep.id is not None
    assert reject.id is not None


def test_label_model_requires_both_decision_classes(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    video_path = tmp_path / "family.mp4"
    video_path.write_bytes(b"video")
    repository = SQLiteDatasetRepository(tmp_path / "dataset.sqlite")
    video = repository.upsert_video(video_record_from_path(video_path))
    assert video.id is not None
    cache = EmbeddingCache(tmp_path / "embeddings")
    _store_frame(repository, cache, video_path, video.id, 0, (1.0, 0.0), FrameLabel.KEEP)

    with pytest.raises(PersonalLabelModelError, match="ACCEPTÉE"):
        PersonalLabelModelTrainer(repository, models_directory=tmp_path / "models").train_and_save()


def _store_frame(
    repository: SQLiteDatasetRepository,
    cache: EmbeddingCache,
    video_path: Path,
    video_id: int,
    frame_index: int,
    embedding: tuple[float, ...],
    label: FrameLabel,
) -> FrameRecord:
    key = EmbeddingCacheKey.for_frame(video_path, float(frame_index), frame_index, "dino-test")
    cache.put(key, embedding)
    return repository.upsert_frame(
        FrameRecord(
            video_id,
            float(frame_index),
            frame_index,
            f"preview-{frame_index}.jpg",
            1.0,
            str(cache.reference_for(key)),
            label=label,
        )
    )
