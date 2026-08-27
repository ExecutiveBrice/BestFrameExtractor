"""Tests de la sélection relative par vidéo et de son export local explicite."""

from __future__ import annotations

from pathlib import Path

from bestshot.dataset.repository import FrameRecord
from bestshot.dataset.sqlite_repository import SQLiteDatasetRepository, video_record_from_path
from bestshot.embedding.cache import EmbeddingCache, EmbeddingCacheKey
from bestshot.infrastructure.selection_export import selected_frame_path
from bestshot.services.selection import (
    SELECTION_DIRECTORY_NAME,
    PersonalSelectionService,
    SelectionSettings,
)


class FakeScorer:
    def score(self, embedding: tuple[float, ...]) -> float:
        return embedding[0]


class FakeExporter:
    def __init__(self) -> None:
        self.requests: list[tuple[Path, tuple[int, ...], Path]] = []

    def export(
        self,
        video_path: Path,
        frames: tuple[FrameRecord, ...],
        destination_directory: Path,
    ) -> tuple[Path, ...]:
        self.requests.append((video_path, tuple(frame.frame_index for frame in frames), destination_directory))
        return tuple(destination_directory / f"{frame.frame_index}.jpg" for frame in frames)


def test_selection_ranks_candidates_per_video_and_exports_beside_source(tmp_path: Path) -> None:
    repository = SQLiteDatasetRepository(tmp_path / "dataset.sqlite")
    cache = EmbeddingCache(tmp_path / "embeddings")
    first_video = _video_with_frames(repository, cache, tmp_path / "first.mp4", ((0, 1.0), (1, 0.0), (2, -1.0)))
    second_video = _video_with_frames(repository, cache, tmp_path / "second.mp4", ((0, -1.0), (1, 1.0)))
    exporter = FakeExporter()
    service = PersonalSelectionService(repository, FakeScorer(), exporter)

    first = service.select_video(first_video, keep_count=2)
    second = service.select_video(second_video, keep_count=1)

    assert first.selected_frame_indexes == (0, 1)
    assert second.selected_frame_indexes == (1,)
    assert exporter.requests == [
        (first_video, (0, 1), tmp_path / SELECTION_DIRECTORY_NAME),
        (second_video, (1,), tmp_path / SELECTION_DIRECTORY_NAME),
    ]


def test_selected_frame_path_is_stable_and_isolated_per_source_video(tmp_path: Path) -> None:
    frame = FrameRecord(
        video_id=1,
        timestamp=1.234,
        frame_index=42,
        preview_reference="preview.jpg",
        sharpness=1.0,
        embedding_reference="embedding.json",
    )

    destination = selected_frame_path(tmp_path / SELECTION_DIRECTORY_NAME, tmp_path / "holiday.mp4", frame)

    assert destination.name == "holiday--mp4--frame-00000042--0000001234ms.jpg"


def test_selection_excludes_visually_quasi_identical_candidates(tmp_path: Path) -> None:
    repository = SQLiteDatasetRepository(tmp_path / "dataset.sqlite")
    cache = EmbeddingCache(tmp_path / "embeddings")
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"clip")
    video = repository.upsert_video(video_record_from_path(video_path))
    assert video.id is not None
    _store_frame(repository, cache, video.id, video_path, 0, 0.0, (1.0, 0.0))
    _store_frame(repository, cache, video.id, video_path, 1, 2.0, (0.999, 0.045))
    _store_frame(repository, cache, video.id, video_path, 2, 4.0, (0.0, 1.0))
    exporter = FakeExporter()
    service = PersonalSelectionService(
        repository,
        FakeScorer(),
        exporter,
        SelectionSettings(minimum_time_separation_seconds=0.0, maximum_cosine_similarity=0.98),
    )

    result = service.select_video(video_path, keep_count=2)

    assert result.selected_frame_indexes == (0, 2)
    assert result.duplicate_count == 1


def test_selection_excludes_candidates_that_are_too_close_in_time(tmp_path: Path) -> None:
    repository = SQLiteDatasetRepository(tmp_path / "dataset.sqlite")
    cache = EmbeddingCache(tmp_path / "embeddings")
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"clip")
    video = repository.upsert_video(video_record_from_path(video_path))
    assert video.id is not None
    _store_frame(repository, cache, video.id, video_path, 0, 0.0, (0.9, 0.1))
    _store_frame(repository, cache, video.id, video_path, 1, 0.5, (0.8, 0.6))
    _store_frame(repository, cache, video.id, video_path, 2, 2.0, (0.7, 0.7))
    service = PersonalSelectionService(
        repository,
        FakeScorer(),
        FakeExporter(),
        SelectionSettings(minimum_time_separation_seconds=1.0, maximum_cosine_similarity=1.0),
    )

    result = service.select_video(video_path, keep_count=2)

    assert result.selected_frame_indexes == (0, 2)
    assert result.duplicate_count == 1


def test_selection_returns_fewer_photos_when_only_duplicates_remain(tmp_path: Path) -> None:
    repository = SQLiteDatasetRepository(tmp_path / "dataset.sqlite")
    cache = EmbeddingCache(tmp_path / "embeddings")
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"clip")
    video = repository.upsert_video(video_record_from_path(video_path))
    assert video.id is not None
    _store_frame(repository, cache, video.id, video_path, 0, 0.0, (1.0, 0.0))
    _store_frame(repository, cache, video.id, video_path, 1, 2.0, (0.999, 0.045))
    service = PersonalSelectionService(
        repository,
        FakeScorer(),
        FakeExporter(),
        SelectionSettings(minimum_time_separation_seconds=0.0, maximum_cosine_similarity=0.98),
    )

    result = service.select_video(video_path, keep_count=2)

    assert result.selected_frame_indexes == (0,)
    assert result.requested_count == 2
    assert result.duplicate_count == 1


def test_selection_uses_the_temporal_setting_supplied_for_this_export(tmp_path: Path) -> None:
    repository = SQLiteDatasetRepository(tmp_path / "dataset.sqlite")
    cache = EmbeddingCache(tmp_path / "embeddings")
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"clip")
    video = repository.upsert_video(video_record_from_path(video_path))
    assert video.id is not None
    _store_frame(repository, cache, video.id, video_path, 0, 0.0, (1.0, 0.0))
    _store_frame(repository, cache, video.id, video_path, 1, 0.5, (0.9, 0.4))
    service = PersonalSelectionService(
        repository,
        FakeScorer(),
        FakeExporter(),
        SelectionSettings(minimum_time_separation_seconds=1.0, maximum_cosine_similarity=1.0),
    )

    result = service.select_video(
        video_path,
        keep_count=2,
        settings=SelectionSettings(minimum_time_separation_seconds=0.0, maximum_cosine_similarity=1.0),
    )

    assert result.selected_frame_indexes == (0, 1)
    assert result.duplicate_count == 0


def _video_with_frames(
    repository: SQLiteDatasetRepository,
    cache: EmbeddingCache,
    video_path: Path,
    frames: tuple[tuple[int, float], ...],
) -> Path:
    video_path.write_bytes(video_path.name.encode())
    video = repository.upsert_video(video_record_from_path(video_path))
    assert video.id is not None
    for frame_index, score in frames:
        _store_frame(repository, cache, video.id, video_path, frame_index, float(frame_index), (score, 1.0))
    return video_path


def _store_frame(
    repository: SQLiteDatasetRepository,
    cache: EmbeddingCache,
    video_id: int,
    video_path: Path,
    frame_index: int,
    timestamp: float,
    embedding: tuple[float, ...],
) -> None:
    key = EmbeddingCacheKey.for_frame(video_path, timestamp, frame_index, "selection-test")
    cache.put(key, embedding)
    repository.upsert_frame(
        FrameRecord(
            video_id,
            timestamp=timestamp,
            frame_index=frame_index,
            preview_reference=f"preview-{frame_index}.jpg",
            sharpness=1.0,
            embedding_reference=str(cache.reference_for(key)),
        )
    )
