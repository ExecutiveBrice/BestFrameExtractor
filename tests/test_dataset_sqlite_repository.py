"""Tests du schéma SQLite local et des invariants du dataset."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from bestshot.dataset.labels import FrameLabel
from bestshot.dataset.repository import FrameRecord, TrainingModel
from bestshot.dataset.sqlite_repository import SQLiteDatasetRepository, video_record_from_path
from bestshot.domain.preferences import PairwisePreference, PreferenceChoice


def test_sqlite_repository_migrates_and_keeps_skip_as_null_without_image_blobs(tmp_path: Path) -> None:
    video_path = tmp_path / "family.mp4"
    video_path.write_bytes(b"local-video-content")
    database_path = tmp_path / "dataset" / "bestshot.db"
    repository = SQLiteDatasetRepository(database_path)
    video = repository.upsert_video(video_record_from_path(video_path))
    assert video.id is not None

    frame = repository.upsert_frame(
        FrameRecord(
            video_id=video.id,
            timestamp=1.25,
            frame_index=42,
            preview_reference="previews/family_000042.jpg",
            sharpness=23.5,
            embedding_reference="embeddings/abc123.json",
        )
    )
    assert frame.id is not None
    assert frame.label is FrameLabel.SKIP

    with sqlite3.connect(database_path) as connection:
        label = connection.execute("SELECT label FROM frames WHERE id = ?", (frame.id,)).fetchone()[0]
        column_types = {row[1]: row[2] for row in connection.execute("PRAGMA table_info(frames)")}
        migrations = connection.execute("SELECT version FROM schema_migrations").fetchall()

    assert label is None
    assert all("BLOB" not in column_type.upper() for column_type in column_types.values())
    assert migrations == [(1,), (2,), (3,)]

    repository.set_frame_label(frame.id, FrameLabel.REJECT)
    refreshed = repository.upsert_frame(
        FrameRecord(
            video_id=video.id,
            timestamp=1.25,
            frame_index=42,
            preview_reference="previews/family_000042.jpg",
            sharpness=24.0,
            embedding_reference="embeddings/abc123.json",
            label=FrameLabel.SKIP,
        )
    )
    assert refreshed.label is FrameLabel.REJECT

    assert repository.stats().reject_count == 1
    assert repository.reset_labels() == 1
    assert repository.stats().skip_count == 1


def test_sqlite_repository_persists_learning_pool_and_resets_only_preferences(tmp_path: Path) -> None:
    video_path = tmp_path / "family.mp4"
    pool_path = tmp_path / "bestshot-candidates"
    video_path.write_bytes(b"local-video-content")
    pool_path.mkdir()
    repository = SQLiteDatasetRepository(tmp_path / "bestshot.db")
    video = repository.upsert_video(video_record_from_path(video_path))
    assert video.id is not None
    first = repository.upsert_frame(
        FrameRecord(video.id, 0.0, 0, "first.jpg", 0.0, "first.json")
    )
    second = repository.upsert_frame(
        FrameRecord(video.id, 1.0, 1, "second.jpg", 0.0, "second.json")
    )
    repository.save_preference(
        PairwisePreference(first.id or 0, second.id or 0, PreferenceChoice.FIRST)
    )
    repository.set_active_learning_pool(pool_path)

    assert repository.get_active_learning_pool() == pool_path.resolve()
    assert repository.reset_preferences() == 1
    assert repository.preference_stats().total_count == 0
    assert len(repository.list_frames_for_video(video.id)) == 2


def test_sqlite_repository_lists_videos_and_reserves_training_model_metadata(tmp_path: Path) -> None:
    video_path = tmp_path / "family.mp4"
    video_path.write_bytes(b"local-video-content")
    repository = SQLiteDatasetRepository(tmp_path / "bestshot.db")
    video = repository.upsert_video(video_record_from_path(video_path))
    assert video.id is not None
    frame = repository.upsert_frame(
        FrameRecord(
            video_id=video.id,
            timestamp=0.0,
            frame_index=0,
            preview_reference="previews/family_000000.jpg",
            sharpness=0.0,
            embedding_reference="embeddings/first.json",
            label=FrameLabel.KEEP,
        )
    )
    repository.upsert_training_model(TrainingModel(name="personal-preferences", version="reserved"))

    summaries = repository.list_videos()
    stats = repository.stats()

    assert frame.label is FrameLabel.KEEP
    assert [(summary.video.video_hash, summary.frame_count, summary.keep_count) for summary in summaries] == [
        (video.video_hash, 1, 1)
    ]
    assert stats.training_model_count == 1
