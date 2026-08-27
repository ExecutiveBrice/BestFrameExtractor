"""Tests des préférences pairwise, sans conversion implicite depuis KEEP/REJECT."""

from __future__ import annotations

from pathlib import Path

from bestshot.dataset.repository import FrameRecord
from bestshot.dataset.sqlite_repository import SQLiteDatasetRepository, video_record_from_path
from bestshot.domain.preferences import PairwisePreference, PreferenceChoice
from bestshot.services.preferences import format_preference_stats


def test_repository_canonicalizes_inverse_pair_and_keeps_skip_out_of_training(tmp_path: Path) -> None:
    repository, first, second = _repository_with_two_frames(tmp_path)

    saved = repository.save_preference(
        PairwisePreference(first.id or 0, second.id or 0, PreferenceChoice.FIRST)
    )
    inverse = repository.save_preference(
        PairwisePreference(second.id or 0, first.id or 0, PreferenceChoice.FIRST)
    )

    assert saved.first_frame_id == min(first.id or 0, second.id or 0)
    assert inverse.id == saved.id
    assert inverse.preference is PreferenceChoice.SECOND
    assert repository.preference_stats().total_count == 1

    skipped = repository.save_preference(
        PairwisePreference(first.id or 0, second.id or 0, PreferenceChoice.SKIP)
    )
    assert skipped.preference is PreferenceChoice.SKIP
    assert repository.list_usable_preferences() == []
    assert "SKIP : 1" in format_preference_stats(repository.preference_stats())
    assert repository.delete_preference(second.id or 0, first.id or 0)
    assert repository.preference_stats().total_count == 0


def test_preference_stats_counts_choices_videos_and_distinct_frames(tmp_path: Path) -> None:
    repository, first, second = _repository_with_two_frames(tmp_path)
    repository.save_preference(PairwisePreference(first.id or 0, second.id or 0, PreferenceChoice.EQUAL))

    stats = repository.preference_stats()

    assert (stats.total_count, stats.equal_count, stats.video_count, stats.distinct_frame_count) == (1, 1, 1, 2)


def _repository_with_two_frames(tmp_path: Path):  # type: ignore[no-untyped-def]
    video_path = tmp_path / "family.mp4"
    video_path.write_bytes(b"video")
    repository = SQLiteDatasetRepository(tmp_path / "bestshot.db")
    video = repository.upsert_video(video_record_from_path(video_path))
    assert video.id is not None
    records = [
        repository.upsert_frame(
            FrameRecord(
                video.id,
                timestamp=float(index),
                frame_index=index,
                preview_reference=f"previews/{index}.jpg",
                sharpness=1.0,
                embedding_reference=f"embeddings/{index}.json",
            )
        )
        for index in range(2)
    ]
    return repository, records[0], records[1]
