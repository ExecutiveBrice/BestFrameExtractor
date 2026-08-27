"""Tests de la collecte locale de labels depuis un dossier de candidates."""

from pathlib import Path

import pytest

from bestshot.dataset.labels import FrameLabel
from bestshot.dataset.repository import FrameRecord
from bestshot.dataset.sqlite_repository import SQLiteDatasetRepository, video_record_from_path
from bestshot.services.candidate_labeling import CandidateLabelingError, CandidateLabelingService


def test_candidate_labeling_lists_only_frames_from_the_selected_candidate_folder(tmp_path: Path) -> None:
    directory = tmp_path / "bestshot-candidates"
    directory.mkdir()
    video_path = tmp_path / "family.mp4"
    video_path.write_bytes(b"video")
    other_directory = tmp_path / "other" / "bestshot-candidates"
    other_directory.mkdir(parents=True)
    other_video_path = other_directory.parent / "other.mp4"
    other_video_path.write_bytes(b"other")
    repository = SQLiteDatasetRepository(tmp_path / "dataset.sqlite")
    first_video = repository.upsert_video(video_record_from_path(video_path))
    second_video = repository.upsert_video(video_record_from_path(other_video_path))
    assert first_video.id is not None
    assert second_video.id is not None
    first = repository.upsert_frame(
        FrameRecord(first_video.id, 1.0, 10, "preview-a.jpg", 2.0, "embedding-a.json")
    )
    repository.upsert_frame(
        FrameRecord(second_video.id, 1.0, 10, "preview-b.jpg", 2.0, "embedding-b.json")
    )

    service = CandidateLabelingService(repository)
    items = service.list_candidates(directory)
    assert [(item.video_path, item.frame.id) for item in items] == [(video_path.resolve(), first.id)]

    service.set_label(first.id or 0, FrameLabel.KEEP)
    assert repository.list_frames_for_video(first_video.id)[0].label is FrameLabel.KEEP


def test_candidate_labeling_rejects_a_directory_that_is_not_an_export_folder(tmp_path: Path) -> None:
    directory = tmp_path / "photos"
    directory.mkdir()

    with pytest.raises(CandidateLabelingError, match="bestshot-candidates"):
        CandidateLabelingService(SQLiteDatasetRepository(tmp_path / "dataset.sqlite")).list_candidates(
            directory
        )
