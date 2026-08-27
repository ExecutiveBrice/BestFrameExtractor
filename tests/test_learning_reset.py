"""Tests de la réinitialisation explicite de l'apprentissage personnel."""

from __future__ import annotations

from pathlib import Path

from bestshot.dataset.repository import FrameRecord
from bestshot.dataset.sqlite_repository import SQLiteDatasetRepository, video_record_from_path
from bestshot.domain.preferences import PairwisePreference, PreferenceChoice
from bestshot.services.learning_reset import PersonalLearningResetService


def test_learning_reset_keeps_candidates_and_disables_the_current_model(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    repository = SQLiteDatasetRepository(tmp_path / "dataset.sqlite")
    video = repository.upsert_video(video_record_from_path(video_path))
    assert video.id is not None
    first = repository.upsert_frame(FrameRecord(video.id, 0.0, 0, "a.jpg", 0.0, "a.json"))
    second = repository.upsert_frame(FrameRecord(video.id, 1.0, 1, "b.jpg", 0.0, "b.json"))
    repository.save_preference(
        PairwisePreference(first.id or 0, second.id or 0, PreferenceChoice.SECOND)
    )
    models_directory = tmp_path / "models"
    models_directory.mkdir()
    current_model = models_directory / "current.json"
    current_model.write_text('{"version": "model-0001"}', encoding="utf-8")

    report = PersonalLearningResetService(repository, models_directory).reset()

    assert report.deleted_preference_count == 1
    assert report.current_model_disabled
    assert not current_model.exists()
    assert len(repository.list_frames_for_video(video.id)) == 2
