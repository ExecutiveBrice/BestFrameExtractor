from pathlib import Path

from typer.testing import CliRunner

from bestshot import cli
from bestshot.cli import app
from bestshot.dataset.labels import FrameLabel
from bestshot.dataset.repository import FrameRecord
from bestshot.dataset.sqlite_repository import SQLiteDatasetRepository, video_record_from_path
from bestshot.embedding.dinov2 import DINOv2ModelManager, DINOv2ModelStatus
from bestshot.services.embeddings import EmbeddingReport
from bestshot.services.presampling import PresamplingReport


def test_presample_command_displays_v2_report(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    video = tmp_path / "family.mp4"
    video.touch()
    monkeypatch.setattr(
        cli,
        "generate_presampling_report",
        lambda video_path, probe, generator: PresamplingReport(10.0, 300, 80, 20),
    )

    result = CliRunner().invoke(app, ["presample", str(video)])

    assert result.exit_code == 0
    assert "Durée : 10.000 s" in result.output
    assert "Frames vidéo : 300" in result.output
    assert "Frames analysées : 80" in result.output
    assert "Candidates générées : 20" in result.output
    assert "Candidates/minute : 120.00" in result.output


def test_embeddings_command_displays_local_cache_report(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    video = tmp_path / "family.mp4"
    video.touch()
    monkeypatch.setattr(cli, "DINOv2EmbeddingProvider", lambda settings: object())
    monkeypatch.setattr(
        cli.VideoEmbeddingRunner,
        "run",
        lambda self, video_path: EmbeddingReport("cpu", "DINOv2 ViT-S/14", 3, 7, 1.25),
    )

    result = CliRunner().invoke(app, ["embeddings", str(video)])

    assert result.exit_code == 0
    assert "Device : CPU" in result.output
    assert "Modèle : DINOv2 ViT-S/14" in result.output
    assert "Embeddings calculés : 3" in result.output
    assert "Embeddings depuis le cache : 7" in result.output
    assert "Temps de traitement : 1.250 s" in result.output


def test_models_download_embedding_uses_explicit_model_manager(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    cache_path = tmp_path / "dinov2"
    monkeypatch.setattr(
        DINOv2ModelManager,
        "download",
        lambda self, settings: DINOv2ModelStatus(True, cache_path, "installé localement"),
    )

    result = CliRunner().invoke(app, ["models", "download", "embedding"])

    assert result.exit_code == 0
    assert f"Modèle embedding : installé localement ({cache_path})" in result.output


def test_dataset_commands_report_videos_and_reset_only_labels(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    video_path = tmp_path / "family.mp4"
    video_path.write_bytes(b"video")
    repository = SQLiteDatasetRepository(tmp_path / "bestshot.db")
    video = repository.upsert_video(video_record_from_path(video_path))
    assert video.id is not None
    repository.upsert_frame(
        FrameRecord(
            video_id=video.id,
            timestamp=0.0,
            frame_index=0,
            preview_reference="previews/0.jpg",
            sharpness=1.0,
            embedding_reference="embeddings/0.json",
            label=FrameLabel.KEEP,
        )
    )
    monkeypatch.setattr(cli, "_create_dataset_repository", lambda: repository)

    stats = CliRunner().invoke(app, ["dataset", "stats"])
    videos = CliRunner().invoke(app, ["dataset", "videos"])
    reset = CliRunner().invoke(app, ["dataset", "reset-labels"])

    assert stats.exit_code == 0
    assert "KEEP : 1" in stats.output
    assert videos.exit_code == 0
    assert str(video_path) in videos.output
    assert reset.exit_code == 0
    assert "1 label(s) réinitialisé(s) en SKIP." in reset.output
    assert repository.stats().skip_count == 1
