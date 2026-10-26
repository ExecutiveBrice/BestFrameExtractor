from typer.testing import CliRunner

from bestshot.cli import app
from bestshot.services.batch import BatchResult


def test_pool_cli_delegates_single_video_and_range(tmp_path, monkeypatch) -> None:
    video = tmp_path / "test.mp4"
    video.touch()
    calls = []

    class FakeRunner:
        def run(self, source, count, output, image_format, **kwargs):
            calls.append((source, count, output, image_format, kwargs))
            return BatchResult(source, ())

    def create(fps):
        assert fps == 1.0
        return FakeRunner()

    monkeypatch.setattr("bestshot.cli.create_photo_pool_runner", create)
    result = CliRunner().invoke(
        app,
        [
            "pool",
            str(video),
            "--output",
            str(tmp_path / "photos"),
            "--fps",
            "1",
            "--start",
            "2",
            "--duration",
            "5",
            "--format",
            "png",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls == [(video, None, tmp_path / "photos", "png", {"start": 2.0, "duration": 5.0})]


def test_pool_cli_reports_missing_model_without_silent_fallback(tmp_path, monkeypatch) -> None:
    video = tmp_path / "test.mp4"
    video.touch()

    def create(fps):
        raise RuntimeError("Modèle CLIP absent ; lancez `bestshot models download clip`.")

    monkeypatch.setattr("bestshot.cli.create_photo_pool_runner", create)
    result = CliRunner().invoke(app, ["pool", str(video)])
    assert result.exit_code == 1
    assert "models download clip" in result.output
    assert not (tmp_path / "photos-pool").exists()
