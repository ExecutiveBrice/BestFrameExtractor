"""Tests unitaires de l'interprétation des réponses ffprobe."""

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bestshot.video.probe import VideoProbe, VideoProbeError


class FakeFFprobeRunner:
    def __init__(self, response: Mapping[str, object]) -> None:
        self.response = response
        self.paths: list[Path] = []

    def probe(self, video_path: Path) -> Mapping[str, object]:
        self.paths.append(video_path)
        return self.response


def test_inspect_parses_simulated_ffprobe_response() -> None:
    response: Mapping[str, object] = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30000/1001",
                "nb_frames": "300",
                "bit_rate": "8000000",
                "side_data_list": [{"rotation": -90}],
            }
        ],
        "format": {
            "duration": "10.01",
            "bit_rate": "9000000",
            "tags": {"creation_time": "2024-06-01T12:30:00Z"},
        },
    }
    runner = FakeFFprobeRunner(response)
    path = Path("GX010123.MP4")

    info = VideoProbe(runner).inspect(path)

    assert runner.paths == [path]
    assert info.path == path
    assert info.codec == "h264"
    assert (info.width, info.height) == (1920, 1080)
    assert info.fps == pytest.approx(29.970029970)
    assert info.approximate_frame_count == 300
    assert info.duration_seconds == pytest.approx(10.01)
    assert info.bitrate == 8_000_000
    assert info.orientation_degrees == 270
    assert info.creation_time == datetime(2024, 6, 1, 12, 30, tzinfo=timezone.utc)


def test_inspect_estimates_frame_count_and_uses_format_bitrate() -> None:
    response: Mapping[str, object] = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "hevc",
                "width": "3840",
                "height": "2160",
                "avg_frame_rate": "24/1",
                "nb_frames": "N/A",
                "tags": {"rotate": "90"},
            }
        ],
        "format": {"duration": "2.5", "bit_rate": "12000000"},
    }

    info = VideoProbe(FakeFFprobeRunner(response)).inspect(Path("movie.mp4"))

    assert info.approximate_frame_count == 60
    assert info.bitrate == 12_000_000
    assert info.orientation_degrees == 90
    assert info.creation_time is None


def test_inspect_rejects_response_without_video_stream() -> None:
    with pytest.raises(VideoProbeError, match="Aucun flux vidéo"):
        VideoProbe(FakeFFprobeRunner({"streams": [], "format": {}})).inspect(Path("audio.mp3"))
