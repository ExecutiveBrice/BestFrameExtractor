"""Tests de la représentation textuelle des informations vidéo."""

from datetime import UTC, datetime
from pathlib import Path

from bestshot.domain.video_info import VideoInfo
from bestshot.services.video_info import format_video_info


def test_format_video_info_displays_unavailable_optional_values() -> None:
    info = VideoInfo(
        path=Path("movie.mp4"),
        codec="h264",
        width=1920,
        height=1080,
        fps=30.0,
        approximate_frame_count=None,
        duration_seconds=None,
        bitrate=None,
        orientation_degrees=0,
        creation_time=None,
    )

    output = format_video_info(info)

    assert "Codec: h264" in output
    assert "Frames approximatives: indisponible" in output
    assert "Date de création: indisponible" in output


def test_format_video_info_displays_creation_time() -> None:
    info = VideoInfo(
        path=Path("movie.mp4"),
        codec="h264",
        width=1920,
        height=1080,
        fps=30.0,
        approximate_frame_count=30,
        duration_seconds=1.0,
        bitrate=100,
        orientation_degrees=0,
        creation_time=datetime(2024, 1, 1, tzinfo=UTC),
    )

    assert "2024-01-01 00:00:00+00:00" in format_video_info(info)
