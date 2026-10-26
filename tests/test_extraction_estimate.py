from pathlib import Path
from unittest.mock import Mock, call

import pytest

from bestshot.domain.video_info import VideoInfo
from bestshot.services.extraction_estimate import VideoDurationSummary, inspect_video_durations
from bestshot.video.probe import VideoProbe, VideoProbeError


def video_info(duration: float | None) -> VideoInfo:
    return VideoInfo(Path("clip.mp4"), "h264", 1920, 1080, 30, None, duration, None, 0, None)


def test_estimate_sums_only_supported_videos_in_the_source_directory(tmp_path: Path) -> None:
    for name in ("a.mp4", "b.MOV", "notes.txt"):
        (tmp_path / name).touch()
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "other.mp4").touch()
    probe = Mock(spec=VideoProbe)
    probe.inspect.side_effect = [video_info(60), video_info(30.5)]

    summary = inspect_video_durations(tmp_path, probe)

    assert summary == VideoDurationSummary(90.5, 2, 0)
    assert summary.estimate_images(2) == 181
    assert summary.estimate_images(3) == 272
    assert probe.inspect.call_args_list == [call(tmp_path / "a.mp4"), call(tmp_path / "b.MOV")]


@pytest.mark.parametrize("duration", [None, 0, -1, float("nan"), float("inf")])
def test_unavailable_duration_keeps_estimate_partial(tmp_path: Path, duration: float | None) -> None:
    (tmp_path / "a.mp4").touch()
    (tmp_path / "b.mp4").touch()
    probe = Mock(spec=VideoProbe)
    probe.inspect.side_effect = [video_info(duration), video_info(10)]

    assert inspect_video_durations(tmp_path, probe) == VideoDurationSummary(10, 2, 1)


def test_unreadable_video_does_not_discard_other_durations(tmp_path: Path) -> None:
    (tmp_path / "a.mp4").touch()
    (tmp_path / "b.mp4").touch()
    probe = Mock(spec=VideoProbe)
    probe.inspect.side_effect = [VideoProbeError("Vidéo illisible"), video_info(60)]

    summary = inspect_video_durations(tmp_path, probe)

    assert summary == VideoDurationSummary(60, 2, 1)
    assert summary.estimate_images(3) == 180


def test_empty_directory_has_no_images(tmp_path: Path) -> None:
    probe = Mock(spec=VideoProbe)

    summary = inspect_video_durations(tmp_path, probe)

    assert summary == VideoDurationSummary(0, 0, 0)
    assert summary.estimate_images(3) == 0
    probe.inspect.assert_not_called()


def test_short_video_rounds_estimate_up() -> None:
    assert VideoDurationSummary(0.1, 1, 0).estimate_images(3) == 1


@pytest.mark.parametrize("fps", [0, -1])
def test_estimate_rejects_invalid_rate(fps: int) -> None:
    with pytest.raises(ValueError, match="cadence"):
        VideoDurationSummary(60, 1, 0).estimate_images(fps)
