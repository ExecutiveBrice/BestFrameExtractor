"""Test d'intégration explicite avec ffmpeg et ffprobe installés localement."""

import shutil
import subprocess
from pathlib import Path

import pytest

from bestshot.infrastructure.ffprobe import SubprocessFFprobeRunner
from bestshot.video.probe import VideoProbe


@pytest.mark.integration
def test_probe_reads_a_real_video_created_locally(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg et ffprobe doivent être installés pour ce test d'intégration")

    video_path = tmp_path / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x48:r=5:d=1",
            "-c:v",
            "mpeg4",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    info = VideoProbe(SubprocessFFprobeRunner()).inspect(video_path)

    assert info.path == video_path
    assert info.codec == "mpeg4"
    assert (info.width, info.height) == (64, 48)
    assert info.fps == pytest.approx(5.0)
    assert info.duration_seconds == pytest.approx(1.0)
