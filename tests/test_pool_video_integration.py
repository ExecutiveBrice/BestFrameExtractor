"""Décodage réel court, généré localement ; aucune vidéo familiale requise."""

import shutil
import subprocess

import pytest

from bestshot.infrastructure.photo_pool import PyAVPoolFrameSource


@pytest.mark.integration
def test_pool_samples_exact_native_frames_in_requested_interval(tmp_path) -> None:
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg absent")
    video = tmp_path / "test.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x90:rate=12:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
        capture_output=True,
    )

    source = PyAVPoolFrameSource()
    frames = list(source.frames(video, 3, start=0.5, duration=0.5))

    assert [frame.frame_index for frame in frames] == [6, 10]
    assert [frame.timestamp for frame in frames] == pytest.approx([0.5, 10 / 12])
    assert all((frame.image.width, frame.image.height) == (160, 90) for frame in frames)
    assert list(source.frames(video, 3, should_stop=lambda: True)) == []
