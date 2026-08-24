"""Tests unitaires de l'adaptateur subprocess vers ffprobe."""

import subprocess
from pathlib import Path

from pytest import MonkeyPatch

from bestshot.infrastructure.ffprobe import SubprocessFFprobeRunner


def test_runner_builds_ffprobe_command_and_decodes_json(monkeypatch: MonkeyPatch) -> None:
    def fake_run(
        command: list[str], *, check: bool, capture_output: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        assert command[0] == "custom-ffprobe"
        assert command[-1] == "movie.mp4"
        assert any("stream_tags=rotate" in argument for argument in command)
        assert check is True
        assert capture_output is True
        assert text is True
        return subprocess.CompletedProcess(command, 0, '{"streams": [], "format": {}}', "")

    monkeypatch.setattr("bestshot.infrastructure.ffprobe.subprocess.run", fake_run)

    response = SubprocessFFprobeRunner("custom-ffprobe").probe(Path("movie.mp4"))

    assert response == {"streams": [], "format": {}}
