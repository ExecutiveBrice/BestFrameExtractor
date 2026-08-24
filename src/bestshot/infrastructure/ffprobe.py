"""Adaptateur local fondé sur l'exécutable ffprobe."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from bestshot.video.probe import VideoProbeError


class SubprocessFFprobeRunner:
    """Exécute ffprobe et retourne sa réponse JSON, sans transfert réseau."""

    def __init__(self, executable: str = "ffprobe") -> None:
        self._executable = executable

    def probe(self, video_path: Path) -> Mapping[str, object]:
        """Interroge la première piste vidéo de ``video_path``."""
        command = [
            self._executable,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            (
                "stream=codec_type,codec_name,width,height,avg_frame_rate,nb_frames,bit_rate:"
                "stream_tags=rotate:stream_side_data=rotation:format=duration,bit_rate:"
                "format_tags=creation_time"
            ),
            "-of",
            "json",
            str(video_path),
        ]
        try:
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError) as error:
            raise VideoProbeError(f"Impossible d'exécuter ffprobe : {error}") from error
        try:
            decoded = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise VideoProbeError("ffprobe n'a pas retourné de JSON valide.") from error
        if not isinstance(decoded, Mapping):
            raise VideoProbeError("ffprobe n'a pas retourné un objet JSON.")
        return cast(Mapping[str, object], decoded)
