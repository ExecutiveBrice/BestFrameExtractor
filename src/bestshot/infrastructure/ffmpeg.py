"""Adaptateur local FFmpeg pour extraire une frame native à un timestamp donné."""

import subprocess
from pathlib import Path


class FFmpegExportError(RuntimeError):
    """FFmpeg n'a pas pu produire l'image demandée."""


class FFmpegFrameExporter:
    """Extrait une frame de la vidéo source sans redimensionnement."""

    def __init__(self, executable: str = "ffmpeg") -> None:
        self._executable = executable

    def extract(self, video_path: Path, timestamp: float, output_path: Path, jpeg_quality: int) -> None:
        """Écrit une image native à ``timestamp`` dans le format de ``output_path``."""
        command = [
            self._executable,
            "-y",
            "-i",
            str(video_path),
            "-ss",
            f"{timestamp:.6f}",
            "-map",
            "0:v:0",
            "-frames:v",
            "1",
        ]
        if output_path.suffix.lower() in {".jpg", ".jpeg"}:
            command.extend(["-q:v", str(jpeg_quality)])
        command.append(str(output_path))
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError) as error:
            raise FFmpegExportError(f"Impossible d'extraire la frame avec FFmpeg : {error}") from error
