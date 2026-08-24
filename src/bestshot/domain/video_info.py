"""Modèle de domaine représentant les métadonnées d'une vidéo."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class VideoInfo:
    """Métadonnées collectées localement à partir d'un flux vidéo."""

    path: Path
    codec: str
    width: int
    height: int
    fps: float
    approximate_frame_count: int | None
    duration_seconds: float | None
    bitrate: int | None
    orientation_degrees: int
    creation_time: datetime | None
