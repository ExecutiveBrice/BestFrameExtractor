"""Estimation locale du nombre d'images avant filtrage."""

import logging
from dataclasses import dataclass
from math import ceil, isfinite
from pathlib import Path

from bestshot.services.batch import find_videos
from bestshot.video.probe import VideoProbe

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VideoDurationSummary:
    """Durée cumulée des vidéos lisibles et nombre de durées indisponibles."""

    duration_seconds: float
    video_count: int
    unavailable_count: int

    def estimate_images(self, fps: int) -> int:
        """Estime les candidates : durée en secondes × photos par seconde."""
        if fps <= 0:
            raise ValueError("La cadence doit être positive.")
        return ceil(self.duration_seconds * fps)


def inspect_video_durations(directory: Path, probe: VideoProbe) -> VideoDurationSummary:
    """Inspecte les mêmes vidéos que le traitement, sans décoder leurs images."""
    videos = find_videos(directory)
    duration_seconds = 0.0
    unavailable_count = 0
    for video in videos:
        try:
            duration = probe.inspect(video).duration_seconds
        except (OSError, RuntimeError, ValueError) as error:
            logger.warning("Durée indisponible pour %s : %s", video.name, error)
            duration = None
        if duration is None or not isfinite(duration) or duration <= 0:
            unavailable_count += 1
        else:
            duration_seconds += duration
    return VideoDurationSummary(duration_seconds, len(videos), unavailable_count)
