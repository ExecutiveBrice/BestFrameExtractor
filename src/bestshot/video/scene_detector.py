"""Détection locale de scènes avec PySceneDetect, sans extraction d'images."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from bestshot.domain.scene import Scene


@dataclass(frozen=True, slots=True)
class SceneDetectorSettings:
    """Paramètres d'``AdaptiveDetector`` lus depuis la configuration YAML."""

    adaptive_threshold: float
    min_scene_len_frames: int
    window_width: int
    min_content_val: float


class SceneDetectionBackend(Protocol):
    """Port qui retourne les limites temporelles calculées par un détecteur."""

    def detect(
        self, video_path: Path, settings: SceneDetectorSettings
    ) -> Sequence[tuple[float, float]]:
        """Retourne les paires ``(début, fin)`` en secondes."""


class SceneDetectionError(RuntimeError):
    """PySceneDetect n'a pas pu analyser la vidéo demandée."""


class PySceneDetectBackend:
    """Adaptateur PySceneDetect fondé sur ``AdaptiveDetector``."""

    def detect(
        self, video_path: Path, settings: SceneDetectorSettings
    ) -> Sequence[tuple[float, float]]:
        """Détecte les scènes en parcourant le flux vidéo séquentiellement."""
        try:
            from scenedetect import (  # type: ignore[import-untyped]
                AdaptiveDetector,
                SceneManager,
                open_video,
            )
        except ImportError as error:
            raise SceneDetectionError("PySceneDetect n'est pas installé.") from error

        try:
            video = open_video(str(video_path))
            scene_manager = SceneManager()
            scene_manager.add_detector(
                AdaptiveDetector(
                    adaptive_threshold=settings.adaptive_threshold,
                    min_scene_len=settings.min_scene_len_frames,
                    window_width=settings.window_width,
                    min_content_val=settings.min_content_val,
                )
            )
            scene_manager.detect_scenes(video)
            scenes = [
                (float(start_time.get_seconds()), float(end_time.get_seconds()))
                for start_time, end_time in scene_manager.get_scene_list()
            ]
            if scenes:
                return scenes
            duration = getattr(video, "duration", None)
            if duration is not None:
                duration_seconds = float(duration.get_seconds())
                if duration_seconds > 0:
                    return [(0.0, duration_seconds)]
            return []
        except Exception as error:
            raise SceneDetectionError(f"Impossible de détecter les scènes : {error}") from error


class SceneDetector:
    """Convertit les limites produites par l'adaptateur en modèles de domaine."""

    def __init__(self, backend: SceneDetectionBackend, settings: SceneDetectorSettings) -> None:
        self._backend = backend
        self._settings = settings

    def detect(self, video_path: Path) -> list[Scene]:
        """Retourne les scènes 1-indexées, sans extraire aucune image."""
        boundaries = self._backend.detect(video_path, self._settings)
        return [
            Scene(
                index=index,
                start_time=start_time,
                end_time=end_time,
                duration=max(0.0, end_time - start_time),
            )
            for index, (start_time, end_time) in enumerate(boundaries, start=1)
        ]
