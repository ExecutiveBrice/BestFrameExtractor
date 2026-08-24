"""Cas d'usage de détection et de représentation textuelle des scènes."""

from pathlib import Path

from bestshot.domain.scene import Scene
from bestshot.video.scene_detector import SceneDetector


def detect_scenes(video_path: Path, detector: SceneDetector) -> list[Scene]:
    """Retourne les scènes détectées pour une vidéo locale."""
    return detector.detect(video_path)


def format_scenes(scenes: list[Scene]) -> str:
    """Produit une liste de scènes stable pour la sortie de la CLI."""
    if not scenes:
        return "Aucune scène détectée."
    return "\n".join(
        f"Scène {scene.index}: {scene.start_time:.3f}s → {scene.end_time:.3f}s "
        f"(durée : {scene.duration:.3f}s)"
        for scene in scenes
    )
