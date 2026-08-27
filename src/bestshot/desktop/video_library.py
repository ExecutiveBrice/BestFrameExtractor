"""Découverte déterministe des vidéos locales choisies dans l'interface."""

from __future__ import annotations

from pathlib import Path

SUPPORTED_VIDEO_EXTENSIONS = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"})


class VideoLibraryError(ValueError):
    """Le dossier sélectionné ne peut pas être utilisé comme bibliothèque vidéo."""


def discover_videos(directory: Path) -> tuple[Path, ...]:
    """Retourne les vidéos du dossier, sans parcourir ses sous-dossiers.

    Le choix non récursif évite d'analyser par surprise des vidéos rangées dans
    d'autres albums. Les chemins sont normalisés pour que l'ingestion et le
    dataset local identifient toujours la même source.
    """
    if not directory.is_dir():
        raise VideoLibraryError(f"Le dossier vidéo est introuvable : {directory}")
    try:
        videos = [
            path.resolve()
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        ]
    except OSError as error:
        raise VideoLibraryError(f"Impossible de lire le dossier vidéo : {directory}") from error
    return tuple(sorted(videos, key=lambda path: (path.name.casefold(), str(path))))
