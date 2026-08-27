"""Stockage externe des aperçus réduits : jamais de pixels dans SQLite."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from bestshot.domain.preview_image import PreviewImage


class PreviewCacheError(RuntimeError):
    """Un aperçu réduit ne peut pas être écrit dans le cache local."""


class PreviewCache:
    """Écrit des JPEG limités par le présampling, en dehors de la base SQLite."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def put(self, video_hash: str, frame_index: int, image: PreviewImage) -> str:
        """Persiste un aperçu local déterministe et renvoie sa référence absolue."""
        if not video_hash or frame_index < 0:
            raise PreviewCacheError("La clé d'aperçu est invalide.")
        if len(image.rgb_bytes) != image.width * image.height * 3:
            raise PreviewCacheError("Les pixels RGB de l'aperçu sont invalides.")
        try:
            from PIL import Image
        except ImportError as error:
            raise PreviewCacheError("Pillow est requis pour écrire les aperçus locaux.") from error
        self._directory.mkdir(parents=True, exist_ok=True)
        destination = self._directory / f"{video_hash[:16]}-{frame_index:08d}.jpg"
        temporary = self._directory / f".{destination.name}.{uuid4().hex}.tmp"
        try:
            image_object = Image.frombytes("RGB", (image.width, image.height), image.rgb_bytes)
            image_object.save(temporary, format="JPEG", quality=88, optimize=True)
            os.replace(temporary, destination)
        except OSError as error:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise PreviewCacheError(f"Impossible d'écrire l'aperçu : {destination}") from error
        return str(destination)
