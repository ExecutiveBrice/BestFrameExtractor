"""Lecture locale et réduite des photos utilisées pour les préférences personnelles."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from bestshot.domain.preview_image import PreviewImage


class PhotoPreviewReadError(RuntimeError):
    """Une photo locale du pool ne peut pas être lue pour son aperçu ou DINOv2."""


class PhotoPreviewReader(Protocol):
    """Port de lecture des photos, sans persister de pixels dans SQLite."""

    def read(self, photo_path: Path, max_width: int) -> PreviewImage:
        """Retourne une copie RGB réduite de la photo source."""


class PillowPhotoPreviewReader:
    """Adaptateur Pillow qui limite la taille avant l'embedding et l'aperçu."""

    def read(self, photo_path: Path, max_width: int) -> PreviewImage:
        if max_width <= 0:
            raise ValueError("La largeur maximale d'aperçu doit être positive.")
        try:
            from PIL import Image
        except ImportError as error:
            raise PhotoPreviewReadError("Pillow est requis pour lire les photos locales.") from error
        try:
            with Image.open(photo_path) as source:
                rgb = source.convert("RGB")
                if rgb.width > max_width:
                    target_height = max(1, round(rgb.height * max_width / rgb.width))
                    rgb = rgb.resize((max_width, target_height), Image.Resampling.LANCZOS)
                return PreviewImage(rgb.width, rgb.height, rgb.tobytes())
        except (OSError, ValueError) as error:
            raise PhotoPreviewReadError(f"Impossible de lire la photo locale : {photo_path}") from error
