"""Découverte déterministe des photos locales destinées à l'apprentissage."""

from __future__ import annotations

from pathlib import Path

SUPPORTED_PHOTO_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


class PhotoLibraryError(ValueError):
    """Le dossier choisi ne peut pas être utilisé comme pool de photos."""


def discover_photos(directory: Path) -> tuple[Path, ...]:
    """Retourne les photos directes d'un dossier, dans un ordre stable.

    Les sous-dossiers sont volontairement ignorés : l'utilisateur contrôle ainsi
    précisément le corpus de préférences personnel.
    """
    if not directory.is_dir():
        raise PhotoLibraryError(f"Le dossier de photos est introuvable : {directory}")
    try:
        photos = [
            path.resolve()
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_PHOTO_EXTENSIONS
        ]
    except OSError as error:
        raise PhotoLibraryError(f"Impossible de lire le dossier de photos : {directory}") from error
    return tuple(sorted(photos, key=lambda path: (path.name.casefold(), str(path))))
