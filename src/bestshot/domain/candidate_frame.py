"""Modèles de domaine représentant une image candidate d'analyse."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PreviewImage:
    """Aperçu RGB redimensionné, conservé uniquement en mémoire."""

    width: int
    height: int
    rgb_bytes: bytes


@dataclass(frozen=True, slots=True)
class CandidateFrame:
    """Image candidate associée à son image source et à une scène."""

    scene_id: int
    timestamp: float
    frame_index: int
    source_width: int
    source_height: int
    preview: PreviewImage
