"""Modèle de domaine des mesures de qualité technique d'une image."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TechnicalScore:
    """Scores normalisés entre 0 (défavorable) et 1 (favorable)."""

    sharpness: float
    exposure: float
    burned_pixels: float
    underexposed_pixels: float
    contrast: float
    motion_blur: float
    global_score: float
