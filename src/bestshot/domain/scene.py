"""Modèle de domaine représentant une scène détectée."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Scene:
    """Intervalle temporel contigu d'une vidéo, exprimé en secondes."""

    index: int
    start_time: float
    end_time: float
    duration: float
