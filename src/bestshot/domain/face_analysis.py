"""Modèles anonymes de qualité de visage, sans reconnaissance d'identité."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FaceAnalysis:
    """Mesures brutes d'un visage détecté dans un aperçu d'analyse."""

    detection_confidence: float | None
    relative_size: float
    yaw_degrees: float | None
    eyes_open: float | None
    positive_expression: float | None
    sharpness_variance: float
    is_cut_off: bool


@dataclass(frozen=True, slots=True)
class FaceScore:
    """Scores agrégés du groupe ; ``None`` signifie qu'aucun visage n'est présent."""

    analyses: tuple[FaceAnalysis, ...]
    detection_confidence: float | None
    size: float | None
    orientation: float | None
    eyes_open: float | None
    positive_expression: float | None
    sharpness: float | None
    crop: float | None
    global_score: float | None
