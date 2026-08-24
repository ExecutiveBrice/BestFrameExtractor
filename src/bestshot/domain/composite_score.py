"""Résultat expliqué de la combinaison des scores de qualité."""

from dataclasses import dataclass

from bestshot.domain.face_analysis import FaceScore
from bestshot.domain.technical_score import TechnicalScore


@dataclass(frozen=True, slots=True)
class AestheticScore:
    """Score esthétique extensible ; neutre tant qu'aucun scorer n'est fourni."""

    global_score: float
    is_neutral: bool = False
    model_name: str | None = None
    inference_ms: float | None = None
    status: str = "available"


@dataclass(frozen=True, slots=True)
class CompositionScore:
    """Score de composition extensible ; neutre tant qu'aucun scorer n'est fourni."""

    global_score: float
    is_neutral: bool = False


@dataclass(frozen=True, slots=True)
class CompositeReason:
    """Contribution pondérée et lisible d'un critère au classement final."""

    criterion: str
    score: float
    weight: float
    contribution: float
    detail: str


@dataclass(frozen=True, slots=True)
class CompositeScore:
    """Score final détaillé, jamais réduit à un flottant opaque."""

    final_score: float
    profile: str
    technical: TechnicalScore
    face: FaceScore
    aesthetic: AestheticScore
    composition: CompositionScore
    reasons: tuple[CompositeReason, ...]
