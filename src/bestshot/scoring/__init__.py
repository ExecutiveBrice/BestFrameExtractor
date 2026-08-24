"""Scorers de qualité appliqués aux aperçus d'analyse."""

from bestshot.scoring.composite import (
    CompositeScorer,
    CompositeScoringSettings,
    CompositeWeights,
    NeutralAestheticScorer,
    NeutralCompositionScorer,
)
from bestshot.scoring.face import FaceScorer, FaceScoringSettings, create_face_scorer
from bestshot.scoring.technical import TechnicalScorer, TechnicalScoringSettings

__all__ = [
    "CompositeScorer",
    "CompositeScoringSettings",
    "CompositeWeights",
    "FaceScorer",
    "FaceScoringSettings",
    "NeutralAestheticScorer",
    "NeutralCompositionScorer",
    "TechnicalScorer",
    "TechnicalScoringSettings",
    "create_face_scorer",
]
