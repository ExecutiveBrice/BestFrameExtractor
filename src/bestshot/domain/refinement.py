"""Modèles de classement grossier et de sélection fine de candidates."""

from dataclasses import dataclass

from bestshot.domain.candidate_frame import CandidateFrame
from bestshot.domain.composite_score import CompositeScore
from bestshot.domain.face_analysis import FaceScore
from bestshot.domain.technical_score import TechnicalScore


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    """Candidate issue de l'analyse grossière et son score composite explicité."""

    candidate: CandidateFrame
    composite_score: CompositeScore


@dataclass(frozen=True, slots=True)
class RefinedCandidate:
    """Meilleure frame locale sélectionnée autour d'une candidate grossière."""

    source_candidate: RankedCandidate
    selected_frame: CandidateFrame
    technical_score: TechnicalScore
    face_score: FaceScore
    composite_score: CompositeScore
