"""Modèles et règles métier indépendants des détails techniques."""

from bestshot.domain.candidate_frame import CandidateFrame, PreviewImage
from bestshot.domain.composite_score import (
    AestheticScore,
    CompositeReason,
    CompositeScore,
    CompositionScore,
)
from bestshot.domain.deduplication import DeduplicationResult, DuplicateCandidate
from bestshot.domain.face_analysis import FaceAnalysis, FaceScore
from bestshot.domain.refinement import RankedCandidate, RefinedCandidate
from bestshot.domain.scene import Scene
from bestshot.domain.selection import SelectionRejection, SelectionResult
from bestshot.domain.technical_score import TechnicalScore
from bestshot.domain.video_info import VideoInfo

__all__ = [
    "AestheticScore",
    "CandidateFrame",
    "CompositeReason",
    "CompositeScore",
    "CompositionScore",
    "DeduplicationResult",
    "DuplicateCandidate",
    "FaceAnalysis",
    "FaceScore",
    "PreviewImage",
    "RankedCandidate",
    "RefinedCandidate",
    "Scene",
    "SelectionRejection",
    "SelectionResult",
    "TechnicalScore",
    "VideoInfo",
]
