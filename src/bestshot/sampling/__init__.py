"""Pipeline V2 de présélection temporelle des frames vidéo."""

from bestshot.sampling.candidate_generator import (
    CandidateGenerationError,
    CandidateGenerationResult,
    CandidateGenerator,
    PresampledCandidate,
)
from bestshot.sampling.sharpness_ranker import RankedAnalysisFrame, SharpnessRanker
from bestshot.sampling.temporal_sampler import (
    AnalysisFrame,
    GrayscaleImage,
    PresamplingSettings,
    TemporalSampler,
    TemporalSamplingError,
)

__all__ = [
    "AnalysisFrame",
    "CandidateGenerationError",
    "CandidateGenerationResult",
    "CandidateGenerator",
    "GrayscaleImage",
    "PresampledCandidate",
    "PresamplingSettings",
    "RankedAnalysisFrame",
    "SharpnessRanker",
    "TemporalSampler",
    "TemporalSamplingError",
]
