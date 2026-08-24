"""Composants applicatifs relatifs aux vidéos."""

from bestshot.video.candidate_extractor import (
    CandidateExtractionError,
    CandidateExtractionSettings,
    CandidateExtractor,
    PyAVCandidateFrameBackend,
)
from bestshot.video.candidate_refiner import CandidateRefiner, RefinementSettings
from bestshot.video.candidate_repository import (
    CandidatePreviewRepository,
    CandidateRepositoryResult,
)
from bestshot.video.probe import FFprobeRunner, VideoProbe, VideoProbeError
from bestshot.video.scene_detector import (
    PySceneDetectBackend,
    SceneDetectionError,
    SceneDetector,
    SceneDetectorSettings,
)

__all__ = [
    "CandidateExtractionError",
    "CandidateExtractionSettings",
    "CandidateExtractor",
    "CandidatePreviewRepository",
    "CandidateRefiner",
    "CandidateRepositoryResult",
    "FFprobeRunner",
    "PyAVCandidateFrameBackend",
    "PySceneDetectBackend",
    "RefinementSettings",
    "SceneDetectionError",
    "SceneDetector",
    "SceneDetectorSettings",
    "VideoProbe",
    "VideoProbeError",
]
