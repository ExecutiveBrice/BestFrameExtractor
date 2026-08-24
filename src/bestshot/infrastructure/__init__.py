"""Adaptateurs aux fichiers et aux bibliothèques externes."""

from bestshot.infrastructure.candidate_repository import (
    CandidateRepositoryError,
    LocalCandidatePreviewRepository,
)
from bestshot.infrastructure.config import (
    DEFAULT_CONFIG_PATH,
    load_candidate_extraction_settings,
    load_composite_scoring_settings,
    load_deduplication_settings,
    load_export_settings,
    load_face_scoring_settings,
    load_refinement_settings,
    load_scene_detector_settings,
    load_selection_settings,
    load_technical_scoring_settings,
)
from bestshot.infrastructure.ffmpeg import FFmpegExportError, FFmpegFrameExporter
from bestshot.infrastructure.ffprobe import SubprocessFFprobeRunner

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "CandidateRepositoryError",
    "FFmpegExportError",
    "FFmpegFrameExporter",
    "LocalCandidatePreviewRepository",
    "SubprocessFFprobeRunner",
    "load_candidate_extraction_settings",
    "load_composite_scoring_settings",
    "load_deduplication_settings",
    "load_export_settings",
    "load_face_scoring_settings",
    "load_refinement_settings",
    "load_scene_detector_settings",
    "load_selection_settings",
    "load_technical_scoring_settings",
]
