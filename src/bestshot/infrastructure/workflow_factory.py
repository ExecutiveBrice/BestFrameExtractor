"""Assemblage des dépendances locales des workflows applicatifs."""

from bestshot.infrastructure.config import (
    load_candidate_extraction_settings,
    load_composite_scoring_settings,
    load_deduplication_settings,
    load_face_scoring_settings,
    load_scene_detector_settings,
    load_selection_settings,
    load_technical_scoring_settings,
)
from bestshot.scoring.composite import CompositeScorer
from bestshot.scoring.face import create_face_scorer
from bestshot.scoring.technical import TechnicalScorer
from bestshot.selection.deduplicate import (
    DeduplicationSettings,
    Deduplicator,
    PerceptualHashSimilarityScorer,
)
from bestshot.selection.selector import BestFrameSelector, SelectionSettings
from bestshot.services.photo_pool import PhotoPoolRunner
from bestshot.services.video_selection import VideoSelectionWorkflow
from bestshot.video.candidate_extractor import CandidateExtractor, PyAVCandidateFrameBackend
from bestshot.video.scene_detector import PySceneDetectBackend, SceneDetector


def create_photo_pool_runner(fps: float | None = None) -> PhotoPoolRunner:
    """Assemble le mode pool strictement local ; aucun téléchargement implicite."""
    from dataclasses import replace

    from bestshot.infrastructure.clip_content import LocalClipContentScorer
    from bestshot.infrastructure.config import load_clip_content_settings, load_photo_pool_settings
    from bestshot.infrastructure.photo_pool import (
        LocalPoolImageWriter,
        LocalPoolRepository,
        OpenCVPoolQualityScorer,
        PyAVPoolFrameSource,
    )

    settings = load_photo_pool_settings()
    if fps is not None:
        settings = replace(settings, fps=fps)
    clip = load_clip_content_settings()
    return PhotoPoolRunner(
        PyAVPoolFrameSource(),
        OpenCVPoolQualityScorer(settings.analysis_width),
        LocalClipContentScorer(clip),
        LocalPoolImageWriter(),
        LocalPoolRepository(),
        settings,
        {
            "repo_id": clip.repo_id,
            "revision": clip.revision,
            "people_prompts": clip.people_prompts,
            "landscape_prompts": clip.landscape_prompts,
            "reject_prompts": clip.reject_prompts,
        },
    )


def create_video_selection_workflow(
    selection_settings: SelectionSettings | None = None,
    deduplication_settings: DeduplicationSettings | None = None,
) -> VideoSelectionWorkflow:
    """Construit le workflow de sélection avec les adaptateurs locaux configurés."""
    face_settings = load_face_scoring_settings()
    resolved_deduplication_settings = deduplication_settings or load_deduplication_settings()
    return VideoSelectionWorkflow(
        scene_detector=SceneDetector(PySceneDetectBackend(), load_scene_detector_settings()),
        candidate_extractor=CandidateExtractor(
            PyAVCandidateFrameBackend(), load_candidate_extraction_settings()
        ),
        technical_scorer=TechnicalScorer(load_technical_scoring_settings()),
        face_scorer=create_face_scorer(face_settings),
        composite_scorer=CompositeScorer(load_composite_scoring_settings()),
        deduplicator=Deduplicator(
            PerceptualHashSimilarityScorer(resolved_deduplication_settings.hash_size),
            resolved_deduplication_settings,
        ),
        selector=BestFrameSelector(selection_settings or load_selection_settings()),
    )
