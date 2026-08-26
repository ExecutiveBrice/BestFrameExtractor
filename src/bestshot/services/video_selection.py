"""Cas d'usage de sélection complète d'une vidéo, réutilisable en lot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bestshot.domain.selection import SelectionResult
from bestshot.scoring.composite import CompositeScorer
from bestshot.scoring.face import FaceScoreProvider
from bestshot.scoring.technical import TechnicalScorer
from bestshot.selection.deduplicate import Deduplicator
from bestshot.selection.selector import BestFrameSelector
from bestshot.services.candidates import extract_candidates
from bestshot.services.scenes import detect_scenes
from bestshot.services.selection import rank_candidates, select_best_frames
from bestshot.video.candidate_extractor import CandidateExtractor
from bestshot.video.scene_detector import SceneDetector


@dataclass(frozen=True, slots=True)
class VideoSelectionWorkflow:
    """Assemble les dépendances locales nécessaires à une sélection finale."""

    scene_detector: SceneDetector
    candidate_extractor: CandidateExtractor
    technical_scorer: TechnicalScorer
    face_scorer: FaceScoreProvider
    composite_scorer: CompositeScorer
    deduplicator: Deduplicator
    selector: BestFrameSelector

    def select(self, video_path: Path, count: int | None) -> SelectionResult:
        """Détecte, classe, dédoublonne et sélectionne les frames d'une vidéo."""
        scenes = detect_scenes(video_path, self.scene_detector)
        ranked = rank_candidates(
            extract_candidates(video_path, scenes, self.candidate_extractor),
            self.technical_scorer,
            self.face_scorer,
            self.composite_scorer,
        )
        return select_best_frames(ranked, scenes, self.deduplicator, self.selector, count)
