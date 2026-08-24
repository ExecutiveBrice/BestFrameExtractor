"""Raffinement local de candidates autour de leurs timestamps initiaux."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from bestshot.domain.candidate_frame import CandidateFrame, PreviewImage
from bestshot.domain.composite_score import CompositeScore
from bestshot.domain.face_analysis import FaceScore
from bestshot.domain.refinement import RankedCandidate, RefinedCandidate
from bestshot.domain.technical_score import TechnicalScore
from bestshot.video.candidate_extractor import (
    CandidateExtractionSettings,
    CandidateFrameBackend,
    DecodedFrame,
)


@dataclass(frozen=True, slots=True)
class RefinementSettings:
    """Fenêtre d'examen locale et nombre de candidates à raffiner par scène."""

    enabled: bool
    window_ms: int
    candidates_per_scene: int


class TechnicalFrameScorer(Protocol):
    """Port de calcul du score technique sur un aperçu de frame."""

    def score(self, preview: PreviewImage) -> TechnicalScore:
        """Retourne les détails du score technique."""


class FaceFrameScorer(Protocol):
    """Port de calcul du score de visage sur un aperçu de frame."""

    def score(self, preview: PreviewImage) -> FaceScore:
        """Retourne les détails du score de visage."""


class FrameCompositeScorer(Protocol):
    """Port de combinaison des détails techniques et visage."""

    def score(self, technical: TechnicalScore, face: FaceScore) -> CompositeScore:
        """Retourne un score composite expliqué."""


class CandidateRefiner:
    """Sélectionne la meilleure frame analysée une seule fois par index vidéo."""

    def __init__(
        self,
        decoder: CandidateFrameBackend,
        extraction_settings: CandidateExtractionSettings,
        technical_scorer: TechnicalFrameScorer,
        face_scorer: FaceFrameScorer,
        composite_scorer: FrameCompositeScorer,
        settings: RefinementSettings,
    ) -> None:
        self._decoder = decoder
        self._extraction_settings = extraction_settings
        self._technical_scorer = technical_scorer
        self._face_scorer = face_scorer
        self._composite_scorer = composite_scorer
        self._settings = settings

    def refine(
        self, video_path: Path, ranked_candidates: Sequence[RankedCandidate]
    ) -> list[RefinedCandidate]:
        """Raffine les meilleures candidates par scène sans exporter de frame source."""
        if not self._settings.enabled:
            return []
        if self._settings.window_ms <= 0 or self._settings.candidates_per_scene <= 0:
            raise ValueError("Les paramètres de raffinement doivent être positifs.")

        targets = _select_targets(ranked_candidates, self._settings.candidates_per_scene)
        best_frames: dict[int, _ScoredFrame] = {}
        scored_frames: dict[int, _ScoredFrame] = {}
        window_seconds = self._settings.window_ms / 1_000.0

        for decoded_frame in self._decoder.decode(video_path, self._extraction_settings):
            matching_targets = tuple(
                target
                for target in targets
                if abs(decoded_frame.timestamp - target.candidate.timestamp) <= window_seconds
            )
            if not matching_targets:
                continue
            scored_frame = scored_frames.get(decoded_frame.frame_index)
            if scored_frame is None:
                scored_frame = self._score_frame(decoded_frame, matching_targets[0].candidate.scene_id)
                scored_frames[decoded_frame.frame_index] = scored_frame
            for target in matching_targets:
                current = best_frames.get(id(target))
                if current is None or _is_better(scored_frame, current, target.candidate.timestamp):
                    best_frames[id(target)] = scored_frame

        return [
            RefinedCandidate(
                source_candidate=target,
                selected_frame=best_frames[id(target)].candidate,
                technical_score=best_frames[id(target)].technical_score,
                face_score=best_frames[id(target)].face_score,
                composite_score=best_frames[id(target)].composite_score,
            )
            for target in targets
            if id(target) in best_frames
        ]

    def _score_frame(self, frame: DecodedFrame, scene_id: int) -> _ScoredFrame:
        candidate = CandidateFrame(
            scene_id=scene_id,
            timestamp=frame.timestamp,
            frame_index=frame.frame_index,
            source_width=frame.source_width,
            source_height=frame.source_height,
            preview=frame.preview,
        )
        technical_score = self._technical_scorer.score(frame.preview)
        face_score = self._face_scorer.score(frame.preview)
        return _ScoredFrame(
            candidate=candidate,
            technical_score=technical_score,
            face_score=face_score,
            composite_score=self._composite_scorer.score(technical_score, face_score),
        )


@dataclass(frozen=True, slots=True)
class _ScoredFrame:
    candidate: CandidateFrame
    technical_score: TechnicalScore
    face_score: FaceScore
    composite_score: CompositeScore


def _select_targets(
    ranked_candidates: Sequence[RankedCandidate], candidates_per_scene: int
) -> list[RankedCandidate]:
    per_scene: defaultdict[int, list[RankedCandidate]] = defaultdict(list)
    for ranked_candidate in ranked_candidates:
        per_scene[ranked_candidate.candidate.scene_id].append(ranked_candidate)
    return [
        candidate
        for scene_candidates in per_scene.values()
        for candidate in sorted(
            scene_candidates,
            key=lambda item: item.composite_score.final_score,
            reverse=True,
        )[:candidates_per_scene]
    ]


def _is_better(
    candidate: _ScoredFrame, current: _ScoredFrame, target_timestamp: float
) -> bool:
    if candidate.composite_score.final_score != current.composite_score.final_score:
        return candidate.composite_score.final_score > current.composite_score.final_score
    return abs(candidate.candidate.timestamp - target_timestamp) < abs(
        current.candidate.timestamp - target_timestamp
    )
