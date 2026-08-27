"""Génération de candidates stables à partir des fenêtres temporelles V2."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from bestshot.sampling.sharpness_ranker import SharpnessRanker
from bestshot.sampling.temporal_sampler import AnalysisFrame, PresamplingSettings, TemporalSampler


@dataclass(frozen=True, slots=True)
class PresampledCandidate:
    """Référence légère vers une frame retenue dans sa fenêtre temporelle."""

    timestamp: float
    frame_index: int
    source_width: int
    source_height: int
    bucket_index: int
    sharpness: float = 0.0


@dataclass(frozen=True, slots=True)
class CandidateGenerationResult:
    """Résultat complet du premier passage, sans conserver les buffers d'analyse."""

    candidates: tuple[PresampledCandidate, ...]
    video_frame_count: int
    analyzed_frame_count: int

    @property
    def candidate_count(self) -> int:
        """Nombre de candidates produites par les fenêtres non vides."""
        return len(self.candidates)


class CandidateGenerationError(RuntimeError):
    """Les candidates V2 ne peuvent pas être générées avec ces réglages."""


class CandidateGenerator:
    """Conserve les ``keep_per_bucket`` frames les plus nettes par fenêtre fixe."""

    def __init__(
        self,
        temporal_sampler: TemporalSampler,
        sharpness_ranker: SharpnessRanker,
        settings: PresamplingSettings,
    ) -> None:
        self._temporal_sampler = temporal_sampler
        self._sharpness_ranker = sharpness_ranker
        self._settings = settings

    def generate(self, video_path: Path) -> CandidateGenerationResult:
        """Consomme la vidéo une fois et ne garde que les métadonnées des candidates."""
        self._validate_settings()
        sampled = self._temporal_sampler.sample(video_path)
        candidates: list[PresampledCandidate] = []
        bucket_frames: list[AnalysisFrame] = []
        current_bucket: int | None = None

        for frame in sampled.samples:
            bucket_index = math.floor(frame.timestamp / self._settings.bucket_seconds)
            if current_bucket is None:
                current_bucket = bucket_index
            elif bucket_index != current_bucket:
                candidates.extend(self._candidates_for_bucket(current_bucket, bucket_frames))
                bucket_frames = []
                current_bucket = bucket_index
            bucket_frames.append(frame)

        if current_bucket is not None:
            candidates.extend(self._candidates_for_bucket(current_bucket, bucket_frames))

        return CandidateGenerationResult(
            candidates=tuple(candidates),
            video_frame_count=sampled.statistics.video_frame_count,
            analyzed_frame_count=sampled.statistics.analyzed_frame_count,
        )

    def _candidates_for_bucket(
        self,
        bucket_index: int,
        frames: list[AnalysisFrame],
    ) -> list[PresampledCandidate]:
        ranked = self._sharpness_ranker.rank(frames, self._settings.keep_per_bucket)
        return [
            PresampledCandidate(
                timestamp=item.frame.timestamp,
                frame_index=item.frame.frame_index,
                source_width=item.frame.source_width,
                source_height=item.frame.source_height,
                bucket_index=bucket_index,
                sharpness=item.sharpness,
            )
            for item in sorted(ranked, key=lambda item: (item.frame.timestamp, item.frame.frame_index))
        ]

    def _validate_settings(self) -> None:
        if self._settings.bucket_seconds <= 0:
            raise CandidateGenerationError("La durée d'une fenêtre doit être positive.")
        if self._settings.keep_per_bucket <= 0:
            raise CandidateGenerationError("Le nombre de frames à conserver doit être positif.")
