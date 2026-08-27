"""Décodage séquentiel et pré-échantillonnage temporel du pipeline V2."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class PresamplingSettings:
    """Réglages communs du présampling temporel et de ses fenêtres."""

    analysis_fps: float
    bucket_seconds: float
    keep_per_bucket: int
    analysis_max_width: int


@dataclass(frozen=True, slots=True)
class GrayscaleImage:
    """Image réduite en niveaux de gris, réservée au calcul de netteté."""

    width: int
    height: int
    gray_bytes: bytes


@dataclass(frozen=True, slots=True)
class DecodedVideoFrame:
    """Frame décodée dont les pixels ne sont convertis qu'en cas d'analyse."""

    timestamp: float | None
    frame_index: int
    source_width: int
    source_height: int
    payload: object = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class AnalysisFrame:
    """Frame pré-échantillonnée avec le seul buffer nécessaire à sa comparaison."""

    timestamp: float
    frame_index: int
    source_width: int
    source_height: int
    grayscale: GrayscaleImage


@dataclass(slots=True)
class TemporalSamplingStatistics:
    """Compteurs d'un décodage complet, mis à jour au fil du flux."""

    video_frame_count: int = 0
    analyzed_frame_count: int = 0


@dataclass(slots=True)
class TemporalSampleStream:
    """Flux de frames analysables et compteurs associés au même passage vidéo."""

    samples: Iterator[AnalysisFrame]
    statistics: TemporalSamplingStatistics


class TemporalSamplingBackend(Protocol):
    """Port local : décodage brut puis conversion ciblée des seules frames retenues."""

    def decode(self, video_path: Path) -> Iterator[DecodedVideoFrame]:
        """Produit toutes les frames vidéo dans leur ordre de décodage."""

    def to_grayscale(self, frame: DecodedVideoFrame, max_width: int) -> GrayscaleImage:
        """Réduit une frame sélectionnée sans conversion RGB/Pillow intermédiaire."""


class TemporalSamplingError(RuntimeError):
    """Le flux vidéo ou ses réglages ne permettent pas le présampling."""


class TemporalSampler:
    """Sélectionne environ ``analysis_fps`` frames/s avant toute conversion de pixels."""

    def __init__(self, backend: TemporalSamplingBackend, settings: PresamplingSettings) -> None:
        self._backend = backend
        self._settings = settings

    def sample(self, video_path: Path) -> TemporalSampleStream:
        """Retourne un flux ; les compteurs deviennent définitifs après sa consommation."""
        self._validate_settings()
        statistics = TemporalSamplingStatistics()
        return TemporalSampleStream(
            samples=self._iter_samples(video_path, statistics),
            statistics=statistics,
        )

    def _iter_samples(
        self,
        video_path: Path,
        statistics: TemporalSamplingStatistics,
    ) -> Iterator[AnalysisFrame]:
        interval = 1.0 / self._settings.analysis_fps
        next_sample_time = 0.0

        for frame in self._backend.decode(video_path):
            statistics.video_frame_count += 1
            if frame.timestamp is None or not math.isfinite(frame.timestamp):
                continue
            if frame.timestamp < next_sample_time:
                continue

            grayscale = self._backend.to_grayscale(frame, self._settings.analysis_max_width)
            statistics.analyzed_frame_count += 1
            yield AnalysisFrame(
                timestamp=frame.timestamp,
                frame_index=frame.frame_index,
                source_width=frame.source_width,
                source_height=frame.source_height,
                grayscale=grayscale,
            )
            while next_sample_time <= frame.timestamp:
                next_sample_time += interval

    def _validate_settings(self) -> None:
        if self._settings.analysis_fps <= 0:
            raise TemporalSamplingError("La cadence d'analyse doit être positive.")
        if self._settings.analysis_max_width <= 0:
            raise TemporalSamplingError("La largeur maximale d'analyse doit être positive.")
