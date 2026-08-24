"""Création séquentielle de candidates d'analyse avec PyAV."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from bestshot.domain.candidate_frame import CandidateFrame, PreviewImage
from bestshot.domain.scene import Scene


@dataclass(frozen=True, slots=True)
class CandidateExtractionSettings:
    """Paramètres de cadence et de taille des aperçus d'analyse."""

    fps: float
    analysis_max_width: int


@dataclass(frozen=True, slots=True)
class DecodedFrame:
    """Frame décodée, avec son aperçu d'analyse uniquement."""

    timestamp: float
    frame_index: int
    source_width: int
    source_height: int
    preview: PreviewImage


class CandidateFrameBackend(Protocol):
    """Port de décodage séquentiel des frames vidéo."""

    def decode(
        self, video_path: Path, settings: CandidateExtractionSettings
    ) -> Iterator[DecodedFrame]:
        """Produit une frame décodée à la fois, dans l'ordre temporel."""


class CandidateExtractionError(RuntimeError):
    """La vidéo ne peut pas être décodée pour créer des candidates."""


class PyAVCandidateFrameBackend:
    """Adaptateur PyAV qui ne conserve qu'un aperçu redimensionné par frame."""

    def decode(
        self, video_path: Path, settings: CandidateExtractionSettings
    ) -> Iterator[DecodedFrame]:
        """Décode le premier flux vidéo séquentiellement, sans export de fichier."""
        try:
            import av
        except ImportError as error:
            raise CandidateExtractionError("PyAV n'est pas installé.") from error

        try:
            with av.open(str(video_path)) as container:
                if not container.streams.video:
                    raise CandidateExtractionError("Aucun flux vidéo n'a été trouvé.")
                stream = container.streams.video[0]
                for frame_index, frame in enumerate(container.decode(stream)):
                    timestamp = frame.time
                    if timestamp is None:
                        continue
                    yield DecodedFrame(
                        timestamp=float(timestamp),
                        frame_index=frame_index,
                        source_width=frame.width,
                        source_height=frame.height,
                        preview=_preview_from_frame(frame, settings.analysis_max_width),
                    )
        except CandidateExtractionError:
            raise
        except Exception as error:
            raise CandidateExtractionError(f"Impossible de décoder la vidéo : {error}") from error


class CandidateExtractor:
    """Échantillonne les frames décodées dans les intervalles des scènes."""

    def __init__(self, backend: CandidateFrameBackend, settings: CandidateExtractionSettings) -> None:
        self._backend = backend
        self._settings = settings

    def extract(self, video_path: Path, scenes: Sequence[Scene]) -> Iterator[CandidateFrame]:
        """Produit les candidates au fil du décodage, sans les accumuler."""
        if self._settings.fps <= 0:
            raise CandidateExtractionError("La cadence d'échantillonnage doit être positive.")
        if self._settings.analysis_max_width <= 0:
            raise CandidateExtractionError("La largeur maximale d'analyse doit être positive.")

        ordered_scenes = sorted(scenes, key=lambda scene: scene.start_time)
        scene_position = 0
        next_sample_time: float | None = None
        interval = 1.0 / self._settings.fps

        for frame in self._backend.decode(video_path, self._settings):
            while (
                scene_position < len(ordered_scenes)
                and frame.timestamp >= ordered_scenes[scene_position].end_time
            ):
                scene_position += 1
                next_sample_time = None
            if scene_position == len(ordered_scenes):
                break

            scene = ordered_scenes[scene_position]
            if frame.timestamp < scene.start_time:
                continue
            if next_sample_time is None:
                next_sample_time = scene.start_time
            if frame.timestamp < next_sample_time:
                continue

            yield CandidateFrame(
                scene_id=scene.index,
                timestamp=frame.timestamp,
                frame_index=frame.frame_index,
                source_width=frame.source_width,
                source_height=frame.source_height,
                preview=frame.preview,
            )
            while next_sample_time <= frame.timestamp:
                next_sample_time += interval


def _preview_from_frame(frame: Any, max_width: int) -> PreviewImage:
    """Convertit une frame PyAV en aperçu RGB, sans persister l'image source."""
    image = frame.to_image().convert("RGB")
    width, height = image.size
    if width > max_width:
        from PIL import Image

        resized_height = round(height * max_width / width)
        image = image.resize((max_width, resized_height), Image.Resampling.LANCZOS)
    return PreviewImage(width=image.width, height=image.height, rgb_bytes=image.tobytes())
