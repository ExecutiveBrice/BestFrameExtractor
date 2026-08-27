"""Décodage PyAV ciblé des candidates V2 à transmettre au provider d'embeddings."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from bestshot.domain.preview_image import PreviewImage
from bestshot.sampling.candidate_generator import PresampledCandidate


@dataclass(frozen=True, slots=True)
class CandidatePreview:
    """Candidate V2 avec son aperçu RGB créé uniquement pour son embedding."""

    candidate: PresampledCandidate
    preview: PreviewImage


class CandidatePreviewReader(Protocol):
    """Port de lecture des seules frames retenues par le présampling."""

    def read(
        self,
        video_path: Path,
        candidates: Sequence[PresampledCandidate],
        max_width: int,
    ) -> Iterator[CandidatePreview]:
        """Produit les aperçus RGB ciblés dans l'ordre du décodage séquentiel."""


class EmbeddingFrameReadError(RuntimeError):
    """Les candidates d'embedding ne peuvent pas être relues depuis la vidéo."""


class PyAVCandidatePreviewReader:
    """Décode tout le flux, sans RGB/Pillow pour les frames absentes de la demande."""

    def read(
        self,
        video_path: Path,
        candidates: Sequence[PresampledCandidate],
        max_width: int,
    ) -> Iterator[CandidatePreview]:
        """Convertit seulement les index de frame qui requièrent une inférence DINOv2."""
        if max_width <= 0:
            raise EmbeddingFrameReadError("La largeur maximale d'embedding doit être positive.")
        requested = {candidate.frame_index: candidate for candidate in candidates}
        if len(requested) != len(candidates):
            raise EmbeddingFrameReadError("Les candidates à embedder doivent avoir des index uniques.")
        if not requested:
            return
        try:
            import av
        except ImportError as error:
            raise EmbeddingFrameReadError("PyAV n'est pas installé.") from error
        try:
            with av.open(str(video_path)) as container:
                if not container.streams.video:
                    raise EmbeddingFrameReadError("Aucun flux vidéo n'a été trouvé.")
                stream = container.streams.video[0]
                for frame_index, frame in enumerate(container.decode(stream)):
                    candidate = requested.get(frame_index)
                    if candidate is None:
                        continue
                    yield CandidatePreview(candidate, _preview_from_frame(frame, max_width))
        except EmbeddingFrameReadError:
            raise
        except Exception as error:
            raise EmbeddingFrameReadError(f"Impossible de relire les frames candidates : {error}") from error


def _preview_from_frame(frame: Any, max_width: int) -> PreviewImage:
    """Réduit une frame demandée en RGB via PyAV, sans conversion Pillow intermédiaire."""
    source_width = int(frame.width)
    source_height = int(frame.height)
    target_width = min(source_width, max_width)
    target_height = max(1, round(source_height * target_width / source_width))
    rgb_frame = cast(Any, frame).reformat(width=target_width, height=target_height, format="rgb24")
    pixels = rgb_frame.to_ndarray()
    return PreviewImage(width=target_width, height=target_height, rgb_bytes=pixels.tobytes())
