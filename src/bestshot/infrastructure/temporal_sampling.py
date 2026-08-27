"""Adaptateur PyAV du présampling temporel V2."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

from bestshot.sampling.temporal_sampler import (
    DecodedVideoFrame,
    GrayscaleImage,
    TemporalSamplingError,
)


class PyAVTemporalSamplingBackend:
    """Décode séquentiellement avec PyAV et ne convertit que les frames analysées."""

    def decode(self, video_path: Path) -> Iterator[DecodedVideoFrame]:
        """Produit des frames brutes sans créer de buffer RGB ou Pillow."""
        try:
            import av
        except ImportError as error:
            raise TemporalSamplingError("PyAV n'est pas installé.") from error

        try:
            with av.open(str(video_path)) as container:
                if not container.streams.video:
                    raise TemporalSamplingError("Aucun flux vidéo n'a été trouvé.")
                stream = container.streams.video[0]
                for frame_index, frame in enumerate(container.decode(stream)):
                    yield DecodedVideoFrame(
                        timestamp=float(frame.time) if frame.time is not None else None,
                        frame_index=frame_index,
                        source_width=frame.width,
                        source_height=frame.height,
                        payload=frame,
                    )
        except TemporalSamplingError:
            raise
        except Exception as error:
            raise TemporalSamplingError(f"Impossible de décoder la vidéo : {error}") from error

    def to_grayscale(self, frame: DecodedVideoFrame, max_width: int) -> GrayscaleImage:
        """Réduit directement la frame ciblée en gris via PyAV, sans Pillow."""
        if max_width <= 0:
            raise TemporalSamplingError("La largeur maximale d'analyse doit être positive.")
        try:
            raw_frame = cast(Any, frame.payload)
            target_width = min(frame.source_width, max_width)
            target_height = max(1, round(frame.source_height * target_width / frame.source_width))
            grayscale_frame = raw_frame.reformat(
                width=target_width,
                height=target_height,
                format="gray",
            )
            pixels = grayscale_frame.to_ndarray()
            return GrayscaleImage(
                width=target_width,
                height=target_height,
                gray_bytes=pixels.tobytes(),
            )
        except TemporalSamplingError:
            raise
        except Exception as error:
            raise TemporalSamplingError(f"Impossible de préparer une frame d'analyse : {error}") from error
