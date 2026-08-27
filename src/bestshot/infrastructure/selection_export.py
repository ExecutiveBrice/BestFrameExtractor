"""Export local haute définition des frames sélectionnées par le modèle personnel."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol
from uuid import uuid4


class ExportableFrame(Protocol):
    """Information minimale nécessaire à l'export d'une frame vidéo."""

    @property
    def timestamp(self) -> float: ...

    @property
    def frame_index(self) -> int: ...


class SelectedFrameExportError(RuntimeError):
    """Une frame sélectionnée n'a pas pu être extraite de sa vidéo locale."""


class PyAVSelectedFrameExporter:
    """Relit séquentiellement une vidéo et écrit uniquement les frames demandées."""

    def export(
        self,
        video_path: Path,
        frames: Sequence[ExportableFrame],
        destination_directory: Path,
    ) -> tuple[Path, ...]:
        if not frames:
            return ()
        requested = {frame.frame_index: frame for frame in frames}
        if len(requested) != len(frames):
            raise SelectedFrameExportError("Les frames à exporter doivent avoir des index uniques.")
        try:
            import av
        except ImportError as error:
            raise SelectedFrameExportError("PyAV est requis pour exporter les photos sélectionnées.") from error
        try:
            destination_directory.mkdir(parents=True, exist_ok=True)
            exported: dict[int, Path] = {}
            with av.open(str(video_path)) as container:
                if not container.streams.video:
                    raise SelectedFrameExportError("Aucun flux vidéo n'a été trouvé.")
                stream = container.streams.video[0]
                for frame_index, decoded in enumerate(container.decode(stream)):
                    record = requested.get(frame_index)
                    if record is None:
                        continue
                    destination = selected_frame_path(destination_directory, video_path, record)
                    _save_jpeg(decoded, destination)
                    exported[frame_index] = destination
                    if len(exported) == len(requested):
                        break
        except SelectedFrameExportError:
            raise
        except Exception as error:
            raise SelectedFrameExportError(f"Impossible d'exporter les frames de {video_path} : {error}") from error
        missing = sorted(set(requested) - set(exported))
        if missing:
            formatted = ", ".join(str(index) for index in missing)
            raise SelectedFrameExportError(f"Frames sélectionnées introuvables dans la vidéo : {formatted}")
        return tuple(exported[frame.frame_index] for frame in frames)


def selected_frame_path(destination_directory: Path, video_path: Path, frame: ExportableFrame) -> Path:
    """Construit un nom stable et lisible sans écraser une autre vidéo du dossier."""
    timestamp_milliseconds = round(frame.timestamp * 1_000)
    source_format = video_path.suffix.lower().removeprefix(".") or "video"
    return destination_directory / (
        f"{video_path.stem}--{source_format}--frame-{frame.frame_index:08d}--"
        f"{timestamp_milliseconds:010d}ms.jpg"
    )


def _save_jpeg(decoded: object, destination: Path) -> None:
    """Écrit atomiquement une image JPEG pleine résolution, hors de la base SQLite."""
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        image = decoded.to_image()  # type: ignore[attr-defined]
        image.save(temporary, format="JPEG", quality=95, optimize=True)
        os.replace(temporary, destination)
    except Exception as error:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise SelectedFrameExportError(f"Impossible d'écrire la photo sélectionnée : {destination}") from error
