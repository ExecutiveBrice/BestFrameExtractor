"""Interprétation des métadonnées ffprobe sans dépendance à subprocess."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Protocol, cast

from bestshot.domain.video_info import VideoInfo


class FFprobeRunner(Protocol):
    """Port vers un exécuteur qui retourne le JSON de ffprobe."""

    def probe(self, video_path: Path) -> Mapping[str, object]:
        """Retourne les métadonnées brutes de la première piste vidéo."""


class VideoProbeError(ValueError):
    """La réponse ffprobe ne décrit pas une piste vidéo exploitable."""


class VideoProbe:
    """Construit un ``VideoInfo`` depuis la réponse d'un ``FFprobeRunner``."""

    def __init__(self, runner: FFprobeRunner) -> None:
        self._runner = runner

    def inspect(self, video_path: Path) -> VideoInfo:
        """Lit les métadonnées de ``video_path`` sans décoder la vidéo."""
        payload = self._runner.probe(video_path)
        stream = _first_video_stream(payload)
        video_format = _mapping(payload.get("format"))

        fps = _frame_rate(stream.get("avg_frame_rate"))
        duration_seconds = _float_or_none(video_format.get("duration"))
        frame_count = _frame_count(stream.get("nb_frames"), duration_seconds, fps)
        bitrate = _int_or_none(stream.get("bit_rate")) or _int_or_none(video_format.get("bit_rate"))

        return VideoInfo(
            path=video_path,
            codec=_required_text(stream, "codec_name"),
            width=_required_positive_int(stream, "width"),
            height=_required_positive_int(stream, "height"),
            fps=fps,
            approximate_frame_count=frame_count,
            duration_seconds=duration_seconds,
            bitrate=bitrate,
            orientation_degrees=_orientation(stream),
            creation_time=_creation_time(video_format),
        )


def _first_video_stream(payload: Mapping[str, object]) -> Mapping[str, object]:
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise VideoProbeError("ffprobe n'a retourné aucune liste de flux.")

    for stream in streams:
        mapping = _mapping(stream)
        if mapping.get("codec_type") == "video":
            return mapping
    raise VideoProbeError("Aucun flux vidéo n'a été trouvé.")


def _mapping(value: object) -> Mapping[str, object]:
    return cast(Mapping[str, object], value) if isinstance(value, Mapping) else {}


def _required_text(data: Mapping[str, object], field: str) -> str:
    value = data.get(field)
    if isinstance(value, str) and value:
        return value
    raise VideoProbeError(f"Champ ffprobe obligatoire absent : {field}.")


def _required_positive_int(data: Mapping[str, object], field: str) -> int:
    value = _int_or_none(data.get(field))
    if value is not None and value > 0:
        return value
    raise VideoProbeError(f"Champ ffprobe obligatoire invalide : {field}.")


def _int_or_none(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _float_or_none(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _frame_rate(value: object) -> float:
    if not isinstance(value, str):
        raise VideoProbeError("Champ ffprobe obligatoire absent : avg_frame_rate.")
    numerator, separator, denominator = value.partition("/")
    if not separator:
        raise VideoProbeError("Format de fréquence d'images invalide.")
    try:
        rate = int(numerator) / int(denominator)
    except (ValueError, ZeroDivisionError) as error:
        raise VideoProbeError("Format de fréquence d'images invalide.") from error
    if rate <= 0:
        raise VideoProbeError("La fréquence d'images doit être positive.")
    return rate


def _frame_count(value: object, duration_seconds: float | None, fps: float) -> int | None:
    exact_count = _int_or_none(value)
    if exact_count is not None:
        return exact_count
    if duration_seconds is None:
        return None
    return round(duration_seconds * fps)


def _orientation(stream: Mapping[str, object]) -> int:
    side_data = stream.get("side_data_list")
    if isinstance(side_data, list):
        for item in side_data:
            rotation = _int_or_none(_mapping(item).get("rotation"))
            if rotation is not None:
                return rotation % 360
    tags = _mapping(stream.get("tags"))
    return (_int_or_none(tags.get("rotate")) or 0) % 360


def _creation_time(video_format: Mapping[str, object]) -> datetime | None:
    tags = _mapping(video_format.get("tags"))
    value = tags.get("creation_time")
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
