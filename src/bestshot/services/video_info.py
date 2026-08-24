"""Cas d'usage et représentation textuelle de l'inspection vidéo."""

from pathlib import Path

from bestshot.domain.video_info import VideoInfo
from bestshot.video.probe import VideoProbe


def get_video_info(video_path: Path, probe: VideoProbe) -> VideoInfo:
    """Retourne les métadonnées d'une vidéo via le port fourni."""
    return probe.inspect(video_path)


def format_video_info(info: VideoInfo) -> str:
    """Produit une représentation stable pour les adaptateurs texte."""
    values = (
        ("Fichier", str(info.path)),
        ("Codec", info.codec),
        ("Dimensions", f"{info.width}x{info.height}"),
        ("FPS", f"{info.fps:.3f}"),
        ("Frames approximatives", _format_optional(info.approximate_frame_count)),
        ("Durée (secondes)", _format_optional(info.duration_seconds)),
        ("Bitrate (bps)", _format_optional(info.bitrate)),
        ("Orientation", f"{info.orientation_degrees}°"),
        ("Date de création", _format_optional(info.creation_time)),
    )
    return "\n".join(f"{label}: {value}" for label, value in values)


def _format_optional(value: object | None) -> str:
    return "indisponible" if value is None else str(value)
