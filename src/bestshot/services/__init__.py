"""Cas d'usage orchestrant le domaine et l'infrastructure."""

from bestshot.services.candidates import extract_candidates, format_candidate_counts
from bestshot.services.scenes import detect_scenes, format_scenes
from bestshot.services.video_info import format_video_info, get_video_info

__all__ = [
    "detect_scenes",
    "extract_candidates",
    "format_candidate_counts",
    "format_scenes",
    "format_video_info",
    "get_video_info",
]
