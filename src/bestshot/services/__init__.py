"""Cas d'usage orchestrant le domaine et l'infrastructure."""

from bestshot.services.candidates import (
    extract_candidates,
    format_candidate_counts,
    format_candidate_repository_result,
    persist_candidate_previews,
)
from bestshot.services.scenes import detect_scenes, format_scenes
from bestshot.services.selection import format_selection_result, rank_candidates, select_best_frames
from bestshot.services.technical_analysis import format_technical_analysis
from bestshot.services.video_info import format_video_info, get_video_info

__all__ = [
    "detect_scenes",
    "extract_candidates",
    "format_candidate_counts",
    "format_candidate_repository_result",
    "format_scenes",
    "format_selection_result",
    "format_technical_analysis",
    "format_video_info",
    "get_video_info",
    "persist_candidate_previews",
    "rank_candidates",
    "select_best_frames",
]
