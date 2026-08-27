"""Modèles de domaine partagés par le pipeline V2."""

from bestshot.domain.preferences import PairwisePreference, PreferenceChoice
from bestshot.domain.preview_image import PreviewImage
from bestshot.domain.video_info import VideoInfo

__all__ = [
    "PairwisePreference",
    "PreferenceChoice",
    "PreviewImage",
    "VideoInfo",
]
