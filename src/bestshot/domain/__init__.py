"""Modèles et règles métier indépendants des détails techniques."""

from bestshot.domain.candidate_frame import CandidateFrame, PreviewImage
from bestshot.domain.scene import Scene
from bestshot.domain.video_info import VideoInfo

__all__ = ["CandidateFrame", "PreviewImage", "Scene", "VideoInfo"]
