"""Adaptateurs locaux nécessaires au pipeline V2."""

from bestshot.infrastructure.config import (
    DEFAULT_CONFIG_PATH,
    load_dataset_settings,
    load_embedding_settings,
    load_presampling_settings,
)
from bestshot.infrastructure.embedding_frames import PyAVCandidatePreviewReader
from bestshot.infrastructure.ffprobe import SubprocessFFprobeRunner
from bestshot.infrastructure.temporal_sampling import PyAVTemporalSamplingBackend

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "PyAVCandidatePreviewReader",
    "PyAVTemporalSamplingBackend",
    "SubprocessFFprobeRunner",
    "load_dataset_settings",
    "load_embedding_settings",
    "load_presampling_settings",
]
