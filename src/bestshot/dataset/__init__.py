"""Dataset local des préférences personnelles, sans modèle d'apprentissage."""

from bestshot.dataset.labels import FrameLabel, is_training_label
from bestshot.dataset.preview_cache import PreviewCache
from bestshot.dataset.repository import (
    DatasetSettings,
    DatasetStats,
    FrameRecord,
    PreferenceStats,
    TrainingModel,
    VideoRecord,
)
from bestshot.dataset.sqlite_repository import SQLiteDatasetRepository

__all__ = [
    "DatasetSettings",
    "DatasetStats",
    "FrameLabel",
    "FrameRecord",
    "PreferenceStats",
    "PreviewCache",
    "SQLiteDatasetRepository",
    "TrainingModel",
    "VideoRecord",
    "is_training_label",
]
