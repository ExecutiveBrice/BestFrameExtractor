"""Tests de lecture de la configuration V2."""

from pathlib import Path

from bestshot.infrastructure.config import (
    load_dataset_settings,
    load_embedding_settings,
    load_presampling_settings,
)


def test_load_presampling_settings_reads_v2_defaults() -> None:
    settings = load_presampling_settings()

    assert settings.analysis_fps == 8.0
    assert settings.bucket_seconds == 1.0
    assert settings.keep_per_bucket == 2
    assert settings.analysis_max_width == 640


def test_load_embedding_settings_reads_dinov2_defaults() -> None:
    settings = load_embedding_settings()

    assert settings.repo_id == "facebook/dinov2-small"
    assert settings.model_version == "dinov2-vits14-1"
    assert settings.embedding_cache_dir.name == "embeddings"


def test_load_dataset_settings_reads_local_defaults() -> None:
    settings = load_dataset_settings()

    assert settings.database_path == Path(".bestshot/dataset/bestshot.db")
    assert settings.preview_cache_dir == Path(".bestshot/dataset/previews")
