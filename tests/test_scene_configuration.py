"""Tests de chargement de la configuration de détection de scènes."""

from pathlib import Path

import pytest

from bestshot.infrastructure.config import ConfigurationError, load_scene_detector_settings


def test_load_scene_detector_settings_reads_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(
        """scene_detection:
  adaptive_threshold: 4.5
  min_scene_len_frames: 20
  window_width: 3
  min_content_val: 12.0
""",
        encoding="utf-8",
    )

    settings = load_scene_detector_settings(config_path)

    assert settings.adaptive_threshold == 4.5
    assert settings.min_scene_len_frames == 20
    assert settings.window_width == 3
    assert settings.min_content_val == 12.0


def test_load_scene_detector_settings_rejects_missing_section(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text("video: {}\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="scene_detection"):
        load_scene_detector_settings(config_path)
