"""Tests de chargement de la configuration de détection de scènes."""

from pathlib import Path

import pytest

from bestshot.infrastructure.config import (
    ConfigurationError,
    load_candidate_extraction_settings,
    load_composite_scoring_settings,
    load_deduplication_settings,
    load_export_settings,
    load_face_scoring_settings,
    load_refinement_settings,
    load_scene_detector_settings,
    load_selection_settings,
    load_technical_scoring_settings,
)


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


def test_load_candidate_extraction_settings_reads_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(
        """candidate_extraction:
  fps: 4
  analysis_max_width: 720
  candidate_repository_dir: /tmp/candidates
""",
        encoding="utf-8",
    )

    settings = load_candidate_extraction_settings(config_path)

    assert settings.fps == 4.0
    assert settings.analysis_max_width == 720
    assert settings.candidate_repository_dir == Path("/tmp/candidates")


def test_load_technical_scoring_settings_reads_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(
        """technical_scoring:
  sharpness_min_variance: 10.0
  sharpness_good_variance: 100.0
  exposure_target: 0.5
  exposure_max_deviation: 0.3
  burned_pixel_threshold: 0.9
  burned_pixels_max_fraction: 0.1
  underexposed_pixel_threshold: 0.1
  underexposed_pixels_max_fraction: 0.1
  contrast_min_stddev: 0.02
  contrast_good_stddev: 0.2
  motion_blur_min_gradient: 0.01
  motion_blur_max_anisotropy: 0.5
  weights:
    sharpness: 1.0
    exposure: 1.0
    burned_pixels: 1.0
    underexposed_pixels: 1.0
    contrast: 1.0
    motion_blur: 1.0
""",
        encoding="utf-8",
    )

    settings = load_technical_scoring_settings(config_path)

    assert settings.sharpness_good_variance == 100.0
    assert settings.motion_blur_weight == 1.0


def test_load_face_scoring_settings_reads_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(
        """face_scoring:
  model_path: models/face_landmarker.task
  max_faces: 5
  min_face_detection_confidence: 0.5
  min_face_presence_confidence: 0.5
  min_tracking_confidence: 0.5
  yaw_scale_degrees: 45.0
  face_cut_off_margin: 0.02
  size_min_relative_area: 0.02
  size_good_relative_area: 0.12
  max_yaw_degrees: 35.0
  sharpness_min_variance: 15.0
  sharpness_good_variance: 200.0
  weights:
    detection_confidence: 1.0
    size: 1.0
    orientation: 1.0
    eyes_open: 1.0
    positive_expression: 1.0
    sharpness: 1.0
    crop: 1.0
""",
        encoding="utf-8",
    )

    settings = load_face_scoring_settings(config_path)

    assert settings.model_path == Path("models/face_landmarker.task")
    assert settings.max_faces == 5


def test_load_composite_scoring_settings_reads_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(
        """composite_scoring:
  neutral_score: 0.5
  people:
    technical: 0.3
    face: 0.4
    aesthetic: 0.2
    composition: 0.1
  no_people:
    technical: 0.4
    aesthetic: 0.35
    composition: 0.25
""",
        encoding="utf-8",
    )

    settings = load_composite_scoring_settings(config_path)

    assert settings.people.face == 0.4
    assert settings.no_people.face == 0.0


def test_load_refinement_settings_reads_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(
        """refinement:
  enabled: true
  window_ms: 500
  candidates_per_scene: 3
""",
        encoding="utf-8",
    )

    settings = load_refinement_settings(config_path)

    assert settings.enabled is True
    assert settings.window_ms == 500


def test_load_deduplication_settings_reads_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(
        """deduplication:
  similarity_threshold: 0.9
  temporal_window_ms: 1000
  hash_size: 8
""",
        encoding="utf-8",
    )

    settings = load_deduplication_settings(config_path)

    assert settings.similarity_threshold == 0.9
    assert settings.temporal_window_ms == 1_000


def test_load_selection_settings_reads_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(
        """selection:
  max_per_scene: 3
  minimum_score: 0.55
""",
        encoding="utf-8",
    )

    settings = load_selection_settings(config_path)

    assert settings.max_per_scene == 3
    assert settings.minimum_score == 0.55


def test_load_export_settings_reads_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text("export:\n  jpeg_quality: 2\n", encoding="utf-8")

    assert load_export_settings(config_path).jpeg_quality == 2


def test_load_aesthetic_model_settings_reads_huggingface_token_environment_name(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(
        """aesthetic_model:
  huggingface_token_env: HF_TOKEN
  clip_repo_id: openai/clip-vit-base-patch32
  model_repo_id: example/aesthetic-scorer
  model_filename: model.pt
  cache_dir: .bestshot/models/aesthetic
  raw_score_min: 0.0
  raw_score_max: 5.0
""",
        encoding="utf-8",
    )

    from bestshot.infrastructure.config import load_aesthetic_model_settings

    assert load_aesthetic_model_settings(config_path).huggingface_token_env == "HF_TOKEN"
