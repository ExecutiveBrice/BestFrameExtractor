"""Lecture typée de la configuration YAML locale."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import yaml  # type: ignore[import-untyped]

from bestshot.video.candidate_extractor import CandidateExtractionSettings
from bestshot.video.scene_detector import SceneDetectorSettings

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "default.yaml"


class ConfigurationError(ValueError):
    """La configuration YAML est absente ou invalide."""


def load_scene_detector_settings(config_path: Path = DEFAULT_CONFIG_PATH) -> SceneDetectorSettings:
    """Charge les paramètres de détection depuis ``config/default.yaml``."""
    root = _load_root(config_path)
    scene_detection = _mapping(root.get("scene_detection"))
    if not scene_detection:
        raise ConfigurationError("Section scene_detection absente de la configuration.")

    return SceneDetectorSettings(
        adaptive_threshold=_positive_float(scene_detection, "adaptive_threshold"),
        min_scene_len_frames=_positive_int(scene_detection, "min_scene_len_frames"),
        window_width=_positive_int(scene_detection, "window_width"),
        min_content_val=_positive_float(scene_detection, "min_content_val"),
    )


def load_candidate_extraction_settings(
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> CandidateExtractionSettings:
    """Charge les paramètres de génération des candidates depuis la configuration YAML."""
    root = _load_root(config_path)
    candidate_extraction = _mapping(root.get("candidate_extraction"))
    if not candidate_extraction:
        raise ConfigurationError("Section candidate_extraction absente de la configuration.")

    return CandidateExtractionSettings(
        fps=_positive_float(candidate_extraction, "fps"),
        analysis_max_width=_positive_int(candidate_extraction, "analysis_max_width"),
    )


def _load_root(config_path: Path) -> Mapping[str, object]:
    """Lit un document YAML local et retourne sa racine sous forme de mapping."""
    try:
        with config_path.open(encoding="utf-8") as config_file:
            document = yaml.safe_load(config_file)
    except OSError as error:
        raise ConfigurationError(f"Impossible de lire la configuration : {config_path}") from error
    return _mapping(document)


def _mapping(value: object) -> Mapping[str, object]:
    return cast(Mapping[str, object], value) if isinstance(value, Mapping) else {}


def _positive_float(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if isinstance(value, (int, float)) and float(value) > 0:
        return float(value)
    raise ConfigurationError(f"Valeur positive attendue pour scene_detection.{key}.")


def _positive_int(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if isinstance(value, int) and value > 0:
        return value
    raise ConfigurationError(f"Entier positif attendu pour scene_detection.{key}.")
