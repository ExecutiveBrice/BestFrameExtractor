"""Lecture typée de la configuration YAML locale."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import yaml  # type: ignore[import-untyped]

from bestshot.plugins.aesthetic import AestheticModelSettings
from bestshot.scoring.composite import CompositeScoringSettings, CompositeWeights
from bestshot.scoring.face import FaceScoringSettings
from bestshot.scoring.technical import TechnicalScoringSettings
from bestshot.selection.deduplicate import DeduplicationSettings
from bestshot.selection.exporter import ExportSettings
from bestshot.selection.selector import SelectionSettings
from bestshot.video.candidate_extractor import CandidateExtractionSettings
from bestshot.video.candidate_refiner import RefinementSettings
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


def load_technical_scoring_settings(
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> TechnicalScoringSettings:
    """Charge les seuils et poids du score technique depuis la configuration YAML."""
    root = _load_root(config_path)
    scoring = _mapping(root.get("technical_scoring"))
    weights = _mapping(scoring.get("weights"))
    if not scoring or not weights:
        raise ConfigurationError("Section technical_scoring ou ses poids absents.")
    settings = TechnicalScoringSettings(
        sharpness_min_variance=_positive_float(scoring, "sharpness_min_variance"),
        sharpness_good_variance=_positive_float(scoring, "sharpness_good_variance"),
        exposure_target=_unit_interval(scoring, "exposure_target"),
        exposure_max_deviation=_positive_float(scoring, "exposure_max_deviation"),
        burned_pixel_threshold=_unit_interval(scoring, "burned_pixel_threshold"),
        burned_pixels_max_fraction=_positive_float(scoring, "burned_pixels_max_fraction"),
        underexposed_pixel_threshold=_unit_interval(scoring, "underexposed_pixel_threshold"),
        underexposed_pixels_max_fraction=_positive_float(
            scoring, "underexposed_pixels_max_fraction"
        ),
        contrast_min_stddev=_positive_float(scoring, "contrast_min_stddev"),
        contrast_good_stddev=_positive_float(scoring, "contrast_good_stddev"),
        motion_blur_min_gradient=_positive_float(scoring, "motion_blur_min_gradient"),
        motion_blur_max_anisotropy=_positive_float(scoring, "motion_blur_max_anisotropy"),
        sharpness_weight=_positive_float(weights, "sharpness"),
        exposure_weight=_positive_float(weights, "exposure"),
        burned_pixels_weight=_positive_float(weights, "burned_pixels"),
        underexposed_pixels_weight=_positive_float(weights, "underexposed_pixels"),
        contrast_weight=_positive_float(weights, "contrast"),
        motion_blur_weight=_positive_float(weights, "motion_blur"),
    )
    if settings.sharpness_good_variance <= settings.sharpness_min_variance:
        raise ConfigurationError("sharpness_good_variance doit dépasser sharpness_min_variance.")
    if settings.contrast_good_stddev <= settings.contrast_min_stddev:
        raise ConfigurationError("contrast_good_stddev doit dépasser contrast_min_stddev.")
    return settings


def load_face_scoring_settings(config_path: Path = DEFAULT_CONFIG_PATH) -> FaceScoringSettings:
    """Charge les paramètres locaux de Face Landmarker et de son score de groupe."""
    root = _load_root(config_path)
    scoring = _mapping(root.get("face_scoring"))
    weights = _mapping(scoring.get("weights"))
    if not scoring or not weights:
        raise ConfigurationError("Section face_scoring ou ses poids absents.")
    model_path_value = scoring.get("model_path")
    if not isinstance(model_path_value, str) or not model_path_value:
        raise ConfigurationError("Chemin de modèle attendu pour face_scoring.model_path.")
    settings = FaceScoringSettings(
        model_path=Path(model_path_value),
        max_faces=_positive_int(scoring, "max_faces"),
        min_face_detection_confidence=_unit_interval(scoring, "min_face_detection_confidence"),
        min_face_presence_confidence=_unit_interval(scoring, "min_face_presence_confidence"),
        min_tracking_confidence=_unit_interval(scoring, "min_tracking_confidence"),
        yaw_scale_degrees=_positive_float(scoring, "yaw_scale_degrees"),
        face_cut_off_margin=_unit_interval(scoring, "face_cut_off_margin"),
        size_min_relative_area=_positive_float(scoring, "size_min_relative_area"),
        size_good_relative_area=_positive_float(scoring, "size_good_relative_area"),
        max_yaw_degrees=_positive_float(scoring, "max_yaw_degrees"),
        sharpness_min_variance=_positive_float(scoring, "sharpness_min_variance"),
        sharpness_good_variance=_positive_float(scoring, "sharpness_good_variance"),
        detection_confidence_weight=_positive_float(weights, "detection_confidence"),
        size_weight=_positive_float(weights, "size"),
        orientation_weight=_positive_float(weights, "orientation"),
        eyes_open_weight=_positive_float(weights, "eyes_open"),
        positive_expression_weight=_positive_float(weights, "positive_expression"),
        sharpness_weight=_positive_float(weights, "sharpness"),
        crop_weight=_positive_float(weights, "crop"),
    )
    if settings.size_good_relative_area <= settings.size_min_relative_area:
        raise ConfigurationError("size_good_relative_area doit dépasser size_min_relative_area.")
    if settings.sharpness_good_variance <= settings.sharpness_min_variance:
        raise ConfigurationError("sharpness_good_variance doit dépasser sharpness_min_variance.")
    return settings


def load_composite_scoring_settings(
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> CompositeScoringSettings:
    """Charge les profils de pondération du score composite depuis le YAML."""
    root = _load_root(config_path)
    scoring = _mapping(root.get("composite_scoring"))
    people = _mapping(scoring.get("people"))
    no_people = _mapping(scoring.get("no_people"))
    if not scoring or not people or not no_people:
        raise ConfigurationError("Section composite_scoring, people ou no_people absente.")
    return CompositeScoringSettings(
        people=CompositeWeights(
            technical=_non_negative_float(people, "technical"),
            face=_non_negative_float(people, "face"),
            aesthetic=_non_negative_float(people, "aesthetic"),
            composition=_non_negative_float(people, "composition"),
        ),
        no_people=CompositeWeights(
            technical=_non_negative_float(no_people, "technical"),
            face=0.0,
            aesthetic=_non_negative_float(no_people, "aesthetic"),
            composition=_non_negative_float(no_people, "composition"),
        ),
        neutral_score=_unit_interval(scoring, "neutral_score"),
    )


def load_refinement_settings(config_path: Path = DEFAULT_CONFIG_PATH) -> RefinementSettings:
    """Charge les paramètres du second passage de raffinement local."""
    root = _load_root(config_path)
    refinement = _mapping(root.get("refinement"))
    if not refinement:
        raise ConfigurationError("Section refinement absente de la configuration.")
    enabled = refinement.get("enabled")
    if not isinstance(enabled, bool):
        raise ConfigurationError("Booléen attendu pour refinement.enabled.")
    return RefinementSettings(
        enabled=enabled,
        window_ms=_positive_int(refinement, "window_ms"),
        candidates_per_scene=_positive_int(refinement, "candidates_per_scene"),
    )


def load_deduplication_settings(config_path: Path = DEFAULT_CONFIG_PATH) -> DeduplicationSettings:
    """Charge les règles locales de dédoublonnage perceptuel."""
    root = _load_root(config_path)
    deduplication = _mapping(root.get("deduplication"))
    if not deduplication:
        raise ConfigurationError("Section deduplication absente de la configuration.")
    return DeduplicationSettings(
        similarity_threshold=_unit_interval(deduplication, "similarity_threshold"),
        temporal_window_ms=_positive_int(deduplication, "temporal_window_ms"),
        hash_size=_positive_int(deduplication, "hash_size"),
    )


def load_selection_settings(config_path: Path = DEFAULT_CONFIG_PATH) -> SelectionSettings:
    """Charge les limites de qualité et de diversité de la sélection finale."""
    root = _load_root(config_path)
    selection = _mapping(root.get("selection"))
    if not selection:
        raise ConfigurationError("Section selection absente de la configuration.")
    return SelectionSettings(
        max_per_scene=_positive_int(selection, "max_per_scene"),
        minimum_score=_unit_interval(selection, "minimum_score"),
    )


def load_export_settings(config_path: Path = DEFAULT_CONFIG_PATH) -> ExportSettings:
    """Charge la qualité JPEG utilisée par FFmpeg pour l'export final."""
    root = _load_root(config_path)
    export = _mapping(root.get("export"))
    if not export:
        raise ConfigurationError("Section export absente de la configuration.")
    jpeg_quality = _positive_int(export, "jpeg_quality")
    if jpeg_quality > 31:
        raise ConfigurationError("export.jpeg_quality doit être comprise entre 1 et 31.")
    return ExportSettings(jpeg_quality)


def load_aesthetic_model_settings(config_path: Path = DEFAULT_CONFIG_PATH) -> AestheticModelSettings:
    root = _load_root(config_path)
    model = _mapping(root.get("aesthetic_model"))
    required = ("clip_repo_id", "predictor_repo_id", "predictor_filename", "cache_dir")
    if not model or any(not isinstance(model.get(key), str) or not model[key] for key in required):
        raise ConfigurationError("Section aesthetic_model invalide.")
    raw_min = _positive_float(model, "raw_score_min")
    raw_max = _positive_float(model, "raw_score_max")
    if raw_max <= raw_min:
        raise ConfigurationError("raw_score_max doit dépasser raw_score_min.")
    return AestheticModelSettings(
        clip_repo_id=str(model["clip_repo_id"]),
        predictor_repo_id=str(model["predictor_repo_id"]),
        predictor_filename=str(model["predictor_filename"]),
        cache_dir=Path(str(model["cache_dir"])),
        embedding_dimension=_positive_int(model, "embedding_dimension"),
        raw_score_min=raw_min,
        raw_score_max=raw_max,
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


def _non_negative_float(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if isinstance(value, (int, float)) and float(value) >= 0:
        return float(value)
    raise ConfigurationError(f"Valeur positive ou nulle attendue pour composite_scoring.{key}.")


def _unit_interval(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0:
        return float(value)
    raise ConfigurationError(f"Valeur entre 0 et 1 attendue pour technical_scoring.{key}.")
