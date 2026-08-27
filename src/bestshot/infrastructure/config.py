"""Lecture typée de la configuration strictement nécessaire au pipeline V2."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import yaml  # type: ignore[import-untyped]

from bestshot.dataset.repository import DatasetSettings
from bestshot.embedding.dinov2 import DINOv2Settings
from bestshot.learning.pair_generator import PairGenerationSettings
from bestshot.learning.ranking_trainer import PersonalRankingSettings
from bestshot.sampling.temporal_sampler import PresamplingSettings
from bestshot.services.photo_pool import PhotoPoolSettings
from bestshot.services.selection import SelectionSettings

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "default.yaml"


class ConfigurationError(ValueError):
    """La configuration YAML est absente ou invalide."""


def load_presampling_settings(config_path: Path = DEFAULT_CONFIG_PATH) -> PresamplingSettings:
    """Charge les paramètres du pipeline V2 de présélection temporelle."""
    root = _load_root(config_path)
    personal_pipeline = _mapping(root.get("personal_pipeline"))
    presampling = _mapping(personal_pipeline.get("presampling"))
    if not presampling:
        raise ConfigurationError("Section personal_pipeline.presampling absente de la configuration.")

    return PresamplingSettings(
        analysis_fps=_positive_float(presampling, "analysis_fps"),
        bucket_seconds=_positive_float(presampling, "bucket_seconds"),
        keep_per_bucket=_positive_int(presampling, "keep_per_bucket"),
        analysis_max_width=_positive_int(presampling, "analysis_max_width"),
    )


def load_embedding_settings(config_path: Path = DEFAULT_CONFIG_PATH) -> DINOv2Settings:
    """Charge le premier provider DINOv2 et son cache d'embeddings local."""
    root = _load_root(config_path)
    model = _mapping(root.get("embedding_model"))
    required = ("repo_id", "revision", "model_version", "model_cache_dir", "embedding_cache_dir")
    if not model or any(not isinstance(model.get(key), str) or not model[key] for key in required):
        raise ConfigurationError("Section embedding_model invalide.")
    token_env_value = model.get("huggingface_token_env")
    if token_env_value is not None and (not isinstance(token_env_value, str) or not token_env_value):
        raise ConfigurationError("embedding_model.huggingface_token_env doit être une chaîne non vide.")
    return DINOv2Settings(
        huggingface_token_env=token_env_value,
        repo_id=str(model["repo_id"]),
        revision=str(model["revision"]),
        model_version=str(model["model_version"]),
        model_cache_dir=Path(str(model["model_cache_dir"])),
        embedding_cache_dir=Path(str(model["embedding_cache_dir"])),
    )


def load_dataset_settings(config_path: Path = DEFAULT_CONFIG_PATH) -> DatasetSettings:
    """Charge les emplacements exclusivement locaux du dataset de préférences."""
    root = _load_root(config_path)
    dataset = _mapping(root.get("dataset"))
    if not dataset:
        raise ConfigurationError("Section dataset absente de la configuration.")
    return DatasetSettings(
        database_path=_path(dataset, "database_path"),
        preview_cache_dir=_path(dataset, "preview_cache_dir"),
    )


def load_pair_generation_settings(config_path: Path = DEFAULT_CONFIG_PATH) -> PairGenerationSettings:
    """Charge les règles locales de proposition des comparaisons pairwise."""
    root = _load_root(config_path)
    values = _mapping(root.get("pair_generation"))
    if not values:
        raise ConfigurationError("Section pair_generation absente de la configuration.")
    return PairGenerationSettings(
        temporal_window_seconds=_positive_float(values, "temporal_window_seconds"),
        max_pairs_per_group=_positive_int(values, "max_pairs_per_group"),
        seed=_non_negative_int(values, "seed"),
        photo_pool_coverage_segment_count=_positive_int(values, "photo_pool_coverage_segment_count"),
        photo_pool_maximum_cosine_similarity=_ratio_inclusive(
            values, "photo_pool_maximum_cosine_similarity"
        ),
        photo_pool_minimum_frame_gap=_non_negative_int(values, "photo_pool_minimum_frame_gap"),
    )


def load_personal_ranking_settings(config_path: Path = DEFAULT_CONFIG_PATH) -> PersonalRankingSettings:
    """Charge les hyperparamètres du seul head de ranking entraînable."""
    root = _load_root(config_path)
    values = _mapping(root.get("personal_ranking"))
    if not values:
        raise ConfigurationError("Section personal_ranking absente de la configuration.")
    model_type = values.get("model_type")
    if model_type != "linear":
        raise ConfigurationError("personal_ranking.model_type doit être 'linear'.")
    return PersonalRankingSettings(
        equal_loss_weight=_non_negative_float(values, "equal_loss_weight"),
        learning_rate=_positive_float(values, "learning_rate"),
        epochs=_positive_int(values, "epochs"),
        weight_decay=_non_negative_float(values, "weight_decay"),
        validation_ratio=_ratio(values, "validation_ratio"),
        early_stopping_patience=_positive_int(values, "early_stopping_patience"),
        equality_margin=_non_negative_float(values, "equality_margin"),
        seed=_non_negative_int(values, "seed"),
    )


def load_selection_settings(config_path: Path = DEFAULT_CONFIG_PATH) -> SelectionSettings:
    """Charge la déduplication locale appliquée seulement à l'export final."""
    root = _load_root(config_path)
    values = _mapping(root.get("personal_selection"))
    if not values:
        raise ConfigurationError("Section personal_selection absente de la configuration.")
    maximum_cosine_similarity = values.get("maximum_cosine_similarity")
    if (
        not isinstance(maximum_cosine_similarity, (int, float))
        or not -1.0 <= float(maximum_cosine_similarity) <= 1.0
    ):
        raise ConfigurationError(
            "personal_selection.maximum_cosine_similarity doit être compris entre -1 et 1."
        )
    return SelectionSettings(
        minimum_time_separation_seconds=_non_negative_float(
            values, "minimum_time_separation_seconds"
        ),
        maximum_cosine_similarity=float(maximum_cosine_similarity),
    )


def load_photo_pool_settings(config_path: Path = DEFAULT_CONFIG_PATH) -> PhotoPoolSettings:
    """Charge la limite d'aperçu du corpus photo d'apprentissage."""
    root = _load_root(config_path)
    values = _mapping(root.get("personal_learning"))
    if not values:
        raise ConfigurationError("Section personal_learning absente de la configuration.")
    return PhotoPoolSettings(preview_max_width=_positive_int(values, "photo_pool_preview_max_width"))


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
    raise ConfigurationError(f"Valeur positive attendue pour {key}.")


def _positive_int(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if isinstance(value, int) and value > 0:
        return value
    raise ConfigurationError(f"Entier positif attendu pour {key}.")


def _non_negative_int(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if isinstance(value, int) and value >= 0:
        return value
    raise ConfigurationError(f"Entier positif ou nul attendu pour {key}.")


def _non_negative_float(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if isinstance(value, (int, float)) and float(value) >= 0:
        return float(value)
    raise ConfigurationError(f"Valeur positive ou nulle attendue pour {key}.")


def _ratio(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if isinstance(value, (int, float)) and 0 <= float(value) < 1:
        return float(value)
    raise ConfigurationError(f"Ratio [0, 1[ attendu pour {key}.")


def _ratio_inclusive(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if isinstance(value, (int, float)) and -1 <= float(value) <= 1:
        return float(value)
    raise ConfigurationError(f"Valeur [-1, 1] attendue pour {key}.")


def _path(values: Mapping[str, object], key: str) -> Path:
    value = values.get(key)
    if isinstance(value, str) and value:
        return Path(value)
    raise ConfigurationError(f"Chemin attendu pour {key}.")
