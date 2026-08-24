"""Scoring technique d'aperçus RGB avec OpenCV et NumPy."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from bestshot.domain.candidate_frame import PreviewImage
from bestshot.domain.technical_score import TechnicalScore


@dataclass(frozen=True, slots=True)
class TechnicalScoringSettings:
    """Seuils et poids utilisés pour normaliser les mesures techniques."""

    sharpness_min_variance: float
    sharpness_good_variance: float
    exposure_target: float
    exposure_max_deviation: float
    burned_pixel_threshold: float
    burned_pixels_max_fraction: float
    underexposed_pixel_threshold: float
    underexposed_pixels_max_fraction: float
    contrast_min_stddev: float
    contrast_good_stddev: float
    motion_blur_min_gradient: float
    motion_blur_max_anisotropy: float
    sharpness_weight: float
    exposure_weight: float
    burned_pixels_weight: float
    underexposed_pixels_weight: float
    contrast_weight: float
    motion_blur_weight: float


class TechnicalScoringError(ValueError):
    """L'aperçu ou la configuration ne permet pas un score technique."""


class TechnicalScorer:
    """Calcule indépendamment les scores techniques d'un aperçu réduit."""

    def __init__(self, settings: TechnicalScoringSettings) -> None:
        self._settings = settings

    def score(self, preview: PreviewImage) -> TechnicalScore:
        """Retourne les scores normalisés pour l'aperçu RGB fourni."""
        grayscale = _to_grayscale(preview)
        sharpness = _linear_score(
            float(cv2.Laplacian(grayscale, cv2.CV_64F).var()),
            self._settings.sharpness_min_variance,
            self._settings.sharpness_good_variance,
        )
        mean_luminance = float(grayscale.mean())
        exposure = _clamp(
            1.0
            - abs(mean_luminance - self._settings.exposure_target)
            / self._settings.exposure_max_deviation
        )
        burned_fraction = float(np.mean(grayscale >= self._settings.burned_pixel_threshold))
        burned_pixels = 1.0 - _clamp(
            burned_fraction / self._settings.burned_pixels_max_fraction
        )
        underexposed_fraction = float(
            np.mean(grayscale <= self._settings.underexposed_pixel_threshold)
        )
        underexposed_pixels = 1.0 - _clamp(
            underexposed_fraction / self._settings.underexposed_pixels_max_fraction
        )
        contrast = _linear_score(
            float(grayscale.std()),
            self._settings.contrast_min_stddev,
            self._settings.contrast_good_stddev,
        )
        motion_blur = _motion_blur_score(grayscale, self._settings)
        global_score = _weighted_average(
            (sharpness, self._settings.sharpness_weight),
            (exposure, self._settings.exposure_weight),
            (burned_pixels, self._settings.burned_pixels_weight),
            (underexposed_pixels, self._settings.underexposed_pixels_weight),
            (contrast, self._settings.contrast_weight),
            (motion_blur, self._settings.motion_blur_weight),
        )
        return TechnicalScore(
            sharpness=sharpness,
            exposure=exposure,
            burned_pixels=burned_pixels,
            underexposed_pixels=underexposed_pixels,
            contrast=contrast,
            motion_blur=motion_blur,
            global_score=global_score,
        )


def _to_grayscale(preview: PreviewImage) -> NDArray[np.float64]:
    expected_size = preview.width * preview.height * 3
    if preview.width <= 0 or preview.height <= 0 or len(preview.rgb_bytes) != expected_size:
        raise TechnicalScoringError("Les données de l'aperçu RGB sont invalides.")
    rgb = np.frombuffer(preview.rgb_bytes, dtype=np.uint8).reshape((preview.height, preview.width, 3))
    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return grayscale.astype(np.float64) / np.iinfo(np.uint8).max


def _motion_blur_score(
    grayscale: NDArray[np.float64], settings: TechnicalScoringSettings
) -> float:
    gradient_x = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0)
    gradient_y = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1)
    mean_x = float(np.abs(gradient_x).mean())
    mean_y = float(np.abs(gradient_y).mean())
    total_gradient = mean_x + mean_y
    if total_gradient <= settings.motion_blur_min_gradient:
        return 0.0
    anisotropy = abs(mean_x - mean_y) / total_gradient
    return 1.0 - _clamp(anisotropy / settings.motion_blur_max_anisotropy)


def _linear_score(value: float, lower_bound: float, upper_bound: float) -> float:
    return _clamp((value - lower_bound) / (upper_bound - lower_bound))


def _weighted_average(*values: tuple[float, float]) -> float:
    total_weight = sum(weight for _, weight in values)
    if total_weight <= 0:
        raise TechnicalScoringError("La somme des poids techniques doit être positive.")
    return sum(score * weight for score, weight in values) / total_weight


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))
