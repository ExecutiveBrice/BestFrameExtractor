"""Tests du score technique avec des aperçus RGB synthétiques."""

import cv2
import numpy as np

from bestshot.domain.candidate_frame import PreviewImage
from bestshot.scoring.technical import TechnicalScorer, TechnicalScoringSettings


def _settings() -> TechnicalScoringSettings:
    return TechnicalScoringSettings(
        sharpness_min_variance=0.001,
        sharpness_good_variance=0.05,
        exposure_target=0.5,
        exposure_max_deviation=0.25,
        burned_pixel_threshold=0.95,
        burned_pixels_max_fraction=0.05,
        underexposed_pixel_threshold=0.05,
        underexposed_pixels_max_fraction=0.05,
        contrast_min_stddev=0.01,
        contrast_good_stddev=0.20,
        motion_blur_min_gradient=0.001,
        motion_blur_max_anisotropy=0.8,
        sharpness_weight=1.0,
        exposure_weight=1.0,
        burned_pixels_weight=1.0,
        underexposed_pixels_weight=1.0,
        contrast_weight=1.0,
        motion_blur_weight=1.0,
    )


def _preview(image: np.ndarray) -> PreviewImage:
    height, width, _ = image.shape
    return PreviewImage(width=width, height=height, rgb_bytes=image.astype(np.uint8).tobytes())


def _checkerboard() -> np.ndarray:
    grid = np.indices((64, 64)).sum(axis=0) % 2 * 255
    return np.repeat(grid[:, :, np.newaxis].astype(np.uint8), 3, axis=2)


def test_sharp_image_scores_higher_than_blurred_image() -> None:
    sharp_image = _checkerboard()
    blurred_image = cv2.GaussianBlur(sharp_image, (9, 9), 0)
    scorer = TechnicalScorer(_settings())

    sharp_score = scorer.score(_preview(sharp_image))
    blurred_score = scorer.score(_preview(blurred_image))

    assert sharp_score.sharpness > blurred_score.sharpness


def test_dark_and_overexposed_images_are_penalized_independently() -> None:
    scorer = TechnicalScorer(_settings())
    dark = scorer.score(_preview(np.full((32, 32, 3), 5, dtype=np.uint8)))
    overexposed = scorer.score(_preview(np.full((32, 32, 3), 255, dtype=np.uint8)))
    normal = scorer.score(_preview(np.full((32, 32, 3), 128, dtype=np.uint8)))

    assert dark.underexposed_pixels < normal.underexposed_pixels
    assert dark.exposure < normal.exposure
    assert overexposed.burned_pixels < normal.burned_pixels
    assert overexposed.exposure < normal.exposure


def test_all_scores_are_normalized_for_synthetic_images() -> None:
    scorer = TechnicalScorer(_settings())
    images = [
        _checkerboard(),
        cv2.GaussianBlur(_checkerboard(), (9, 9), 0),
        np.full((32, 32, 3), 5, dtype=np.uint8),
        np.full((32, 32, 3), 255, dtype=np.uint8),
        np.full((32, 32, 3), 128, dtype=np.uint8),
    ]

    for image in images:
        score = scorer.score(_preview(image))
        assert all(
            0.0 <= value <= 1.0
            for value in (
                score.sharpness,
                score.exposure,
                score.burned_pixels,
                score.underexposed_pixels,
                score.contrast,
                score.motion_blur,
                score.global_score,
            )
        )
