"""Tests du score de groupe de visages sans modèle ni identité réelle."""

from collections.abc import Sequence
from pathlib import Path

from bestshot.domain.candidate_frame import PreviewImage
from bestshot.domain.face_analysis import FaceAnalysis
from bestshot.scoring.face import FaceScorer, FaceScoringSettings


class FakeFaceLandmarkerBackend:
    def __init__(self, analyses: Sequence[FaceAnalysis]) -> None:
        self.analyses = analyses

    def detect(self, preview: PreviewImage) -> Sequence[FaceAnalysis]:
        return self.analyses


def _settings() -> FaceScoringSettings:
    return FaceScoringSettings(
        model_path=Path("models/face_landmarker.task"),
        max_faces=10,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        yaw_scale_degrees=45.0,
        face_cut_off_margin=0.02,
        size_min_relative_area=0.02,
        size_good_relative_area=0.12,
        max_yaw_degrees=35.0,
        sharpness_min_variance=15.0,
        sharpness_good_variance=200.0,
        detection_confidence_weight=1.0,
        size_weight=1.0,
        orientation_weight=1.0,
        eyes_open_weight=1.5,
        positive_expression_weight=0.5,
        sharpness_weight=1.5,
        crop_weight=1.5,
    )


def _preview() -> PreviewImage:
    return PreviewImage(width=1, height=1, rgb_bytes=bytes((128, 128, 128)))


def test_no_face_is_neutral_and_not_an_error() -> None:
    score = FaceScorer(FakeFaceLandmarkerBackend(()), _settings()).score(_preview())

    assert score.analyses == ()
    assert score.global_score is None
    assert score.eyes_open is None


def test_group_score_penalizes_a_blurry_closed_turned_and_cut_face() -> None:
    good_face = FaceAnalysis(0.9, 0.15, 0.0, 1.0, 0.8, 250.0, False)
    poor_face = FaceAnalysis(0.9, 0.15, 45.0, 0.0, 0.8, 0.0, True)
    scorer = FaceScorer(FakeFaceLandmarkerBackend((good_face, poor_face)), _settings())

    score = scorer.score(_preview())

    assert score.orientation == 0.0
    assert score.eyes_open == 0.0
    assert score.sharpness == 0.0
    assert score.crop == 0.0
    assert score.global_score is not None and score.global_score < 0.5


def test_face_score_is_normalized_for_a_favorable_face() -> None:
    face = FaceAnalysis(0.9, 0.15, 0.0, 1.0, 0.8, 250.0, False)

    score = FaceScorer(FakeFaceLandmarkerBackend((face,)), _settings()).score(_preview())

    assert score.global_score is not None
    assert all(
        0.0 <= value <= 1.0
        for value in (
            score.detection_confidence,
            score.size,
            score.orientation,
            score.eyes_open,
            score.positive_expression,
            score.sharpness,
            score.crop,
            score.global_score,
        )
        if value is not None
    )
