from pathlib import Path

from bestshot.scoring.face import FaceScoringSettings, UnavailableFaceScorer, create_face_scorer


def test_factory_uses_neutral_fallback_when_model_is_missing() -> None:
    settings = FaceScoringSettings(
        model_path=Path("missing-face-landmarker.task"),
        max_faces=1,
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
        eyes_open_weight=1.0,
        positive_expression_weight=1.0,
        sharpness_weight=1.0,
        crop_weight=1.0,
    )

    scorer = create_face_scorer(settings)

    assert isinstance(scorer, UnavailableFaceScorer)
    assert "continue sans analyse" in scorer.reason
