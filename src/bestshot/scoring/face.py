"""Évaluation qualitative et locale de visages avec MediaPipe Face Landmarker."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from bestshot.domain.candidate_frame import PreviewImage
from bestshot.domain.face_analysis import FaceAnalysis, FaceScore


@dataclass(frozen=True, slots=True)
class FaceScoringSettings:
    """Paramètres du détecteur local et poids de la qualité de groupe."""

    model_path: Path
    max_faces: int
    min_face_detection_confidence: float
    min_face_presence_confidence: float
    min_tracking_confidence: float
    yaw_scale_degrees: float
    face_cut_off_margin: float
    size_min_relative_area: float
    size_good_relative_area: float
    max_yaw_degrees: float
    sharpness_min_variance: float
    sharpness_good_variance: float
    detection_confidence_weight: float
    size_weight: float
    orientation_weight: float
    eyes_open_weight: float
    positive_expression_weight: float
    sharpness_weight: float
    crop_weight: float


class FaceLandmarkerBackend(Protocol):
    """Port vers un détecteur de visages sans identité ni reconnaissance."""

    def detect(self, preview: PreviewImage) -> Sequence[FaceAnalysis]:
        """Retourne les analyses anonymes de tous les visages présents."""


class FaceScoreProvider(Protocol):
    """Port de score visage utilisable même si le modèle optionnel est absent."""

    def score(self, preview: PreviewImage) -> FaceScore:
        """Retourne un score de groupe, potentiellement sans visage détecté."""


class FaceScoringError(RuntimeError):
    """Le modèle Face Landmarker ne peut pas analyser l'aperçu fourni."""


class MediaPipeFaceLandmarkerBackend:
    """Adaptateur MediaPipe Face Landmarker, exécuté entièrement localement."""

    def __init__(self, settings: FaceScoringSettings) -> None:
        self._settings = settings

    def detect(self, preview: PreviewImage) -> Sequence[FaceAnalysis]:
        """Détecte les visages et produit des métriques anonymes par visage."""
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as error:
            raise FaceScoringError("MediaPipe n'est pas installé.") from error
        if not self._settings.model_path.is_file():
            raise FaceScoringError(f"Modèle Face Landmarker introuvable : {self._settings.model_path}")

        rgb = _preview_to_rgb(preview)
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(self._settings.model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=self._settings.max_faces,
            min_face_detection_confidence=self._settings.min_face_detection_confidence,
            min_face_presence_confidence=self._settings.min_face_presence_confidence,
            min_tracking_confidence=self._settings.min_tracking_confidence,
            output_face_blendshapes=True,
        )
        try:
            with vision.FaceLandmarker.create_from_options(options) as landmarker:
                result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        except Exception as error:
            raise FaceScoringError(f"Échec de l'analyse MediaPipe : {error}") from error

        blendshapes = result.face_blendshapes or []
        return tuple(
            _analysis_from_landmarks(
                landmarks,
                blendshapes[index] if index < len(blendshapes) else (),
                preview,
                self._settings,
            )
            for index, landmarks in enumerate(result.face_landmarks)
        )


class FaceScorer:
    """Normalise les métriques anonymes et pénalise le visage le moins favorable."""

    def __init__(self, backend: FaceLandmarkerBackend, settings: FaceScoringSettings) -> None:
        self._backend = backend
        self._settings = settings

    def score(self, preview: PreviewImage) -> FaceScore:
        """Calcule un score de groupe ; l'absence de visage reste neutre."""
        analyses = tuple(self._backend.detect(preview))
        if not analyses:
            return FaceScore(analyses, None, None, None, None, None, None, None, None)

        detection_confidence = _minimum_available(
            analysis.detection_confidence for analysis in analyses
        )
        size = min(
            _linear_score(
                analysis.relative_size,
                self._settings.size_min_relative_area,
                self._settings.size_good_relative_area,
            )
            for analysis in analyses
        )
        orientation = _minimum_available(
            _clamp(1.0 - abs(analysis.yaw_degrees) / self._settings.max_yaw_degrees)
            if analysis.yaw_degrees is not None
            else None
            for analysis in analyses
        )
        eyes_open = _minimum_available(analysis.eyes_open for analysis in analyses)
        positive_expression = _minimum_available(
            analysis.positive_expression for analysis in analyses
        )
        sharpness = min(
            _linear_score(
                analysis.sharpness_variance,
                self._settings.sharpness_min_variance,
                self._settings.sharpness_good_variance,
            )
            for analysis in analyses
        )
        crop = min(0.0 if analysis.is_cut_off else 1.0 for analysis in analyses)
        global_score = _weighted_average(
            (detection_confidence, self._settings.detection_confidence_weight),
            (size, self._settings.size_weight),
            (orientation, self._settings.orientation_weight),
            (eyes_open, self._settings.eyes_open_weight),
            (positive_expression, self._settings.positive_expression_weight),
            (sharpness, self._settings.sharpness_weight),
            (crop, self._settings.crop_weight),
        )
        return FaceScore(
            analyses,
            detection_confidence,
            size,
            orientation,
            eyes_open,
            positive_expression,
            sharpness,
            crop,
            global_score,
        )


class UnavailableFaceScorer:
    """Fallback neutre lorsque le modèle local Face Landmarker est indisponible.

    L'absence d'un modèle optionnel ne doit pas empêcher les parcours de sélection
    et d'export ; le composite applique alors le profil ``no_people``.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def score(self, preview: PreviewImage) -> FaceScore:
        """Représente l'absence de détection sans pénaliser la candidate."""
        del preview
        return FaceScore((), None, None, None, None, None, None, None, None)


def create_face_scorer(settings: FaceScoringSettings) -> FaceScoreProvider:
    """Construit le scorer MediaPipe ou un fallback neutre sans accès réseau."""
    if not settings.model_path.is_file():
        return UnavailableFaceScorer(
            f"Modèle Face Landmarker introuvable : {settings.model_path}. "
            "La sélection continue sans analyse de visages."
        )
    return FaceScorer(MediaPipeFaceLandmarkerBackend(settings), settings)


def _analysis_from_landmarks(
    landmarks: Sequence[Any], blendshapes: Sequence[Any], preview: PreviewImage, settings: FaceScoringSettings
) -> FaceAnalysis:
    if not landmarks:
        raise FaceScoringError("Face Landmarker a retourné un visage sans repères.")
    x_values = [float(landmark.x) for landmark in landmarks]
    y_values = [float(landmark.y) for landmark in landmarks]
    min_x, max_x = max(0.0, min(x_values)), min(1.0, max(x_values))
    min_y, max_y = max(0.0, min(y_values)), min(1.0, max(y_values))
    width, height = max_x - min_x, max_y - min_y
    midpoint = min_x + width / 2.0
    nose_x = float(landmarks[1].x) if len(landmarks) > 1 else midpoint
    yaw_degrees = 0.0 if width == 0 else (nose_x - midpoint) / (width / 2.0) * settings.yaw_scale_degrees
    scores = {str(item.category_name): float(item.score) for item in blendshapes}
    eyes_open = _eyes_open(scores)
    positive_expression = _positive_expression(scores)
    return FaceAnalysis(
        detection_confidence=None,
        relative_size=width * height,
        yaw_degrees=yaw_degrees,
        eyes_open=eyes_open,
        positive_expression=positive_expression,
        sharpness_variance=_face_sharpness(preview, min_x, min_y, max_x, max_y),
        is_cut_off=(
            min_x <= settings.face_cut_off_margin
            or min_y <= settings.face_cut_off_margin
            or max_x >= 1.0 - settings.face_cut_off_margin
            or max_y >= 1.0 - settings.face_cut_off_margin
        ),
    )


def _preview_to_rgb(preview: PreviewImage) -> np.ndarray[Any, np.dtype[np.uint8]]:
    expected_size = preview.width * preview.height * 3
    if preview.width <= 0 or preview.height <= 0 or len(preview.rgb_bytes) != expected_size:
        raise FaceScoringError("Les données de l'aperçu RGB sont invalides.")
    return np.frombuffer(preview.rgb_bytes, dtype=np.uint8).reshape((preview.height, preview.width, 3))


def _face_sharpness(preview: PreviewImage, min_x: float, min_y: float, max_x: float, max_y: float) -> float:
    rgb = _preview_to_rgb(preview)
    left, right = round(min_x * preview.width), round(max_x * preview.width)
    top, bottom = round(min_y * preview.height), round(max_y * preview.height)
    crop = rgb[top:bottom, left:right]
    if crop.size == 0:
        return 0.0
    grayscale = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(grayscale, cv2.CV_64F).var())


def _eyes_open(scores: dict[str, float]) -> float | None:
    left, right = scores.get("eyeBlinkLeft"), scores.get("eyeBlinkRight")
    if left is None or right is None:
        return None
    return _clamp(1.0 - (left + right) / 2.0)


def _positive_expression(scores: dict[str, float]) -> float | None:
    left, right = scores.get("mouthSmileLeft"), scores.get("mouthSmileRight")
    if left is None or right is None:
        return None
    return _clamp((left + right) / 2.0)


def _linear_score(value: float, lower_bound: float, upper_bound: float) -> float:
    return _clamp((value - lower_bound) / (upper_bound - lower_bound))


def _minimum_available(values: Sequence[float | None] | Any) -> float | None:
    available = tuple(value for value in values if value is not None)
    return min(available) if available else None


def _weighted_average(*values: tuple[float | None, float]) -> float:
    available = tuple((score, weight) for score, weight in values if score is not None)
    total_weight = sum(weight for _, weight in available)
    if total_weight <= 0:
        raise FaceScoringError("La somme des poids de visage doit être positive.")
    return sum(score * weight for score, weight in available) / total_weight


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))
