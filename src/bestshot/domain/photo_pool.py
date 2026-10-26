"""Données et règles du pool photo, indépendantes des modèles et des E/S."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Literal

from bestshot.domain.candidate_frame import PreviewImage


@dataclass(frozen=True, slots=True)
class PhotoPoolSettings:
    fps: float
    analysis_width: int
    minimum_sharpness: float
    minimum_detail_ratio: float
    minimum_native_sharpness: float
    minimum_contrast: float
    minimum_luminance: float
    maximum_luminance: float
    minimum_content_similarity: float
    content_margin: float
    review_similarity: float
    review_margin: float
    jpeg_quality: int

    def __post_init__(self) -> None:
        positive = (
            self.fps,
            self.minimum_sharpness,
            self.minimum_detail_ratio,
            self.minimum_native_sharpness,
            self.minimum_contrast,
        )
        if any(not isfinite(v) or v <= 0 for v in positive) or self.analysis_width < 32:
            raise ValueError("Cadence, dimensions et seuils de qualité doivent être positifs.")
        if not 0 <= self.minimum_luminance < self.maximum_luminance <= 255:
            raise ValueError("La plage de luminance doit être comprise entre 0 et 255.")
        thresholds = (
            self.minimum_content_similarity,
            self.review_similarity,
            self.content_margin,
            self.review_margin,
        )
        if any(not isfinite(v) or not -1 <= v <= 1 for v in thresholds):
            raise ValueError("Les seuils CLIP doivent être compris entre -1 et 1.")
        if (
            self.review_similarity > self.minimum_content_similarity
            or self.review_margin > self.content_margin
        ):
            raise ValueError("Les seuils de vérification doivent être moins stricts que le pool.")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("La qualité JPEG du pool doit être comprise entre 1 et 100.")


@dataclass(frozen=True, slots=True)
class ClipContentSettings:
    repo_id: str
    cache_dir: Path
    revision: str
    people_prompts: tuple[str, ...]
    landscape_prompts: tuple[str, ...]
    reject_prompts: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.repo_id or not self.revision:
            raise ValueError("Le dépôt CLIP et sa révision doivent être renseignés.")
        if not all((self.people_prompts, self.landscape_prompts, self.reject_prompts)):
            raise ValueError("Chaque groupe de descriptions CLIP doit être renseigné.")


@dataclass(frozen=True, slots=True)
class PoolFrame:
    timestamp: float
    frame_index: int
    image: PreviewImage  # RGB natif transitoire ; jamais accumulé par le workflow.


@dataclass(frozen=True, slots=True)
class PoolQuality:
    sharpness: float  # Variance du Laplacien sur niveaux de gris 0..255 à taille fixe.
    detail_ratio: float
    native_sharpness: float
    contrast: float
    luminance: float


@dataclass(frozen=True, slots=True)
class PoolContent:
    people: float  # Similarités cosinus, pas des probabilités.
    landscape: float
    unwanted: float
    best_prompt: str
    unwanted_prompt: str = ""

    @property
    def similarity(self) -> float:
        return max(self.people, self.landscape)

    @property
    def margin(self) -> float:
        return self.similarity - self.unwanted

    @property
    def category(self) -> str:
        return "personnes" if self.people >= self.landscape else "paysage"


@dataclass(frozen=True, slots=True)
class PoolDecision:
    status: Literal["accepted", "review", "rejected"]
    reason: str


class PhotoPoolPolicy:
    """Deux conditions obligatoires : qualité suffisante et contenu pertinent."""

    def __init__(self, settings: PhotoPoolSettings) -> None:
        self.settings = settings

    def quality_rejection(self, quality: PoolQuality) -> str | None:
        values = (
            quality.sharpness,
            quality.detail_ratio,
            quality.native_sharpness,
            quality.contrast,
            quality.luminance,
        )
        if not all(isfinite(v) for v in values):
            return "mesures de qualité invalides"
        if (
            quality.sharpness < self.settings.minimum_sharpness
            or quality.detail_ratio < self.settings.minimum_detail_ratio
        ):
            return "netteté insuffisante"
        if quality.native_sharpness < self.settings.minimum_native_sharpness:
            return "netteté native insuffisante"
        if quality.contrast < self.settings.minimum_contrast:
            return "contraste insuffisant"
        if (
            not self.settings.minimum_luminance
            <= quality.luminance
            <= self.settings.maximum_luminance
        ):
            return "exposition inexploitable"
        return None

    def decide(self, quality: PoolQuality, content: PoolContent | None) -> PoolDecision:
        reason = self.quality_rejection(quality)
        if reason:
            return PoolDecision("rejected", reason)
        if content is None or not all(
            isfinite(v) for v in (content.people, content.landscape, content.unwanted)
        ):
            return PoolDecision("rejected", "analyse de contenu absente ou invalide")
        if (
            content.similarity >= self.settings.minimum_content_similarity
            and content.margin >= self.settings.content_margin
        ):
            return PoolDecision("accepted", content.category)
        if (
            content.similarity >= self.settings.review_similarity
            and content.margin >= self.settings.review_margin
        ):
            return PoolDecision("review", "contenu incertain : " + content.category)
        return PoolDecision("rejected", "aucun paysage ou personnage suffisamment identifiable")
