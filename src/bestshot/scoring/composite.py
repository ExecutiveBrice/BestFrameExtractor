"""Combinaison expliquée des scores techniques, visage et futurs critères."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from bestshot.domain.candidate_frame import PreviewImage
from bestshot.domain.composite_score import (
    AestheticScore,
    CompositeReason,
    CompositeScore,
    CompositionScore,
)
from bestshot.domain.face_analysis import FaceScore
from bestshot.domain.technical_score import TechnicalScore


@dataclass(frozen=True, slots=True)
class CompositeWeights:
    """Poids associés aux critères d'un profil de classement."""

    technical: float
    face: float
    aesthetic: float
    composition: float


@dataclass(frozen=True, slots=True)
class CompositeScoringSettings:
    """Poids pour les vidéos avec et sans visage, et valeur neutre temporaire."""

    people: CompositeWeights
    no_people: CompositeWeights
    neutral_score: float


class AestheticScorer(Protocol):
    """Interface réservée à un futur score esthétique sur aperçu réduit."""

    def score(self, preview: PreviewImage) -> AestheticScore:
        """Retourne un score esthétique structuré."""


class CompositionScorer(Protocol):
    """Interface réservée à un futur score de composition sur aperçu réduit."""

    def score(self, preview: PreviewImage) -> CompositionScore:
        """Retourne un score de composition structuré."""


class NeutralAestheticScorer:
    """Implémentation temporaire qui n'influence pas le classement."""

    def __init__(self, neutral_score: float) -> None:
        self._neutral_score = neutral_score

    def score(self, preview: PreviewImage) -> AestheticScore:
        """Retourne une valeur explicitement signalée comme neutre."""
        return AestheticScore(global_score=self._neutral_score, is_neutral=True)


class NeutralCompositionScorer:
    """Implémentation temporaire qui n'influence pas le classement."""

    def __init__(self, neutral_score: float) -> None:
        self._neutral_score = neutral_score

    def score(self, preview: PreviewImage) -> CompositionScore:
        """Retourne une valeur explicitement signalée comme neutre."""
        return CompositionScore(global_score=self._neutral_score, is_neutral=True)


class CompositeScorer:
    """Produit un score final et des raisons de classement traçables."""

    def __init__(self, settings: CompositeScoringSettings) -> None:
        self._settings = settings

    def score(
        self,
        technical: TechnicalScore,
        face: FaceScore,
        aesthetic: AestheticScore | None = None,
        composition: CompositionScore | None = None,
    ) -> CompositeScore:
        """Combine les scores disponibles sans masquer les valeurs de remplacement."""
        has_people = face.global_score is not None
        profile = "people" if has_people else "no_people"
        weights = self._settings.people if has_people else self._settings.no_people
        aesthetic = aesthetic or AestheticScore(self._settings.neutral_score, is_neutral=True)
        composition = composition or CompositionScore(self._settings.neutral_score, is_neutral=True)
        entries = [
            _reason("technical", technical.global_score, weights.technical, False),
            _reason("aesthetic", aesthetic.global_score, weights.aesthetic, aesthetic.is_neutral),
            _reason("composition", composition.global_score, weights.composition, composition.is_neutral),
        ]
        if has_people:
            entries.append(_reason("face", face.global_score, weights.face, False))

        final_score = _weighted_average(entries)
        reasons = tuple(
            sorted(
                entries,
                key=lambda reason: abs(reason.score - self._settings.neutral_score) * reason.weight,
                reverse=True,
            )
        )
        return CompositeScore(
            final_score=final_score,
            profile=profile,
            technical=technical,
            face=face,
            aesthetic=aesthetic,
            composition=composition,
            reasons=reasons,
        )


def _reason(criterion: str, score: float | None, weight: float, is_neutral: bool) -> CompositeReason:
    if score is None:
        raise ValueError(f"Score {criterion} absent pour un profil qui l'exige.")
    detail = "valeur neutre temporaire" if is_neutral else "score calculé"
    return CompositeReason(
        criterion=criterion,
        score=score,
        weight=weight,
        contribution=score * weight,
        detail=detail,
    )


def _weighted_average(reasons: list[CompositeReason]) -> float:
    total_weight = sum(reason.weight for reason in reasons)
    if total_weight <= 0:
        raise ValueError("La somme des poids composites doit être positive.")
    return sum(reason.contribution for reason in reasons) / total_weight
