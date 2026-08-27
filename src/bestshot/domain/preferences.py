"""Concepts de domaine pour les préférences relatives entre candidates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class PreferenceChoice(StrEnum):
    """Réponse utilisateur pour une paire ordonnée de frames.

    ``SKIP`` signifie explicitement qu'aucune information d'entraînement n'a
    été fournie. Il est représenté par ``NULL`` en persistance et ne devient
    jamais une préférence négative.
    """

    FIRST = "FIRST"
    SECOND = "SECOND"
    EQUAL = "EQUAL"
    SKIP = "SKIP"

    @property
    def is_usable_for_training(self) -> bool:
        """Indique si le choix porte une information supervisée."""
        return self is not PreferenceChoice.SKIP

    def inverted(self) -> PreferenceChoice:
        """Retourne le même jugement lorsque l'ordre des frames est inversé."""
        if self is PreferenceChoice.FIRST:
            return PreferenceChoice.SECOND
        if self is PreferenceChoice.SECOND:
            return PreferenceChoice.FIRST
        return self


@dataclass(frozen=True, slots=True)
class PairwisePreference:
    """Préférence persistée pour une paire de frames canonique.

    Le repository renvoie les identifiants croissants après persistance. Lors
    d'une soumission UI, l'ordre d'affichage peut être arbitraire : il est
    alors normalisé avec le choix associé.
    """

    first_frame_id: int
    second_frame_id: int
    preference: PreferenceChoice
    id: int | None = None
    created_at: str | None = None
    updated_at: str | None = None

    def __post_init__(self) -> None:
        if self.first_frame_id <= 0 or self.second_frame_id <= 0:
            raise ValueError("Les identifiants de frames doivent être positifs.")
        if self.first_frame_id == self.second_frame_id:
            raise ValueError("Une préférence exige deux frames distinctes.")


def canonicalize_preference(
    first_frame_id: int,
    second_frame_id: int,
    preference: PreferenceChoice,
) -> tuple[int, int, PreferenceChoice]:
    """Canonise une paire demandée par l'UI tout en conservant son jugement."""
    if first_frame_id <= 0 or second_frame_id <= 0 or first_frame_id == second_frame_id:
        raise ValueError("Une préférence exige deux frames distinctes et positives.")
    if first_frame_id < second_frame_id:
        return first_frame_id, second_frame_id, preference
    return second_frame_id, first_frame_id, preference.inverted()
