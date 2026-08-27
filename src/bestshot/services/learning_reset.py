"""Réinitialisation explicite des préférences et du modèle personnel courant."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bestshot.dataset.repository import DatasetRepository


class LearningResetError(RuntimeError):
    """Les données d'apprentissage locales ne peuvent pas être réinitialisées."""


@dataclass(frozen=True, slots=True)
class LearningResetReport:
    """Effets de la réinitialisation, sans supprimer les sources ni les caches."""

    deleted_preference_count: int
    current_model_disabled: bool


class PersonalLearningResetService:
    """Efface les votes pairwise et invalide le modèle personnel actuellement actif."""

    def __init__(self, repository: DatasetRepository, models_directory: Path) -> None:
        self._repository = repository
        self._models_directory = models_directory

    def reset(self) -> LearningResetReport:
        """Réinitialise l'apprentissage tout en conservant candidates, photos et historiques."""
        deleted_preference_count = self._repository.reset_preferences()
        current_model = self._models_directory / "current.json"
        try:
            current_model.unlink(missing_ok=True)
        except OSError as error:
            raise LearningResetError(
                f"Impossible de désactiver le modèle personnel courant : {current_model}"
            ) from error
        return LearningResetReport(
            deleted_preference_count=deleted_preference_count,
            current_model_disabled=current_model.exists() is False,
        )
