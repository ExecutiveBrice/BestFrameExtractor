"""Port stable pour les backbones d'embeddings visuels exécutés localement."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from bestshot.domain.preview_image import PreviewImage

type EmbeddingVector = tuple[float, ...]


class EmbeddingError(ValueError):
    """Un vecteur d'embedding ne peut pas être normalisé ou consommé localement."""


class ImageEmbeddingProvider(Protocol):
    """Backbone interchangeable qui retourne toujours un vecteur L2 normalisé."""

    @property
    def device(self) -> str:
        """Indique le périphérique d'inférence effectivement employé."""

    @property
    def model_name(self) -> str:
        """Nom lisible du backbone utilisé."""

    @property
    def model_version(self) -> str:
        """Identifiant stable des poids, utilisé dans les clés de cache."""

    def embed(self, image: PreviewImage) -> EmbeddingVector:
        """Calcule localement un embedding L2 normalisé pour une image RGB."""


def normalize_embedding(values: Sequence[float]) -> EmbeddingVector:
    """Normalise un vecteur de manière déterministe avant son retour ou son stockage."""
    vector = tuple(float(value) for value in values)
    if not vector or not all(math.isfinite(value) for value in vector):
        raise EmbeddingError("L'embedding doit contenir des valeurs finies.")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        raise EmbeddingError("Un embedding nul ne peut pas être normalisé.")
    return tuple(value / norm for value in vector)
