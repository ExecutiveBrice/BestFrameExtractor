"""Apprentissage personnel local fondé sur des préférences pairwise."""

from bestshot.learning.pair_generator import (
    GeneratedPair,
    MixedPairSelectionStrategy,
    NearbyPairSelectionStrategy,
    PairGenerationSettings,
    RandomPairSelectionStrategy,
    SimilarityPairSelectionStrategy,
)

__all__ = [
    "GeneratedPair",
    "MixedPairSelectionStrategy",
    "NearbyPairSelectionStrategy",
    "PairGenerationSettings",
    "RandomPairSelectionStrategy",
    "SimilarityPairSelectionStrategy",
]
