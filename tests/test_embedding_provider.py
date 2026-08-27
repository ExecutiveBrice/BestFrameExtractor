"""Tests du contrat commun des providers d'embeddings."""

from math import isclose

import pytest

from bestshot.embedding.provider import EmbeddingError, normalize_embedding


def test_normalize_embedding_returns_a_unit_vector() -> None:
    vector = normalize_embedding((3.0, 4.0))

    assert vector == (0.6, 0.8)
    assert isclose(sum(value * value for value in vector), 1.0)


def test_normalize_embedding_rejects_a_zero_vector() -> None:
    with pytest.raises(EmbeddingError, match="nul"):
        normalize_embedding((0.0, 0.0))
