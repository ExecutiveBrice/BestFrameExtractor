"""Embeddings visuels locaux et interchangeables du pipeline V2."""

from bestshot.embedding.cache import EmbeddingCache, EmbeddingCacheKey
from bestshot.embedding.dinov2 import (
    DINOv2EmbeddingProvider,
    DINOv2ModelManager,
    DINOv2Settings,
)
from bestshot.embedding.provider import EmbeddingVector, ImageEmbeddingProvider

__all__ = [
    "DINOv2EmbeddingProvider",
    "DINOv2ModelManager",
    "DINOv2Settings",
    "EmbeddingCache",
    "EmbeddingCacheKey",
    "EmbeddingVector",
    "ImageEmbeddingProvider",
]
