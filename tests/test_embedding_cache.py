"""Tests du cache local persistant d'embeddings."""

from pathlib import Path

from bestshot.embedding.cache import EmbeddingCache, EmbeddingCacheKey


def test_cache_persists_a_normalized_vector_with_a_video_frame_model_key(tmp_path: Path) -> None:
    video = tmp_path / "family.mp4"
    video.write_bytes(b"local-video")
    key = EmbeddingCacheKey.for_frame(video, timestamp=1.25, frame_index=42, model_version="dino-test-1")
    cache = EmbeddingCache(tmp_path / "embeddings")

    cache.put(key, (3.0, 4.0))

    assert cache.get(key) == (0.6, 0.8)
    assert EmbeddingCache(tmp_path / "embeddings").get(key) == (0.6, 0.8)
    assert key.digest != EmbeddingCacheKey.for_frame(
        video,
        timestamp=1.25,
        frame_index=42,
        model_version="dino-test-2",
    ).digest
    assert key.digest != EmbeddingCacheKey.for_frame(
        video,
        timestamp=1.50,
        frame_index=42,
        model_version="dino-test-1",
    ).digest
