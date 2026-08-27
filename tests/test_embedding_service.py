"""Tests du calcul local et de la réutilisation du cache d'embeddings."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

from bestshot.domain.preview_image import PreviewImage
from bestshot.embedding.cache import EmbeddingCache
from bestshot.embedding.provider import EmbeddingVector
from bestshot.infrastructure.embedding_frames import CandidatePreview
from bestshot.sampling.candidate_generator import CandidateGenerationResult, PresampledCandidate
from bestshot.services.embeddings import VideoEmbeddingRunner, format_embedding_report


class FakeCandidateGenerator:
    def __init__(self, candidates: tuple[PresampledCandidate, ...]) -> None:
        self.candidates = candidates

    def generate(self, video_path: Path) -> CandidateGenerationResult:
        del video_path
        return CandidateGenerationResult(self.candidates, video_frame_count=12, analyzed_frame_count=4)


class FakePreviewReader:
    def __init__(self) -> None:
        self.requested_indexes: list[list[int]] = []

    def read(
        self,
        video_path: Path,
        candidates: Sequence[PresampledCandidate],
        max_width: int,
    ) -> Iterator[CandidatePreview]:
        del video_path
        assert max_width == 640
        self.requested_indexes.append([candidate.frame_index for candidate in candidates])
        for candidate in candidates:
            yield CandidatePreview(candidate, PreviewImage(1, 1, b"\x00\x00\x00"))


class FakeEmbeddingProvider:
    device = "cpu"
    model_name = "Fake DINO"
    model_version = "fake-dino-test-1"

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, image: PreviewImage) -> EmbeddingVector:
        assert image.rgb_bytes == b"\x00\x00\x00"
        self.calls += 1
        return (3.0, 4.0)


def _candidate(index: int) -> PresampledCandidate:
    return PresampledCandidate(
        timestamp=index / 10.0,
        frame_index=index,
        source_width=1920,
        source_height=1080,
        bucket_index=0,
    )


def test_runner_uses_persistent_cache_before_calling_provider(tmp_path: Path) -> None:
    video = tmp_path / "family.mp4"
    video.write_bytes(b"video")
    reader = FakePreviewReader()
    provider = FakeEmbeddingProvider()
    runner = VideoEmbeddingRunner(
        FakeCandidateGenerator((_candidate(2), _candidate(8))),  # type: ignore[arg-type]
        reader,
        provider,
        EmbeddingCache(tmp_path / "cache"),
        analysis_max_width=640,
    )

    first = runner.run(video)
    second = runner.run(video)

    assert (first.computed_count, first.cached_count) == (2, 0)
    assert (second.computed_count, second.cached_count) == (0, 2)
    assert provider.calls == 2
    assert reader.requested_indexes == [[2, 8], []]
    assert "Embeddings calculés : 0" in format_embedding_report(second)
