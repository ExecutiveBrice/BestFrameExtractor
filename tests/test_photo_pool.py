"""Tests de l'ingestion locale d'un pool photo dédié aux préférences."""

from __future__ import annotations

from pathlib import Path

from bestshot.dataset.preview_cache import PreviewCache
from bestshot.dataset.sqlite_repository import SQLiteDatasetRepository
from bestshot.domain.preview_image import PreviewImage
from bestshot.embedding.cache import EmbeddingCache
from bestshot.infrastructure.photo_pool import PillowPhotoPreviewReader
from bestshot.learning.pair_generator import PairGenerationSettings
from bestshot.services.photo_pool import PhotoPoolEmbeddingRunner, PhotoPoolSettings
from bestshot.services.preferences import generate_photo_pool_preferences


class FakePhotoReader:
    def read(self, photo_path: Path, max_width: int) -> PreviewImage:
        assert max_width == 320
        value = (sum(photo_path.name.encode()) % 3) * 100 + 30
        return PreviewImage(1, 1, bytes((value, 255 - value, 3)))


class FakeProvider:
    device = "cpu"
    model_name = "test-dino"
    model_version = "test-dino-1"

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, image: PreviewImage) -> tuple[float, ...]:
        self.calls += 1
        return (float(image.rgb_bytes[0]), float(image.rgb_bytes[1]))


def test_pillow_reader_reduces_photo_before_embedding(tmp_path: Path) -> None:
    from PIL import Image

    photo = tmp_path / "wide.jpg"
    Image.new("RGB", (1_000, 500), "red").save(photo)

    preview = PillowPhotoPreviewReader().read(photo, max_width=200)

    assert (preview.width, preview.height) == (200, 100)
    assert len(preview.rgb_bytes) == 200 * 100 * 3


def test_photo_pool_is_ingested_locally_and_reuses_embedding_cache(tmp_path: Path) -> None:
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    repository = SQLiteDatasetRepository(tmp_path / "dataset.sqlite")
    provider = FakeProvider()
    runner = PhotoPoolEmbeddingRunner(
        FakePhotoReader(),
        provider,
        EmbeddingCache(tmp_path / "embeddings"),
        repository,
        PreviewCache(tmp_path / "previews"),
        PhotoPoolSettings(preview_max_width=320),
    )

    first_report = runner.run(tmp_path, (first, second))
    second_report = runner.run(tmp_path, (first, second))

    assert first_report.photo_count == 2
    assert first_report.computed_count == 2
    assert second_report.cached_count == 2
    assert provider.calls == 2
    pool = repository.get_video_by_source_path(tmp_path)
    assert pool is not None and pool.id is not None
    assert repository.get_active_learning_pool() == tmp_path.resolve()
    frames = repository.list_frames_for_video(pool.id)
    assert [frame.frame_index for frame in frames] == [0, 1]
    assert all(Path(frame.preview_reference).is_file() for frame in frames)


def test_changed_photo_pool_keeps_history_but_uses_its_latest_version(tmp_path: Path) -> None:
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    third = tmp_path / "third.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    repository = SQLiteDatasetRepository(tmp_path / "dataset.sqlite")
    runner = PhotoPoolEmbeddingRunner(
        FakePhotoReader(),
        FakeProvider(),
        EmbeddingCache(tmp_path / "embeddings"),
        repository,
        PreviewCache(tmp_path / "previews"),
        PhotoPoolSettings(preview_max_width=320),
    )
    runner.run(tmp_path, (first, second))
    initial = repository.get_video_by_source_path(tmp_path)
    assert initial is not None and initial.id is not None

    third.write_bytes(b"third")
    runner.run(tmp_path, (first, second, third))

    latest = repository.get_video_by_source_path(tmp_path)
    assert latest is not None and latest.id is not None
    assert latest.id != initial.id
    assert len(repository.list_frames_for_video(latest.id)) == 3


def test_photo_pool_preferences_use_similarity_without_video_presampling(tmp_path: Path) -> None:
    photos = tuple(tmp_path / name for name in ("a.jpg", "b.jpg", "c.jpg"))
    for index, photo in enumerate(photos):
        photo.write_bytes(f"photo-{index}".encode())
    repository = SQLiteDatasetRepository(tmp_path / "dataset.sqlite")
    runner = PhotoPoolEmbeddingRunner(
        FakePhotoReader(),
        FakeProvider(),
        EmbeddingCache(tmp_path / "embeddings"),
        repository,
        PreviewCache(tmp_path / "previews"),
        PhotoPoolSettings(preview_max_width=320),
    )
    runner.run(tmp_path, photos)

    pairs = generate_photo_pool_preferences(
        repository,
        tmp_path,
        PairGenerationSettings(
            max_pairs_per_group=2,
            photo_pool_maximum_cosine_similarity=0.98,
            photo_pool_minimum_frame_gap=1,
        ),
    )

    assert len(pairs) == 2
    assert {pair.reason for pair in pairs} == {"coverage-similar"}
