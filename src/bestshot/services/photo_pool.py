"""Ingestion locale d'un pool de photos pour l'apprentissage pairwise."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from bestshot.dataset.preview_cache import PreviewCache
from bestshot.dataset.repository import DatasetRepository, FrameRecord
from bestshot.dataset.sqlite_repository import photo_pool_record_from_paths
from bestshot.embedding.cache import EmbeddingCache, EmbeddingCacheKey
from bestshot.embedding.provider import ImageEmbeddingProvider
from bestshot.infrastructure.photo_pool import PhotoPreviewReader


class PhotoPoolError(RuntimeError):
    """Le pool local ne peut pas alimenter l'apprentissage personnel."""


@dataclass(frozen=True, slots=True)
class PhotoPoolSettings:
    """Taille maximale des aperçus, propre au pool d'apprentissage photo."""

    preview_max_width: int = 640

    def __post_init__(self) -> None:
        if self.preview_max_width <= 0:
            raise ValueError("La largeur maximale des photos du pool doit être positive.")


@dataclass(frozen=True, slots=True)
class PhotoPoolReport:
    """Indicateurs d'ingestion, sans exposer les pixels ni les embeddings."""

    device: str
    model_name: str
    photo_count: int
    computed_count: int
    cached_count: int
    elapsed_seconds: float


class PhotoPoolEmbeddingRunner:
    """Prépare les photos d'un dossier pour les comparaisons, entièrement en local."""

    def __init__(
        self,
        preview_reader: PhotoPreviewReader,
        provider: ImageEmbeddingProvider,
        cache: EmbeddingCache,
        repository: DatasetRepository,
        preview_cache: PreviewCache,
        settings: PhotoPoolSettings,
    ) -> None:
        self._preview_reader = preview_reader
        self._provider = provider
        self._cache = cache
        self._repository = repository
        self._preview_cache = preview_cache
        self._settings = settings

    def run(self, directory: Path, photos: tuple[Path, ...]) -> PhotoPoolReport:
        """Calcule les embeddings absents et persiste les références d'aperçu du pool."""
        if len(photos) < 2:
            raise PhotoPoolError("Le pool doit contenir au moins deux photos.")
        started = perf_counter()
        pool = self._repository.upsert_video(photo_pool_record_from_paths(directory, photos))
        if pool.id is None:
            raise PhotoPoolError("Le pool de photos n'a pas reçu d'identifiant dans le dataset.")

        computed_count = 0
        cached_count = 0
        for index, photo_path in enumerate(photos):
            preview = self._preview_reader.read(photo_path, self._settings.preview_max_width)
            key = EmbeddingCacheKey.for_frame(photo_path, 0.0, 0, self._provider.model_version)
            if self._cache.get(key) is None:
                self._cache.put(key, self._provider.embed(preview))
                computed_count += 1
            else:
                cached_count += 1
            self._repository.upsert_frame(
                FrameRecord(
                    video_id=pool.id,
                    timestamp=float(index),
                    frame_index=index,
                    preview_reference=self._preview_cache.put(pool.video_hash, index, preview),
                    sharpness=0.0,
                    embedding_reference=str(self._cache.reference_for(key)),
                )
            )
        self._repository.set_active_learning_pool(directory)

        return PhotoPoolReport(
            device=self._provider.device,
            model_name=self._provider.model_name,
            photo_count=len(photos),
            computed_count=computed_count,
            cached_count=cached_count,
            elapsed_seconds=perf_counter() - started,
        )
