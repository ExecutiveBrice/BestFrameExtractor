"""Cas d'usage local de calcul et de cache des embeddings des candidates V2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from bestshot.dataset.preview_cache import PreviewCache
from bestshot.dataset.repository import DatasetRepository, FrameRecord
from bestshot.dataset.sqlite_repository import video_record_from_path
from bestshot.embedding.cache import EmbeddingCache, EmbeddingCacheKey
from bestshot.embedding.provider import ImageEmbeddingProvider
from bestshot.infrastructure.embedding_frames import CandidatePreviewReader, EmbeddingFrameReadError
from bestshot.sampling.candidate_generator import CandidateGenerator, PresampledCandidate


@dataclass(frozen=True, slots=True)
class EmbeddingReport:
    """Statistiques d'exécution, sans exposer les vecteurs ni les images."""

    device: str
    model_name: str
    computed_count: int
    cached_count: int
    elapsed_seconds: float


class VideoEmbeddingRunner:
    """Calcule les seuls embeddings absents du cache persistant local."""

    def __init__(
        self,
        candidate_generator: CandidateGenerator,
        preview_reader: CandidatePreviewReader,
        provider: ImageEmbeddingProvider,
        cache: EmbeddingCache,
        analysis_max_width: int,
        dataset_repository: DatasetRepository | None = None,
        preview_cache: PreviewCache | None = None,
    ) -> None:
        self._candidate_generator = candidate_generator
        self._preview_reader = preview_reader
        self._provider = provider
        self._cache = cache
        self._analysis_max_width = analysis_max_width
        self._dataset_repository = dataset_repository
        self._preview_cache = preview_cache
        if (dataset_repository is None) != (preview_cache is None):
            raise ValueError("Le repository et le cache d'aperçus doivent être fournis ensemble.")

    def run(self, video_path: Path) -> EmbeddingReport:
        """Présélectionne, relit puis embedde localement les candidates non mises en cache."""
        started = perf_counter()
        candidates = self._candidate_generator.generate(video_path).candidates
        dataset_video = (
            self._dataset_repository.upsert_video(video_record_from_path(video_path))
            if self._dataset_repository is not None
            else None
        )
        if dataset_video is not None and dataset_video.id is None:
            raise RuntimeError("La vidéo du dataset n'a pas reçu d'identifiant.")
        dataset_video_id = dataset_video.id if dataset_video is not None else None
        keys: dict[int, EmbeddingCacheKey] = {}
        missing: list[tuple[PresampledCandidate, EmbeddingCacheKey]] = []
        cached_count = 0
        for candidate in candidates:
            key = EmbeddingCacheKey.for_frame(
                video_path,
                candidate.timestamp,
                candidate.frame_index,
                self._provider.model_version,
            )
            keys[candidate.frame_index] = key
            if self._cache.get(key) is None:
                missing.append((candidate, key))
            else:
                cached_count += 1

        keys_by_frame = {candidate.frame_index: key for candidate, key in missing}
        computed_count = 0
        for candidate_preview in self._preview_reader.read(
            video_path,
            [candidate for candidate, _ in missing],
            self._analysis_max_width,
        ):
            frame_index = candidate_preview.candidate.frame_index
            if frame_index not in keys_by_frame:
                raise EmbeddingFrameReadError("Une frame relue ne correspond à aucune candidate attendue.")
            key = keys_by_frame.pop(frame_index)
            self._cache.put(key, self._provider.embed(candidate_preview.preview))
            computed_count += 1
        if keys_by_frame:
            missing_indexes = ", ".join(str(index) for index in sorted(keys_by_frame))
            raise EmbeddingFrameReadError(f"Frames candidates introuvables dans la vidéo : {missing_indexes}")

        if dataset_video_id is not None and self._preview_cache is not None:
            self._store_dataset_candidates(video_path, candidates, keys, dataset_video_id)

        return EmbeddingReport(
            device=self._provider.device,
            model_name=self._provider.model_name,
            computed_count=computed_count,
            cached_count=cached_count,
            elapsed_seconds=perf_counter() - started,
        )

    def _store_dataset_candidates(
        self,
        video_path: Path,
        candidates: tuple[PresampledCandidate, ...],
        keys: dict[int, EmbeddingCacheKey],
        video_id: int,
    ) -> None:
        """Rélit seulement les candidates retenues pour persister leurs aperçus réduits."""
        assert self._dataset_repository is not None
        assert self._preview_cache is not None
        dataset_video = self._dataset_repository.get_video_by_source_path(video_path)
        if dataset_video is None:
            raise RuntimeError("La vidéo ne peut pas être relue après son ingestion.")
        pending = {candidate.frame_index: candidate for candidate in candidates}
        for candidate_preview in self._preview_reader.read(
            video_path, candidates, self._analysis_max_width
        ):
            candidate = candidate_preview.candidate
            key = keys.get(candidate.frame_index)
            if key is None:
                raise EmbeddingFrameReadError("Une candidate de dataset n'a pas de clé d'embedding.")
            self._dataset_repository.upsert_frame(
                FrameRecord(
                    video_id=video_id,
                    timestamp=candidate.timestamp,
                    frame_index=candidate.frame_index,
                    preview_reference=self._preview_cache.put(
                        dataset_video.video_hash, candidate.frame_index, candidate_preview.preview
                    ),
                    sharpness=candidate.sharpness,
                    embedding_reference=str(self._cache.reference_for(key)),
                )
            )
            pending.pop(candidate.frame_index, None)
        if pending:
            indexes = ", ".join(str(index) for index in sorted(pending))
            raise EmbeddingFrameReadError(f"Aperçus candidates introuvables dans la vidéo : {indexes}")


def format_embedding_report(report: EmbeddingReport) -> str:
    """Produit les cinq indicateurs requis par ``bestshot embeddings``."""
    return "\n".join(
        (
            f"Device : {report.device.upper()}",
            f"Modèle : {report.model_name}",
            f"Embeddings calculés : {report.computed_count}",
            f"Embeddings depuis le cache : {report.cached_count}",
            f"Temps de traitement : {report.elapsed_seconds:.3f} s",
        )
    )
