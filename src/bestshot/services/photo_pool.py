"""Extraction en flux d'un pool : pas de quota par scène ni dédoublonnage."""

from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import asdict
from math import isfinite
from pathlib import Path
from typing import Protocol

from bestshot.domain.candidate_frame import PreviewImage
from bestshot.domain.photo_pool import (
    PhotoPoolPolicy,
    PhotoPoolSettings,
    PoolContent,
    PoolFrame,
    PoolQuality,
)
from bestshot.services.batch import BatchProgress, BatchResult, BatchVideoResult, find_videos


class PoolFrameSource(Protocol):
    def frames(
        self,
        video_path: Path,
        fps: float,
        start: float = 0,
        duration: float | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> Iterator[PoolFrame]: ...


class PoolQualityScorer(Protocol):
    def score(self, image: PreviewImage) -> PoolQuality: ...


class PoolContentScorer(Protocol):
    def score(self, preview: PreviewImage) -> PoolContent: ...


class PoolImageWriter(Protocol):
    def write(self, image: PreviewImage, path: Path, image_format: str, quality: int) -> None: ...


class PoolReport(Protocol):
    directory: Path

    def append(self, record: dict[str, object]) -> None: ...

    def finish(self, summary: dict[str, object]) -> None: ...


class PoolRepository(Protocol):
    def create(self, root: Path, stem: str) -> PoolReport: ...


class PhotoPoolRunner:
    def __init__(
        self,
        source: PoolFrameSource,
        quality: PoolQualityScorer,
        content: PoolContentScorer,
        writer: PoolImageWriter,
        repository: PoolRepository,
        settings: PhotoPoolSettings,
        model_metadata: dict[str, object],
    ) -> None:
        self._source = source
        self._quality = quality
        self._content = content
        self._writer = writer
        self._repository = repository
        self._policy = PhotoPoolPolicy(settings)
        self._model_metadata = model_metadata

    def run(
        self,
        directory: Path,
        count: int | None,
        output_directory: Path,
        image_format: str = "jpeg",
        on_progress: Callable[[BatchProgress], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        *,
        start: float = 0,
        duration: float | None = None,
    ) -> BatchResult:
        """Accepte un fichier ou un dossier ; les photos sortent dès leur validation."""
        if count is not None:
            raise ValueError("Le pool photo n'utilise pas de quota ; count doit être None.")
        if image_format not in {"jpeg", "jpg", "png"}:
            raise ValueError("Le format doit être jpeg ou png.")
        if (
            not isfinite(start)
            or start < 0
            or (duration is not None and (not isfinite(duration) or duration <= 0))
        ):
            raise ValueError("Le début doit être positif ou nul et la durée positive.")
        if not directory.exists():
            raise ValueError("La source vidéo n'existe pas.")
        videos = (directory,) if directory.is_file() else find_videos(directory)
        results: list[BatchVideoResult] = []
        cumulative: Counter[str] = Counter()
        stop = should_stop or (lambda: False)
        output_directory = output_directory.resolve()
        if any(output_directory == video.resolve().parent for video in videos):
            raise ValueError("Choisissez une destination distincte du dossier des vidéos.")
        for index, video in enumerate(videos, 1):
            if stop():
                return BatchResult(directory, tuple(results), cancelled=True)

            def report(
                state: str,
                paths: tuple[Path, ...] = (),
                current: int = index,
                current_video: Path = video,
            ) -> None:
                if on_progress:
                    on_progress(
                        BatchProgress(
                            current,
                            len(videos),
                            current_video,
                            state,
                            paths,
                            cumulative["selected"],
                            cumulative["exported"],
                        )
                    )

            report("started")
            store = self._repository.create(output_directory, video.stem)
            destination = store.directory
            image_paths: list[Path] = []
            totals: Counter[str] = Counter()
            error: str | None = None
            cancelled = False
            try:
                # Un enregistrement JSON par candidate ; aucun buffer d'image conservé.
                for frame in self._source.frames(
                    video, self._policy.settings.fps, start, duration, stop
                ):
                    if stop():
                        break
                    quality = self._quality.score(frame.image)
                    content = None
                    if self._policy.quality_rejection(quality) is None:
                        if stop():
                            break
                        content = self._content.score(frame.image)
                    decision = self._policy.decide(quality, content)
                    if stop():
                        break
                    relative_path = None
                    if decision.status != "rejected":
                        folder = "pool" if decision.status == "accepted" else "a_verifier"
                        extension = "png" if image_format == "png" else "jpg"
                        filename = f"{video.stem}_{frame.frame_index:08d}.{extension}"
                        relative_path = Path(folder) / filename
                        target = destination / relative_path
                        self._writer.write(
                            frame.image, target, image_format, self._policy.settings.jpeg_quality
                        )
                        if decision.status == "accepted":
                            image_paths.append(target)
                            cumulative["selected"] += 1
                            cumulative["exported"] += 1
                    totals[decision.status] += 1
                    record: dict[str, object] = {
                        "timestamp": frame.timestamp,
                        "frame": frame.frame_index,
                        "width": frame.image.width,
                        "height": frame.image.height,
                        "status": decision.status,
                        "reason": decision.reason,
                        "file": str(relative_path) if relative_path else None,
                        "quality": asdict(quality),
                        "content": asdict(content) if content else None,
                    }
                    store.append(record)
                    if decision.status == "accepted":
                        report("exported", (target,))
                    elif totals.total() % 30 == 0:
                        report("analysed")
                cancelled = stop()
            except (OSError, RuntimeError, ValueError) as exception:
                error = str(exception)
            finally:
                store.finish(
                    {
                        "mode": "photo-pool",
                        "source_video": str(video.resolve()),
                        "settings": asdict(self._policy.settings),
                        "model": self._model_metadata,
                        "start": start,
                        "duration": duration,
                        "counts": dict(totals),
                        "cancelled": cancelled or stop(),
                        "error": error,
                        "decisions": "decisions.jsonl",
                        "images": [str(path.relative_to(destination)) for path in image_paths],
                    }
                )
            results.append(
                BatchVideoResult(
                    video,
                    len(image_paths),
                    destination,
                    error,
                    tuple(image_paths),
                    len(image_paths),
                    cancelled or stop(),
                )
            )
            report("failed" if error else "stopped" if stop() else "completed")
            if stop():
                return BatchResult(directory, tuple(results), cancelled=True)
        return BatchResult(directory, tuple(results))
