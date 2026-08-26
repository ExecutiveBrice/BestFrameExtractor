"""Traitement local par lot des vidéos d'un répertoire."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from bestshot.domain.selection import SelectionResult
from bestshot.selection.exporter import ExportResult

SUPPORTED_VIDEO_EXTENSIONS = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4"})


@dataclass(frozen=True, slots=True)
class BatchVideoResult:
    """Résultat détaillé du traitement d'une vidéo individuelle."""

    video_path: Path
    exported_count: int | None
    output_directory: Path | None
    error: str | None
    image_paths: tuple[Path, ...] = ()
    selected_count: int | None = None
    cancelled: bool = False


@dataclass(frozen=True, slots=True)
class BatchResult:
    """Synthèse complète d'un traitement de répertoire."""

    source_directory: Path
    videos: tuple[BatchVideoResult, ...]
    cancelled: bool = False

    @property
    def successes(self) -> tuple[BatchVideoResult, ...]:
        return tuple(result for result in self.videos if result.error is None)

    @property
    def failures(self) -> tuple[BatchVideoResult, ...]:
        return tuple(result for result in self.videos if result.error is not None)

    @property
    def selected_count(self) -> int:
        """Nombre de frames retenues, y compris avant un éventuel arrêt."""
        return sum(result.selected_count or 0 for result in self.videos)

    @property
    def exported_count(self) -> int:
        """Nombre de fichiers effectivement extraits."""
        return sum(result.exported_count or 0 for result in self.videos)


@dataclass(frozen=True, slots=True)
class BatchProgress:
    """Évènement de progression indépendant de toute interface graphique.

    Les chemins sont renseignés à la fin réussie d'une vidéo afin qu'une présentation
    puisse afficher les exports sans attendre la fin du lot.
    """

    current: int
    total: int
    video_path: Path
    state: str
    image_paths: tuple[Path, ...] = ()
    selected_total: int = 0
    exported_total: int = 0


class VideoSelector(Protocol):
    """Port de sélection d'images pour une vidéo."""

    def select(self, video_path: Path, count: int | None) -> SelectionResult:
        """Retourne les meilleures frames de la vidéo."""


class SelectionExporter(Protocol):
    """Port d'export d'une sélection de frames."""

    def export(
        self,
        video_path: Path,
        selection: SelectionResult,
        output_directory: Path,
        image_format: str = "jpeg",
        *,
        on_image_exported: Callable[[Path], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> ExportResult:
        """Exporte les frames sélectionnées au format demandé."""


def find_videos(directory: Path) -> tuple[Path, ...]:
    """Retourne les vidéos directement présentes dans le répertoire, triées."""
    return tuple(
        path
        for path in sorted(directory.iterdir(), key=lambda entry: entry.name.casefold())
        if path.is_file() and path.suffix.casefold() in SUPPORTED_VIDEO_EXTENSIONS
    )


def process_video_batch(
    directory: Path,
    process_video: Callable[[Path], tuple[int, Path]],
) -> BatchResult:
    """Traite chaque vidéo sans interrompre le lot lorsqu'une vidéo échoue."""
    results: list[BatchVideoResult] = []
    for video_path in find_videos(directory):
        try:
            exported_count, output_directory = process_video(video_path)
        except (OSError, RuntimeError, ValueError) as error:
            results.append(BatchVideoResult(video_path, None, None, str(error)))
        else:
            results.append(BatchVideoResult(video_path, exported_count, output_directory, None))
    return BatchResult(directory, tuple(results))


class BatchExportRunner:
    """Orchestre sélection et export pour chaque vidéo, avec remontée de progression."""

    def __init__(self, selector: VideoSelector, exporter: SelectionExporter) -> None:
        self._selector = selector
        self._exporter = exporter

    def run(
        self,
        directory: Path,
        count: int | None,
        output_directory: Path,
        image_format: str = "jpeg",
        on_progress: Callable[[BatchProgress], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> BatchResult:
        """Traite le dossier sans interrompre le lot lorsqu'une vidéo échoue."""
        videos = find_videos(directory)
        results: list[BatchVideoResult] = []
        selected_total = 0
        exported_total = 0
        for index, video_path in enumerate(videos, start=1):
            if _stop_requested(should_stop):
                _report_progress(
                    on_progress,
                    BatchProgress(
                        index,
                        len(videos),
                        video_path,
                        "stopped",
                        selected_total=selected_total,
                        exported_total=exported_total,
                    ),
                )
                return BatchResult(directory, tuple(results), cancelled=True)
            _report_progress(
                on_progress,
                BatchProgress(
                    index,
                    len(videos),
                    video_path,
                    "started",
                    selected_total=selected_total,
                    exported_total=exported_total,
                ),
            )
            selected_count = 0
            try:
                selection = self._selector.select(video_path, count)
                selected_count = len(selection.selected)
                selected_total += selected_count
                _report_progress(
                    on_progress,
                    BatchProgress(
                        index,
                        len(videos),
                        video_path,
                        "selected",
                        selected_total=selected_total,
                        exported_total=exported_total,
                    ),
                )
                if _stop_requested(should_stop):
                    results.append(
                        BatchVideoResult(
                            video_path,
                            0,
                            None,
                            None,
                            selected_count=selected_count,
                            cancelled=True,
                        )
                    )
                    _report_progress(
                        on_progress,
                        BatchProgress(
                            index,
                            len(videos),
                            video_path,
                            "stopped",
                            selected_total=selected_total,
                            exported_total=exported_total,
                        ),
                    )
                    return BatchResult(directory, tuple(results), cancelled=True)

                def on_image_exported(
                    image_path: Path,
                    current: int = index,
                    total: int = len(videos),
                    current_video: Path = video_path,
                    selection_total: int = selected_total,
                ) -> None:
                    nonlocal exported_total
                    exported_total += 1
                    _report_progress(
                        on_progress,
                        BatchProgress(
                            current,
                            total,
                            current_video,
                            "exported",
                            (image_path,),
                            selection_total,
                            exported_total,
                        ),
                    )

                export = self._exporter.export(
                    video_path,
                    selection,
                    output_directory / video_path.stem,
                    image_format,
                    on_image_exported=on_image_exported,
                    should_stop=should_stop,
                )
            except (OSError, RuntimeError, ValueError) as error:
                result = BatchVideoResult(
                    video_path,
                    None,
                    None,
                    str(error),
                    selected_count=selected_count,
                )
                _report_progress(
                    on_progress,
                    BatchProgress(
                        index,
                        len(videos),
                        video_path,
                        "failed",
                        selected_total=selected_total,
                        exported_total=exported_total,
                    ),
                )
            else:
                result = BatchVideoResult(
                    video_path,
                    len(export.image_paths),
                    export.output_directory,
                    None,
                    export.image_paths,
                    selected_count,
                    export.cancelled,
                )
                if export.cancelled:
                    _report_progress(
                        on_progress,
                        BatchProgress(
                            index,
                            len(videos),
                            video_path,
                            "stopped",
                            export.image_paths,
                            selected_total,
                            exported_total,
                        ),
                    )
                    results.append(result)
                    return BatchResult(directory, tuple(results), cancelled=True)
                _report_progress(
                    on_progress,
                    BatchProgress(
                        index,
                        len(videos),
                        video_path,
                        "completed",
                        export.image_paths,
                        selected_total,
                        exported_total,
                    ),
                )
            results.append(result)
        return BatchResult(directory, tuple(results))


def format_batch_result(result: BatchResult) -> str:
    """Produit un rapport CLI lisible, y compris pour les vidéos en échec."""
    if not result.videos:
        extensions = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        return f"Aucune vidéo trouvée dans {result.source_directory} (extensions : {extensions})."
    status = "Traitement arrêté" if result.cancelled else "Traitement terminé"
    lines = [f"{status} : {len(result.successes)} vidéo(s), {len(result.failures)} échec(s)."]
    lines.extend(
        f"OK — {item.video_path.name}: {item.exported_count} image(s) dans {item.output_directory}"
        for item in result.successes
    )
    lines.extend(
        f"ÉCHEC — {item.video_path.name}: {item.error}" for item in result.failures
    )
    return "\n".join(lines)


def _report_progress(
    callback: Callable[[BatchProgress], None] | None, progress: BatchProgress
) -> None:
    if callback is not None:
        callback(progress)


def _stop_requested(callback: Callable[[], bool] | None) -> bool:
    return callback is not None and callback()
