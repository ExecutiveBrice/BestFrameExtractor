"""Traitement local par lot des vidéos d'un répertoire."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_VIDEO_EXTENSIONS = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4"})


@dataclass(frozen=True, slots=True)
class BatchVideoResult:
    """Résultat détaillé du traitement d'une vidéo individuelle."""

    video_path: Path
    exported_count: int | None
    output_directory: Path | None
    error: str | None


@dataclass(frozen=True, slots=True)
class BatchResult:
    """Synthèse complète d'un traitement de répertoire."""

    source_directory: Path
    videos: tuple[BatchVideoResult, ...]

    @property
    def successes(self) -> tuple[BatchVideoResult, ...]:
        return tuple(result for result in self.videos if result.error is None)

    @property
    def failures(self) -> tuple[BatchVideoResult, ...]:
        return tuple(result for result in self.videos if result.error is not None)


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


def format_batch_result(result: BatchResult) -> str:
    """Produit un rapport CLI lisible, y compris pour les vidéos en échec."""
    if not result.videos:
        extensions = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        return f"Aucune vidéo trouvée dans {result.source_directory} (extensions : {extensions})."
    lines = [
        f"{len(result.successes)} vidéo(s) traitée(s), {len(result.failures)} échec(s).",
    ]
    lines.extend(
        f"OK — {item.video_path.name}: {item.exported_count} image(s) dans {item.output_directory}"
        for item in result.successes
    )
    lines.extend(
        f"ÉCHEC — {item.video_path.name}: {item.error}" for item in result.failures
    )
    return "\n".join(lines)
