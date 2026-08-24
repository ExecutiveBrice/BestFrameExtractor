"""Export des frames sélectionnées depuis la vidéo source et manifeste JSON."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from bestshot.domain.selection import SelectionResult


@dataclass(frozen=True, slots=True)
class ExportSettings:
    jpeg_quality: int


@dataclass(frozen=True, slots=True)
class ExportResult:
    output_directory: Path
    image_paths: tuple[Path, ...]
    manifest_path: Path


class FrameExporter(Protocol):
    def extract(self, video_path: Path, timestamp: float, output_path: Path, jpeg_quality: int) -> None:
        """Extrait une frame native sans passer par les previews."""


class FinalExporter:
    """Exporte les sélections et leurs métadonnées sans agrandir un aperçu."""

    def __init__(self, frame_exporter: FrameExporter, settings: ExportSettings) -> None:
        self._frame_exporter = frame_exporter
        self._settings = settings

    def export(
        self, video_path: Path, selection: SelectionResult, output_directory: Path, image_format: str = "jpeg"
    ) -> ExportResult:
        extension = _extension_for(image_format)
        if not 1 <= self._settings.jpeg_quality <= 31:
            raise ValueError("La qualité JPEG FFmpeg doit être comprise entre 1 et 31.")
        output_directory.mkdir(parents=True, exist_ok=True)
        image_paths: list[Path] = []
        manifest_images: list[dict[str, object]] = []
        for index, ranked in enumerate(selection.selected, start=1):
            candidate = ranked.candidate
            output_path = output_directory / f"{video_path.stem}_{index:04d}.{extension}"
            self._frame_exporter.extract(video_path, candidate.timestamp, output_path, self._settings.jpeg_quality)
            image_paths.append(output_path)
            manifest_images.append(
                {
                    "file": output_path.name,
                    "source_video": str(video_path),
                    "timestamp": candidate.timestamp,
                    "frame": candidate.frame_index,
                    "score_final": ranked.composite_score.final_score,
                    "scores": {
                        "technical": asdict(ranked.composite_score.technical),
                        "face": asdict(ranked.composite_score.face),
                        "aesthetic": asdict(ranked.composite_score.aesthetic),
                        "composition": asdict(ranked.composite_score.composition),
                        "profile": ranked.composite_score.profile,
                        "reasons": asdict(ranked.composite_score)["reasons"],
                    },
                    "scene_source": candidate.scene_id,
                }
            )
        manifest_path = output_directory / "manifest.json"
        manifest_path.write_text(
            json.dumps({"source_video": str(video_path), "images": manifest_images}, indent=2),
            encoding="utf-8",
        )
        return ExportResult(output_directory, tuple(image_paths), manifest_path)


def _extension_for(image_format: str) -> str:
    normalized = image_format.lower()
    if normalized in {"jpg", "jpeg"}:
        return "jpg"
    if normalized == "png":
        return "png"
    raise ValueError("Le format d'export doit être jpeg ou png.")
