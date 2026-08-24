"""Persistance locale et séquentielle des aperçus de candidates."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import TextIO

from bestshot.domain.candidate_frame import CandidateFrame
from bestshot.video.candidate_repository import CandidateRepositoryResult


class CandidateRepositoryError(RuntimeError):
    """Un aperçu de candidate ne peut pas être enregistré localement."""


class LocalCandidatePreviewRepository:
    """Écrit les previews JPEG une par une, sans conserver la collection en mémoire."""

    def __init__(self, root_directory: Path) -> None:
        self._root_directory = root_directory

    def store(
        self, video_path: Path, candidates: Iterable[CandidateFrame]
    ) -> CandidateRepositoryResult:
        """Enregistre les candidates et un manifeste JSON dans le dépôt configuré."""
        output_directory = self._root_directory / video_path.stem
        manifest_path = output_directory / "manifest.json"
        temporary_manifest = output_directory / "manifest.json.tmp"
        try:
            output_directory.mkdir(parents=True, exist_ok=True)
            with temporary_manifest.open("w", encoding="utf-8") as manifest:
                count, scene_counts = self._write_candidates(
                    video_path, candidates, output_directory, manifest
                )
            temporary_manifest.replace(manifest_path)
        except (OSError, ValueError) as error:
            raise CandidateRepositoryError(f"Impossible d'enregistrer les candidates : {error}") from error
        return CandidateRepositoryResult(output_directory, count, scene_counts, manifest_path)

    def _write_candidates(
        self,
        video_path: Path,
        candidates: Iterable[CandidateFrame],
        output_directory: Path,
        manifest: TextIO,
    ) -> tuple[int, dict[int, int]]:
        manifest.write('{"source_video": ')
        json.dump(str(video_path), manifest)
        manifest.write(', "candidates": [')
        count = 0
        scene_counts: dict[int, int] = {}
        for candidate in candidates:
            if count:
                manifest.write(",")
            filename = f"scene_{candidate.scene_id:03d}_frame_{candidate.frame_index:08d}.jpg"
            self._save_preview(candidate, output_directory / filename)
            json.dump(_manifest_entry(candidate, filename), manifest)
            count += 1
            scene_counts[candidate.scene_id] = scene_counts.get(candidate.scene_id, 0) + 1
        manifest.write("]}\n")
        return count, scene_counts

    @staticmethod
    def _save_preview(candidate: CandidateFrame, output_path: Path) -> None:
        from PIL import Image

        preview = candidate.preview
        image = Image.frombytes("RGB", (preview.width, preview.height), preview.rgb_bytes)
        image.save(output_path, format="JPEG", quality=90, optimize=True)


def _manifest_entry(candidate: CandidateFrame, filename: str) -> dict[str, object]:
    return {
        "file": filename,
        "scene_id": candidate.scene_id,
        "timestamp": candidate.timestamp,
        "frame_index": candidate.frame_index,
        "source_width": candidate.source_width,
        "source_height": candidate.source_height,
        "preview_width": candidate.preview.width,
        "preview_height": candidate.preview.height,
    }
