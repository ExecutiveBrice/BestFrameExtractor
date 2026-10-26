"""Décodage échantillonné, mesures OpenCV et écriture locale du pool photo."""

import json
from collections.abc import Callable, Iterator
from math import isfinite
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from bestshot.domain.candidate_frame import PreviewImage
from bestshot.domain.photo_pool import PoolFrame, PoolQuality
from bestshot.video.candidate_extractor import CandidateExtractionError


class PyAVPoolFrameSource:
    def frames(
        self,
        video_path: Path,
        fps: float,
        start: float = 0,
        duration: float | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> Iterator[PoolFrame]:
        """Décode en flux ; seules les frames échantillonnées deviennent des buffers RGB."""
        import av

        if (
            not isfinite(fps)
            or fps <= 0
            or not isfinite(start)
            or start < 0
            or (duration is not None and (not isfinite(duration) or duration <= 0))
        ):
            raise ValueError("Cadence/durée positives et début positif ou nul attendus.")
        next_sample = start
        end = None if duration is None else start + duration
        try:
            with av.open(str(video_path)) as container:
                if not container.streams.video:
                    raise CandidateExtractionError("Aucun flux vidéo n'a été trouvé.")
                stream = container.streams.video[0]
                origin = (
                    float(stream.start_time * stream.time_base)
                    if stream.start_time is not None and stream.time_base is not None
                    else 0.0
                )
                for index, frame in enumerate(container.decode(stream)):
                    if should_stop is not None and should_stop():
                        break
                    if frame.time is None:
                        continue
                    timestamp = float(frame.time) - origin
                    if end is not None and timestamp >= end:
                        break
                    if timestamp + 1e-8 < next_sample:
                        continue
                    image = frame.to_image().convert("RGB")  # type: ignore[no-untyped-call]
                    yield PoolFrame(
                        timestamp, index, PreviewImage(image.width, image.height, image.tobytes())
                    )
                    while next_sample <= timestamp + 1e-8:
                        next_sample += 1 / fps
        except (OSError, ValueError, RuntimeError) as error:
            raise CandidateExtractionError(
                f"Impossible de décoder {video_path.name} : {error}"
            ) from error


class OpenCVPoolQualityScorer:
    def __init__(self, analysis_width: int) -> None:
        self._width = analysis_width

    def score(self, image: PreviewImage) -> PoolQuality:
        rgb = np.frombuffer(image.rgb_bytes, dtype=np.uint8).reshape(image.height, image.width, 3)
        native = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        # Le test natif empêche le seul redimensionnement de masquer certains flous.
        native_sharpness = float(cv2.Laplacian(native, cv2.CV_64F).var())
        gray = native
        if image.width > self._width:
            gray = cv2.resize(
                native,
                (self._width, max(1, round(image.height * self._width / image.width))),
                interpolation=cv2.INTER_AREA,
            )
        # Atténue le bruit, qui pourrait sinon faire passer une image floue pour nette.
        filtered = cv2.GaussianBlur(gray, (3, 3), 0.5)
        sharpness = float(cv2.Laplacian(filtered, cv2.CV_64F).var())
        gx = cv2.Sobel(filtered, cv2.CV_64F, 1, 0)
        gy = cv2.Sobel(filtered, cv2.CV_64F, 0, 1)
        energy = float(np.mean(gx * gx + gy * gy))
        return PoolQuality(
            sharpness,
            sharpness / max(energy, 1e-9),
            native_sharpness,
            float(gray.std()),
            float(gray.mean()),
        )


class LocalPoolImageWriter:
    def write(self, image: PreviewImage, path: Path, image_format: str, quality: int) -> None:
        """Écrit les pixels de la frame native sélectionnée, sans réextraction approximative."""
        path.parent.mkdir(parents=True, exist_ok=True)
        native = Image.frombytes("RGB", (image.width, image.height), image.rgb_bytes)
        if image_format == "png":
            native.save(path, format="PNG")
        else:
            native.save(path, format="JPEG", quality=quality, subsampling=0)


class LocalPoolReport:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._records = (directory / "decisions.jsonl").open("w", encoding="utf-8")

    def append(self, record: dict[str, object]) -> None:
        self._records.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._records.flush()

    def finish(self, summary: dict[str, object]) -> None:
        self._records.close()
        (self.directory / "manifest.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )


class LocalPoolRepository:
    def create(self, root: Path, stem: str) -> LocalPoolReport:
        """Préserve les essais précédents et les éventuels fichiers utilisateur."""
        root.mkdir(parents=True, exist_ok=True)
        for suffix in range(1, 100_000):
            candidate = root / (stem if suffix == 1 else f"{stem}_{suffix:03d}")
            try:
                candidate.mkdir()
            except FileExistsError:
                continue
            return LocalPoolReport(candidate)
        raise ValueError("Trop de répertoires de résultats pour cette vidéo.")
