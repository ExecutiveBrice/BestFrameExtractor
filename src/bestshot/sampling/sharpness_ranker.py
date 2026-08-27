"""Comparaison locale de netteté au sein d'une même fenêtre temporelle."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np

from bestshot.sampling.temporal_sampler import AnalysisFrame


@dataclass(frozen=True, slots=True)
class RankedAnalysisFrame:
    """Frame et mesure brute de netteté, uniquement valable dans sa fenêtre."""

    frame: AnalysisFrame
    sharpness: float


class SharpnessRanker:
    """Classe les frames d'une fenêtre par variance du Laplacien, sans seuil global."""

    def rank(
        self,
        frames: Iterable[AnalysisFrame],
        keep_per_bucket: int,
    ) -> list[RankedAnalysisFrame]:
        """Retourne les meilleures frames, même si toutes sont peu nettes."""
        if keep_per_bucket <= 0:
            raise ValueError("Le nombre de frames à conserver doit être positif.")
        ranked = [RankedAnalysisFrame(frame=frame, sharpness=self.measure(frame)) for frame in frames]
        return sorted(
            ranked,
            key=lambda item: (-item.sharpness, item.frame.timestamp, item.frame.frame_index),
        )[:keep_per_bucket]

    def measure(self, frame: AnalysisFrame) -> float:
        """Mesure brute : elle n'est jamais normalisée ni comparée entre vidéos."""
        image = frame.grayscale
        expected_size = image.width * image.height
        if image.width <= 0 or image.height <= 0 or len(image.gray_bytes) != expected_size:
            raise ValueError("Le buffer de niveaux de gris ne correspond pas aux dimensions annoncées.")
        pixels = np.frombuffer(image.gray_bytes, dtype=np.uint8).reshape((image.height, image.width))
        return float(cv2.Laplacian(pixels, cv2.CV_64F).var())
