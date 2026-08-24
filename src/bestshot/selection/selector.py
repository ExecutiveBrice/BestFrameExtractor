"""Sélection diversifiée et expliquée des meilleures candidates dédoublonnées."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from bestshot.domain.deduplication import DeduplicationResult
from bestshot.domain.refinement import RankedCandidate
from bestshot.domain.scene import Scene
from bestshot.domain.selection import SelectionRejection, SelectionResult


@dataclass(frozen=True, slots=True)
class SelectionSettings:
    max_per_scene: int
    minimum_score: float


class BestFrameSelector:
    """Répartit les meilleures candidates entre les scènes sans remplir par médiocrité."""

    def __init__(self, settings: SelectionSettings) -> None:
        self._settings = settings

    def select(
        self,
        candidates: Sequence[RankedCandidate],
        scenes: Sequence[Scene],
        deduplication: DeduplicationResult,
        count: int,
    ) -> SelectionResult:
        if count <= 0 or self._settings.max_per_scene <= 0:
            raise ValueError("Le nombre demandé et le maximum par scène doivent être positifs.")
        if not 0.0 <= self._settings.minimum_score <= 1.0:
            raise ValueError("Le score minimal doit être compris entre 0 et 1.")

        known_scenes = {scene.index for scene in scenes}
        kept = set(deduplication.kept)
        duplicates = {duplicate.discarded for duplicate in deduplication.duplicates}
        rejections = [
            SelectionRejection(candidate, "doublon visuel proche")
            for candidate in candidates
            if candidate in duplicates
        ]
        per_scene: defaultdict[int, list[RankedCandidate]] = defaultdict(list)
        for candidate in candidates:
            if candidate not in kept:
                continue
            if candidate.candidate.scene_id not in known_scenes:
                rejections.append(SelectionRejection(candidate, "scène inconnue"))
            elif candidate.composite_score.final_score < self._settings.minimum_score:
                rejections.append(SelectionRejection(candidate, "score inférieur au minimum"))
            else:
                per_scene[candidate.candidate.scene_id].append(candidate)
        for values in per_scene.values():
            values.sort(key=lambda item: item.composite_score.final_score, reverse=True)

        selected = self._round_robin(per_scene, count)
        selected_set = set(selected)
        for values in per_scene.values():
            for position, candidate in enumerate(values):
                if candidate not in selected_set:
                    reason = (
                        "maximum par scène atteint"
                        if position >= self._settings.max_per_scene
                        else "nombre de sélections demandé atteint"
                    )
                    rejections.append(SelectionRejection(candidate, reason))
        return SelectionResult(count, tuple(selected), tuple(rejections), deduplication)

    def _round_robin(
        self, per_scene: dict[int, list[RankedCandidate]], count: int
    ) -> list[RankedCandidate]:
        selected: list[RankedCandidate] = []
        positions = {scene_id: 0 for scene_id in per_scene}
        while len(selected) < count:
            heads = [
                values[positions[scene_id]]
                for scene_id, values in per_scene.items()
                if positions[scene_id] < len(values)
                and positions[scene_id] < self._settings.max_per_scene
            ]
            if not heads:
                break
            for candidate in sorted(heads, key=lambda item: item.composite_score.final_score, reverse=True):
                if len(selected) == count:
                    break
                selected.append(candidate)
                positions[candidate.candidate.scene_id] += 1
        return selected
