"""Orchestration de classement, dédoublonnage et sélection finale."""

from collections.abc import Iterable, Sequence

from bestshot.domain.candidate_frame import CandidateFrame
from bestshot.domain.refinement import RankedCandidate
from bestshot.domain.scene import Scene
from bestshot.domain.selection import SelectionResult
from bestshot.scoring.composite import CompositeScorer
from bestshot.scoring.face import FaceScoreProvider
from bestshot.scoring.technical import TechnicalScorer
from bestshot.selection.deduplicate import Deduplicator
from bestshot.selection.selector import BestFrameSelector


def rank_candidates(
    candidates: Iterable[CandidateFrame],
    technical_scorer: TechnicalScorer,
    face_scorer: FaceScoreProvider,
    composite_scorer: CompositeScorer,
) -> list[RankedCandidate]:
    """Calcule les scores structurés requis par la sélection."""
    return [
        RankedCandidate(
            candidate,
            composite_scorer.score(
                technical_scorer.score(candidate.preview), face_scorer.score(candidate.preview)
            ),
        )
        for candidate in candidates
    ]


def select_best_frames(
    ranked_candidates: Sequence[RankedCandidate],
    scenes: Sequence[Scene],
    deduplicator: Deduplicator,
    selector: BestFrameSelector,
    count: int | None,
) -> SelectionResult:
    """Dédoublonne puis sélectionne les frames sans cacher les exclusions."""
    return selector.select(
        ranked_candidates,
        scenes,
        deduplicator.deduplicate(list(ranked_candidates)),
        count,
    )


def format_selection_result(result: SelectionResult) -> str:
    """Produit une sortie CLI concise et vérifiable."""
    requested = str(result.requested_count) if result.requested_count is not None else "sans quota"
    lines = [f"{len(result.selected)} sélection(s) — {requested}."]
    lines.extend(
        f"Scène {item.candidate.scene_id} — {item.candidate.timestamp:.3f}s "
        f"— score {item.composite_score.final_score:.3f}"
        for item in result.selected
    )
    lines.append(f"Rejets : {len(result.rejections)}")
    return "\n".join(lines)
