"""Tests de sélection diversifiée, qualitative et expliquée."""

from bestshot.domain.candidate_frame import CandidateFrame, PreviewImage
from bestshot.domain.composite_score import AestheticScore, CompositeScore, CompositionScore
from bestshot.domain.deduplication import DeduplicationResult, DuplicateCandidate
from bestshot.domain.face_analysis import FaceScore
from bestshot.domain.refinement import RankedCandidate
from bestshot.domain.scene import Scene
from bestshot.domain.technical_score import TechnicalScore
from bestshot.selection.selector import BestFrameSelector, SelectionSettings


def _ranked(scene_id: int, timestamp: float, score: float) -> RankedCandidate:
    preview = PreviewImage(1, 1, b"\0\0\0")
    candidate = CandidateFrame(scene_id, timestamp, round(timestamp * 1_000), 1, 1, preview)
    technical = TechnicalScore(score, score, score, score, score, score, score)
    face = FaceScore((), None, None, None, None, None, None, None, None)
    composite = CompositeScore(
        score,
        "no_people",
        technical,
        face,
        AestheticScore(0.5, is_neutral=True),
        CompositionScore(0.5, is_neutral=True),
        (),
    )
    return RankedCandidate(candidate, composite)


def _deduplication(*candidates: RankedCandidate) -> DeduplicationResult:
    return DeduplicationResult(candidates, ())


def test_selection_rounds_between_scenes_before_reusing_one_scene() -> None:
    first_best, first_second = _ranked(1, 1.0, 0.90), _ranked(1, 2.0, 0.89)
    second_best, second_second = _ranked(2, 3.0, 0.80), _ranked(2, 4.0, 0.70)
    candidates = [first_best, first_second, second_best, second_second]

    result = BestFrameSelector(SelectionSettings(3, 0.55)).select(
        candidates,
        [Scene(1, 0.0, 2.5, 2.5), Scene(2, 2.5, 5.0, 2.5)],
        _deduplication(*candidates),
        4,
    )

    assert [item.candidate.scene_id for item in result.selected] == [1, 2, 1, 2]


def test_selection_does_not_fill_quota_with_weak_candidates() -> None:
    good, second, third = _ranked(1, 1.0, 0.90), _ranked(1, 2.0, 0.80), _ranked(1, 3.0, 0.70)
    excess, weak = _ranked(1, 4.0, 0.60), _ranked(2, 5.0, 0.40)
    candidates = [good, second, third, excess, weak]

    result = BestFrameSelector(SelectionSettings(3, 0.55)).select(
        candidates,
        [Scene(1, 0.0, 4.5, 4.5), Scene(2, 4.5, 6.0, 1.5)],
        _deduplication(*candidates),
        20,
    )

    assert result.selected == (good, second, third)
    assert {rejection.reason for rejection in result.rejections} == {
        "maximum par scène atteint",
        "score inférieur au minimum",
    }


def test_selection_excludes_candidates_marked_as_duplicates() -> None:
    retained, duplicate = _ranked(1, 1.0, 0.90), _ranked(1, 1.1, 0.89)
    deduplication = DeduplicationResult(
        (retained,),
        (DuplicateCandidate(duplicate, retained, 0.99, 0.1),),
    )

    result = BestFrameSelector(SelectionSettings(3, 0.55)).select(
        [retained, duplicate], [Scene(1, 0.0, 2.0, 2.0)], deduplication, 2
    )

    assert result.selected == (retained,)
    assert result.rejections[0].reason == "doublon visuel proche"
