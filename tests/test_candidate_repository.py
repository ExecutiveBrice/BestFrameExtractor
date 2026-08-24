import json
from pathlib import Path

from bestshot.domain.candidate_frame import CandidateFrame, PreviewImage
from bestshot.infrastructure.candidate_repository import LocalCandidatePreviewRepository


def _candidate(scene_id: int, frame_index: int) -> CandidateFrame:
    return CandidateFrame(
        scene_id=scene_id,
        timestamp=frame_index / 10.0,
        frame_index=frame_index,
        source_width=1920,
        source_height=1080,
        preview=PreviewImage(2, 1, b"\xff\x00\x00\x00\xff\x00"),
    )


def test_repository_persists_reduced_previews_and_manifest_in_a_stream(tmp_path: Path) -> None:
    consumed = 0

    def candidates():  # type: ignore[no-untyped-def]
        nonlocal consumed
        for candidate in (_candidate(1, 4), _candidate(2, 8)):
            consumed += 1
            yield candidate

    result = LocalCandidatePreviewRepository(tmp_path / "candidates").store(
        Path("vacances.mp4"), candidates()
    )

    assert consumed == 2
    assert result.output_directory == tmp_path / "candidates" / "vacances"
    assert result.candidate_count == 2
    assert result.scene_counts == {1: 1, 2: 1}
    assert (result.output_directory / "scene_001_frame_00000004.jpg").is_file()
    assert (result.output_directory / "scene_002_frame_00000008.jpg").is_file()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_video"] == "vacances.mp4"
    assert manifest["candidates"][0]["preview_width"] == 2
