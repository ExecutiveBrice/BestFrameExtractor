from pathlib import Path

from bestshot.services.batch import find_videos, format_batch_result, process_video_batch


def test_batch_discovers_supported_files_and_continues_after_a_failure(tmp_path: Path) -> None:
    (tmp_path / "B.MOV").touch()
    (tmp_path / "a.mp4").touch()
    (tmp_path / "notes.txt").touch()
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "ignored.mp4").touch()

    def process(video_path: Path) -> tuple[int, Path]:
        if video_path.name == "B.MOV":
            raise RuntimeError("codec non pris en charge")
        return 2, Path("photos") / video_path.stem

    result = process_video_batch(tmp_path, process)

    assert find_videos(tmp_path) == (tmp_path / "a.mp4", tmp_path / "B.MOV")
    assert len(result.successes) == 1
    assert len(result.failures) == 1
    assert "a.mp4: 2 image(s)" in format_batch_result(result)
    assert "B.MOV: codec non pris en charge" in format_batch_result(result)
