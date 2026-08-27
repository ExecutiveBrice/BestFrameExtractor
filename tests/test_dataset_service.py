"""Tests des représentations texte du dataset local."""

from pathlib import Path

from bestshot.dataset.repository import DatasetStats, VideoDatasetSummary, VideoRecord
from bestshot.services.dataset import format_dataset_stats, format_dataset_videos


def test_dataset_formatters_show_skip_separately_from_reject() -> None:
    stats = DatasetStats(2, 10, 3, 2, 5, 0)
    video = VideoRecord(Path("family.mp4"), "0123456789abcdef", 1, 1, id=1)
    summary = VideoDatasetSummary(video, frame_count=10, keep_count=3, reject_count=2, skip_count=5)

    assert "SKIP : 5" in format_dataset_stats(stats)
    assert "REJECT : 2" in format_dataset_stats(stats)
    assert format_dataset_videos([summary]) == (
        "family.mp4 | 0123456789ab | frames=10 KEEP=3 REJECT=2 SKIP=5"
    )
