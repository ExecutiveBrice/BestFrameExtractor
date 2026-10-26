from pathlib import Path
from unittest.mock import Mock

import pytest

pytest.importorskip("PySide6")

from bestshot.desktop import estimate_worker
from bestshot.desktop.estimate_worker import EstimateWorker
from bestshot.services.extraction_estimate import VideoDurationSummary


def test_estimate_worker_emits_summary_with_request_id(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = VideoDurationSummary(60, 1, 0)
    inspect = Mock(return_value=summary)
    monkeypatch.setattr(estimate_worker, "inspect_video_durations", inspect)
    worker = EstimateWorker(Path("videos"), 4)
    results: list[tuple[int, VideoDurationSummary]] = []
    worker.signals.completed.connect(lambda request, result: results.append((request, result)))

    worker.run()

    assert results == [(4, summary)]
    assert inspect.call_args.args[0] == Path("videos")


def test_estimate_worker_reports_directory_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        estimate_worker, "inspect_video_durations", Mock(side_effect=OSError("Accès refusé"))
    )
    worker = EstimateWorker(Path("videos"), 4)
    failures: list[tuple[int, str]] = []
    worker.signals.failed.connect(lambda request, message: failures.append((request, message)))

    worker.run()

    assert failures == [(4, "Accès refusé")]
