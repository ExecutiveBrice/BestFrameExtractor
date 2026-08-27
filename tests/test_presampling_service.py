"""Tests du rapport textuel de présampling V2."""

from bestshot.services.presampling import PresamplingReport, format_presampling_report


def test_format_presampling_report_lists_the_required_indicators() -> None:
    report = PresamplingReport(
        duration_seconds=120.0,
        video_frame_count=3_600,
        analyzed_frame_count=960,
        candidate_count=30,
    )

    assert format_presampling_report(report).splitlines() == [
        "Durée : 120.000 s",
        "Frames vidéo : 3600",
        "Frames analysées : 960",
        "Candidates générées : 30",
        "Candidates/minute : 15.00",
    ]
