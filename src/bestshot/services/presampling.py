"""Cas d'usage de présélection temporelle de la V2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bestshot.sampling.candidate_generator import CandidateGenerationResult, CandidateGenerator
from bestshot.video.probe import VideoProbe


@dataclass(frozen=True, slots=True)
class PresamplingReport:
    """Indicateurs destinés aux adaptateurs, sans exposer de score de netteté global."""

    duration_seconds: float | None
    video_frame_count: int
    analyzed_frame_count: int
    candidate_count: int

    @property
    def candidates_per_minute(self) -> float | None:
        """Calcule le débit réel seulement lorsque la durée est disponible."""
        if self.duration_seconds is None or self.duration_seconds <= 0:
            return None
        return self.candidate_count * 60.0 / self.duration_seconds


def generate_presampling_report(
    video_path: Path,
    probe: VideoProbe,
    generator: CandidateGenerator,
) -> PresamplingReport:
    """Inspecte la durée puis réalise un unique passage de décodage V2."""
    info = probe.inspect(video_path)
    result = generator.generate(video_path)
    return _report_from_result(info.duration_seconds, result)


def format_presampling_report(report: PresamplingReport) -> str:
    """Produit les cinq indicateurs stables affichés par ``bestshot presample``."""
    duration = "indisponible" if report.duration_seconds is None else f"{report.duration_seconds:.3f} s"
    rate = (
        "indisponible"
        if report.candidates_per_minute is None
        else f"{report.candidates_per_minute:.2f}"
    )
    return "\n".join(
        (
            f"Durée : {duration}",
            f"Frames vidéo : {report.video_frame_count}",
            f"Frames analysées : {report.analyzed_frame_count}",
            f"Candidates générées : {report.candidate_count}",
            f"Candidates/minute : {rate}",
        )
    )


def _report_from_result(
    duration_seconds: float | None,
    result: CandidateGenerationResult,
) -> PresamplingReport:
    return PresamplingReport(
        duration_seconds=duration_seconds,
        video_frame_count=result.video_frame_count,
        analyzed_frame_count=result.analyzed_frame_count,
        candidate_count=result.candidate_count,
    )
