"""Assemblage local des dépendances nécessaires à l'application de bureau."""

from __future__ import annotations

from dataclasses import dataclass, replace

from bestshot.infrastructure.config import (
    load_deduplication_settings,
    load_export_settings,
    load_selection_settings,
)
from bestshot.infrastructure.ffmpeg import FFmpegFrameExporter
from bestshot.infrastructure.workflow_factory import create_video_selection_workflow
from bestshot.selection.deduplicate import DeduplicationSettings
from bestshot.selection.exporter import FinalExporter
from bestshot.selection.selector import SelectionSettings
from bestshot.services.batch import BatchExportRunner


@dataclass(frozen=True, slots=True)
class DesktopSelectionParameters:
    """Réglages de sélectivité exposés par l'interface graphique."""

    minimum_score: float
    similarity_threshold: float
    temporal_window_ms: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_score <= 1.0:
            raise ValueError("Le score minimal doit être compris entre 0 et 1.")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("Le seuil de similarité doit être compris entre 0 et 1.")
        if self.temporal_window_ms <= 0:
            raise ValueError("La fenêtre de déduplication doit être positive.")


def load_desktop_selection_parameters() -> DesktopSelectionParameters:
    """Propose dans l'interface les valeurs typées du YAML local."""
    selection_settings = load_selection_settings()
    deduplication_settings = load_deduplication_settings()
    return DesktopSelectionParameters(
        minimum_score=selection_settings.minimum_score,
        similarity_threshold=deduplication_settings.similarity_threshold,
        temporal_window_ms=deduplication_settings.temporal_window_ms,
    )


def create_batch_export_runner(parameters: DesktopSelectionParameters) -> BatchExportRunner:
    """Construit un lot local avec les seuils momentanément choisis dans la fenêtre."""
    selection_settings = _selection_settings(parameters)
    deduplication_settings = _deduplication_settings(parameters)
    return BatchExportRunner(
        create_video_selection_workflow(selection_settings, deduplication_settings),
        FinalExporter(FFmpegFrameExporter(), load_export_settings()),
    )


def _selection_settings(parameters: DesktopSelectionParameters) -> SelectionSettings:
    return replace(load_selection_settings(), minimum_score=parameters.minimum_score)


def _deduplication_settings(parameters: DesktopSelectionParameters) -> DeduplicationSettings:
    return replace(
        load_deduplication_settings(),
        similarity_threshold=parameters.similarity_threshold,
        temporal_window_ms=parameters.temporal_window_ms,
    )
