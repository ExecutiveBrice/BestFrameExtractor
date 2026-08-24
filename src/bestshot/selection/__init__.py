"""Sélection finale, dédoublonnage et diversification des candidates."""

from bestshot.selection.deduplicate import (
    DeduplicationSettings,
    Deduplicator,
    PerceptualHashSimilarityScorer,
    SimilarityScorer,
)
from bestshot.selection.exporter import ExportResult, ExportSettings, FinalExporter
from bestshot.selection.selector import BestFrameSelector, SelectionSettings

__all__ = [
    "BestFrameSelector",
    "DeduplicationSettings",
    "Deduplicator",
    "ExportResult",
    "ExportSettings",
    "FinalExporter",
    "PerceptualHashSimilarityScorer",
    "SelectionSettings",
    "SimilarityScorer",
]
