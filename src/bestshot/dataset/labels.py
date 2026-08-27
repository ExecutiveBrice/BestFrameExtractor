"""Labels explicites du dataset de préférences personnelles."""

from enum import StrEnum


class FrameLabel(StrEnum):
    """Décision utilisateur associée à une candidate."""

    KEEP = "KEEP"
    REJECT = "REJECT"
    SKIP = "SKIP"


def is_training_label(label: FrameLabel) -> bool:
    """Seuls KEEP et REJECT peuvent un jour servir à entraîner un modèle personnel."""
    return label in (FrameLabel.KEEP, FrameLabel.REJECT)


def to_storage_value(label: FrameLabel) -> str | None:
    """Encode SKIP en absence de valeur pour empêcher toute assimilation à REJECT."""
    return None if label is FrameLabel.SKIP else label.value


def from_storage_value(value: object) -> FrameLabel:
    """Décode une valeur SQLite ; NULL est toujours un SKIP."""
    if value is None:
        return FrameLabel.SKIP
    if value == FrameLabel.KEEP.value:
        return FrameLabel.KEEP
    if value == FrameLabel.REJECT.value:
        return FrameLabel.REJECT
    raise ValueError(f"Label de frame invalide dans le dataset : {value!r}")
