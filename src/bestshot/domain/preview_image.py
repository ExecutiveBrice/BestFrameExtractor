"""Buffer RGB transitoire utilisé uniquement par les providers d'embeddings locaux."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PreviewImage:
    """Aperçu RGB réduit, jamais persistant ni envoyé à un service distant."""

    width: int
    height: int
    rgb_bytes: bytes
