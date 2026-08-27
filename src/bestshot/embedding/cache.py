"""Cache persistant local des embeddings, sans copie des images source."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

from bestshot.embedding.provider import EmbeddingVector, normalize_embedding


@dataclass(frozen=True, slots=True)
class EmbeddingCacheKey:
    """Identifie une frame d'une vidéo pour une version précise du backbone."""

    video_path: str
    video_size: int
    video_mtime_ns: int
    timestamp: float
    frame_index: int
    model_version: str

    @classmethod
    def for_frame(
        cls,
        video_path: Path,
        timestamp: float,
        frame_index: int,
        model_version: str,
    ) -> EmbeddingCacheKey:
        """Inclut l'identité locale de la vidéo, de la frame et des poids employés."""
        if frame_index < 0:
            raise ValueError("L'index de frame doit être positif ou nul.")
        if not model_version:
            raise ValueError("La version du modèle d'embedding est obligatoire.")
        try:
            stat = video_path.stat()
        except OSError as error:
            raise ValueError(f"Impossible d'identifier la vidéo pour le cache : {video_path}") from error
        return cls(
            video_path=str(video_path.resolve()),
            video_size=stat.st_size,
            video_mtime_ns=stat.st_mtime_ns,
            timestamp=timestamp,
            frame_index=frame_index,
            model_version=model_version,
        )

    @property
    def digest(self) -> str:
        """Retourne le nom de fichier déterministe sans exposer le chemin de la vidéo."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """Lit et écrit des vecteurs normalisés dans un répertoire local privé au projet."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def get(self, key: EmbeddingCacheKey) -> EmbeddingVector | None:
        """Retourne un embedding valide ou ``None`` après un cache miss/corrompu."""
        path = self._path_for(key)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(document, dict) or document.get("key") != asdict(key):
                return None
            values = document.get("values")
            if not isinstance(values, list) or not all(isinstance(value, (int, float)) for value in values):
                return None
            return normalize_embedding(values)
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def put(self, key: EmbeddingCacheKey, vector: EmbeddingVector) -> None:
        """Persiste atomiquement un vecteur normalisé, sans jamais écrire de pixels."""
        normalized = normalize_embedding(vector)
        self._directory.mkdir(parents=True, exist_ok=True)
        destination = self._path_for(key)
        temporary = self._directory / f".{key.digest}.{uuid4().hex}.tmp"
        payload = {"key": asdict(key), "values": normalized}
        try:
            temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
            os.replace(temporary, destination)
        except OSError as error:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise RuntimeError(f"Impossible d'écrire le cache d'embeddings : {destination}") from error

    def reference_for(self, key: EmbeddingCacheKey) -> Path:
        """Expose la référence locale à stocker dans le dataset, sans écrire de donnée."""
        return self._path_for(key)

    @staticmethod
    def load_reference(reference: str | Path) -> EmbeddingVector:
        """Lit un embedding référencé par le dataset et valide sa normalisation."""
        path = Path(reference)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            values = document.get("values") if isinstance(document, dict) else None
            if not isinstance(values, list) or not all(isinstance(value, (int, float)) for value in values):
                raise ValueError("vecteur absent ou invalide")
            return normalize_embedding(values)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Impossible de lire l'embedding local : {path}") from error

    def _path_for(self, key: EmbeddingCacheKey) -> Path:
        return self._directory / f"{key.digest}.json"
