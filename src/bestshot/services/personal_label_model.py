"""Tête binaire locale entraînée sur les labels KEEP/REJECT et embeddings DINO figés."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from bestshot.dataset.labels import FrameLabel
from bestshot.dataset.repository import DatasetRepository
from bestshot.embedding.cache import EmbeddingCache
from bestshot.embedding.provider import EmbeddingVector, normalize_embedding

PERSONAL_LABEL_MODELS_DIRECTORY = Path(".bestshot/models/personal-labels")


class PersonalLabelModelError(RuntimeError):
    """La tête locale de décision ne peut pas être entraînée ou rechargée."""


@dataclass(frozen=True, slots=True)
class PersonalLabelModelSettings:
    """Réglages de l'optimisation de la seule tête binaire entraînable."""

    epochs: int = 150
    learning_rate: float = 0.01

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.learning_rate <= 0:
            raise ValueError("Le nombre d'epochs et le taux d'apprentissage doivent être positifs.")


@dataclass(frozen=True, slots=True)
class LabelModelTrainingReport:
    """Indicateurs locaux de l'entraînement à partir des choix personnels."""

    keep_count: int
    reject_count: int
    embedding_dimension: int
    device: str


class PersonalLabelModel:
    """Couche linéaire binaire au-dessus d'embeddings DINO déjà calculés et frozen."""

    def __init__(self, embedding_dimension: int, device: str | None = None) -> None:
        if embedding_dimension <= 0:
            raise PersonalLabelModelError("La dimension des embeddings doit être positive.")
        torch = _torch()
        self._torch = torch
        self.embedding_dimension = embedding_dimension
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._head: Any = torch.nn.Linear(embedding_dimension, 1).to(self.device)

    def logits(self, embeddings: Any) -> Any:
        """Applique uniquement la tête locale ; DINO n'est pas chargé ici."""
        return self._head(embeddings).squeeze(-1)

    def predict_keep(self, embedding: EmbeddingVector) -> bool:
        """Décide KEEP lorsque la probabilité locale prédite atteint 50 %."""
        vector = normalize_embedding(embedding)
        if len(vector) != self.embedding_dimension:
            raise PersonalLabelModelError("La dimension de l'embedding ne correspond pas au modèle.")
        with self._torch.inference_mode():
            values = self._torch.tensor(vector, dtype=self._torch.float32, device=self.device).unsqueeze(0)
            return bool(self._torch.sigmoid(self.logits(values)).item() >= 0.5)

    def train(self) -> None:
        self._head.train()

    def eval(self) -> None:
        self._head.eval()

    def parameters(self) -> Any:
        return self._head.parameters()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._torch.save(
            {"format_version": 1, "embedding_dimension": self.embedding_dimension, "state_dict": self._head.state_dict()},
            path,
        )


class PersonalLabelModelTrainer:
    """Entraîne et persiste la tête binaire à partir de KEEP/REJECT uniquement."""

    def __init__(
        self,
        repository: DatasetRepository,
        settings: PersonalLabelModelSettings | None = None,
        models_directory: Path = PERSONAL_LABEL_MODELS_DIRECTORY,
    ) -> None:
        self._repository = repository
        self._settings = settings or PersonalLabelModelSettings()
        self._models_directory = models_directory

    def train_and_save(self) -> tuple[PersonalLabelModel, LabelModelTrainingReport]:
        examples = self._examples()
        keep_count = sum(label is FrameLabel.KEEP for _, label in examples)
        reject_count = sum(label is FrameLabel.REJECT for _, label in examples)
        if keep_count == 0 or reject_count == 0:
            raise PersonalLabelModelError(
                "L'apprentissage IA requiert au moins une candidate ACCEPTÉE et une REJETÉE."
            )
        dimension = len(examples[0][0])
        if any(len(embedding) != dimension for embedding, _ in examples):
            raise PersonalLabelModelError("Les embeddings de labels n'ont pas une dimension homogène.")
        model = PersonalLabelModel(dimension)
        torch = _torch()
        values = torch.tensor([embedding for embedding, _ in examples], dtype=torch.float32, device=model.device)
        targets = torch.tensor(
            [1.0 if label is FrameLabel.KEEP else 0.0 for _, label in examples],
            dtype=torch.float32,
            device=model.device,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self._settings.learning_rate)
        loss_function = torch.nn.BCEWithLogitsLoss()
        for _ in range(self._settings.epochs):
            model.train()
            optimizer.zero_grad()
            loss = loss_function(model.logits(values), targets)
            loss.backward()
            optimizer.step()
        model.eval()
        self._save(model, keep_count, reject_count)
        return model, LabelModelTrainingReport(keep_count, reject_count, dimension, model.device)

    def _examples(self) -> tuple[tuple[EmbeddingVector, FrameLabel], ...]:
        examples: list[tuple[EmbeddingVector, FrameLabel]] = []
        for summary in self._repository.list_videos():
            video = summary.video
            if video.id is None:
                continue
            for frame in self._repository.list_frames_for_video(video.id):
                if frame.label is FrameLabel.SKIP:
                    continue
                examples.append((EmbeddingCache.load_reference(frame.embedding_reference), frame.label))
        return tuple(examples)

    def _save(self, model: PersonalLabelModel, keep_count: int, reject_count: int) -> None:
        self._models_directory.mkdir(parents=True, exist_ok=True)
        model_path = self._models_directory / "model.pt"
        metadata_path = self._models_directory / "metadata.json"
        model.save(model_path)
        payload = {
            "format_version": 1,
            "embedding_dimension": model.embedding_dimension,
            "keep_count": keep_count,
            "reject_count": reject_count,
            "device": model.device,
        }
        temporary = metadata_path.with_name(f".{metadata_path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            os.replace(temporary, metadata_path)
        except OSError as error:
            temporary.unlink(missing_ok=True)
            raise PersonalLabelModelError("Impossible de persister le modèle personnel local.") from error


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise PersonalLabelModelError("Installez l'extra : pip install -e '.[embedding]'.") from error
    return torch
