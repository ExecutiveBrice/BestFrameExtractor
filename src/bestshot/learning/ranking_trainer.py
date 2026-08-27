"""Apprentissage RankNet local du head personnel, groupé par vidéo."""

from __future__ import annotations

import json
import math
import os
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from bestshot.domain.preferences import PreferenceChoice
from bestshot.embedding.provider import EmbeddingVector, normalize_embedding
from bestshot.learning.ranking_model import LinearRankingModel, RankingModelError


@dataclass(frozen=True, slots=True)
class PersonalRankingSettings:
    """Hyperparamètres du head linéaire ; DINOv2 reste hors de l'optimiseur."""

    equal_loss_weight: float = 0.5
    learning_rate: float = 0.001
    epochs: int = 100
    weight_decay: float = 0.0001
    validation_ratio: float = 0.20
    early_stopping_patience: int = 10
    equality_margin: float = 0.05
    seed: int = 42

    def __post_init__(self) -> None:
        if self.equal_loss_weight < 0 or self.weight_decay < 0 or self.equality_margin < 0:
            raise ValueError("Les poids et marges doivent être positifs ou nuls.")
        if self.learning_rate <= 0 or self.epochs <= 0 or self.early_stopping_patience <= 0:
            raise ValueError("Le taux, les epochs et la patience doivent être positifs.")
        if not 0 <= self.validation_ratio < 1:
            raise ValueError("Le ratio de validation doit appartenir à [0, 1[.")


@dataclass(frozen=True, slots=True)
class RankingExample:
    """Une préférence enrichie de ses vecteurs et groupes vidéo."""

    first_embedding: EmbeddingVector
    second_embedding: EmbeddingVector
    preference: PreferenceChoice
    first_video_id: int
    second_video_id: int

    def __post_init__(self) -> None:
        if not self.preference.is_usable_for_training:
            raise ValueError("SKIP ne doit jamais devenir un exemple d'entraînement.")
        if self.first_video_id <= 0 or self.second_video_id <= 0:
            raise ValueError("Les groupes vidéo doivent être positifs.")
        first = normalize_embedding(self.first_embedding)
        second = normalize_embedding(self.second_embedding)
        if len(first) != len(second):
            raise ValueError("Les embeddings d'une comparaison doivent avoir la même dimension.")
        object.__setattr__(self, "first_embedding", first)
        object.__setattr__(self, "second_embedding", second)


@dataclass(frozen=True, slots=True)
class RankingMetrics:
    """Métriques interprétables par type de préférence."""

    pairwise_accuracy: float | None
    first_accuracy: float | None
    second_accuracy: float | None
    equality_accuracy: float | None
    comparison_count: int
    first_count: int
    second_count: int
    equal_count: int
    video_count: int
    loss: float | None


@dataclass(frozen=True, slots=True)
class RankingSplit:
    """Split sans fuite de vidéo ; les paires traversant la frontière sont ignorées."""

    train_examples: tuple[RankingExample, ...]
    validation_examples: tuple[RankingExample, ...]
    train_video_ids: tuple[int, ...]
    validation_video_ids: tuple[int, ...]
    excluded_cross_split_count: int


@dataclass(frozen=True, slots=True)
class RankingTrainingResult:
    """Résultat d'entraînement et modèles en mémoire avant sauvegarde versionnée."""

    model: LinearRankingModel
    train_metrics: RankingMetrics
    validation_metrics: RankingMetrics
    split: RankingSplit
    epochs_completed: int
    best_epoch: int


@dataclass(frozen=True, slots=True)
class RankingArtifact:
    """Emplacement versionné et métadonnées du modèle personnel local."""

    directory: Path
    version: str
    model_path: Path
    metadata_path: Path
    metrics_path: Path


class RankingTrainingError(RuntimeError):
    """Le dataset de préférences ne permet pas un entraînement fiable."""


def split_by_video(examples: Sequence[RankingExample], settings: PersonalRankingSettings) -> RankingSplit:
    """Sépare par groupes vidéo et élimine uniquement les paires qui traversent le split."""
    video_ids = sorted({video_id for item in examples for video_id in (item.first_video_id, item.second_video_id)})
    validation_ids: set[int] = set()
    if len(video_ids) >= 2 and settings.validation_ratio > 0:
        validation_count = max(1, round(len(video_ids) * settings.validation_ratio))
        validation_count = min(validation_count, len(video_ids) - 1)
        shuffled = video_ids.copy()
        random.Random(settings.seed).shuffle(shuffled)
        validation_ids = set(shuffled[:validation_count])
    train_ids = set(video_ids) - validation_ids
    train: list[RankingExample] = []
    validation: list[RankingExample] = []
    excluded = 0
    for item in examples:
        first_validation = item.first_video_id in validation_ids
        second_validation = item.second_video_id in validation_ids
        if first_validation != second_validation:
            excluded += 1
        elif first_validation:
            validation.append(item)
        else:
            train.append(item)
    return RankingSplit(
        tuple(train),
        tuple(validation),
        tuple(sorted(train_ids)),
        tuple(sorted(validation_ids)),
        excluded,
    )


class RankingTrainer:
    """Optimise exclusivement le head linéaire à partir de comparaisons locales."""

    def __init__(self, settings: PersonalRankingSettings) -> None:
        self._settings = settings

    def train(self, examples: Sequence[RankingExample]) -> RankingTrainingResult:
        """Exécute une optimisation RankNet, avec arrêt anticipé sur la validation."""
        usable = tuple(item for item in examples if item.preference.is_usable_for_training)
        if not usable:
            raise RankingTrainingError("Aucune préférence FIRST, SECOND ou EQUAL n'est disponible.")
        dimension = len(usable[0].first_embedding)
        if any(len(item.first_embedding) != dimension for item in usable):
            raise RankingTrainingError("Les embeddings du dataset n'ont pas une dimension homogène.")
        torch = _torch()
        torch.manual_seed(self._settings.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._settings.seed)
        split = split_by_video(usable, self._settings)
        if not split.train_examples:
            raise RankingTrainingError("Le split ne contient aucune comparaison d'entraînement.")
        model = LinearRankingModel(dimension)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self._settings.learning_rate,
            weight_decay=self._settings.weight_decay,
        )
        best_state: Any = None
        best_loss = math.inf
        best_epoch = 0
        stale_epochs = 0
        epochs_completed = 0
        for epoch in range(1, self._settings.epochs + 1):
            model.train()
            optimizer.zero_grad()
            loss = self._loss(model, split.train_examples)
            loss.backward()
            optimizer.step()
            epochs_completed = epoch
            validation_loss = self._loss_value(model, split.validation_examples)
            monitored_loss = validation_loss if validation_loss is not None else float(loss.detach().item())
            if monitored_loss < best_loss:
                best_loss = monitored_loss
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self._settings.early_stopping_patience:
                    break
        if best_state is None:
            raise RankingTrainingError("L'entraînement n'a produit aucun état de modèle.")
        model.load_state_dict(best_state)
        model.eval()
        return RankingTrainingResult(
            model=model,
            train_metrics=self._metrics(model, split.train_examples),
            validation_metrics=self._metrics(model, split.validation_examples),
            split=split,
            epochs_completed=epochs_completed,
            best_epoch=best_epoch,
        )

    def _loss(self, model: LinearRankingModel, examples: Sequence[RankingExample]) -> Any:
        first, second, choices = _tensors(examples, model.device)
        first_scores = model.score_tensor(first)
        second_scores = model.score_tensor(second)
        return pairwise_loss(
            first_scores,
            second_scores,
            choices,
            equal_loss_weight=self._settings.equal_loss_weight,
        )

    def _loss_value(self, model: LinearRankingModel, examples: Sequence[RankingExample]) -> float | None:
        if not examples:
            return None
        torch = _torch()
        model.eval()
        with torch.inference_mode():
            return float(self._loss(model, examples).item())

    def _metrics(self, model: LinearRankingModel, examples: Sequence[RankingExample]) -> RankingMetrics:
        if not examples:
            return RankingMetrics(None, None, None, None, 0, 0, 0, 0, 0, None)
        scores = [(model.score(item.first_embedding), model.score(item.second_embedding)) for item in examples]
        correctness: dict[PreferenceChoice, list[bool]] = {
            PreferenceChoice.FIRST: [],
            PreferenceChoice.SECOND: [],
            PreferenceChoice.EQUAL: [],
        }
        for item, (first_score, second_score) in zip(examples, scores, strict=True):
            if item.preference is PreferenceChoice.FIRST:
                correctness[PreferenceChoice.FIRST].append(first_score > second_score)
            elif item.preference is PreferenceChoice.SECOND:
                correctness[PreferenceChoice.SECOND].append(second_score > first_score)
            else:
                correctness[PreferenceChoice.EQUAL].append(
                    abs(first_score - second_score) <= self._settings.equality_margin
                )
        all_values = [value for values in correctness.values() for value in values]
        return RankingMetrics(
            pairwise_accuracy=_accuracy(all_values),
            first_accuracy=_accuracy(correctness[PreferenceChoice.FIRST]),
            second_accuracy=_accuracy(correctness[PreferenceChoice.SECOND]),
            equality_accuracy=_accuracy(correctness[PreferenceChoice.EQUAL]),
            comparison_count=len(examples),
            first_count=len(correctness[PreferenceChoice.FIRST]),
            second_count=len(correctness[PreferenceChoice.SECOND]),
            equal_count=len(correctness[PreferenceChoice.EQUAL]),
            video_count=len({video for item in examples for video in (item.first_video_id, item.second_video_id)}),
            loss=self._loss_value(model, examples),
        )


def save_ranking_artifact(
    result: RankingTrainingResult,
    directory: Path,
    *,
    embedding_model_version: str,
    settings: PersonalRankingSettings,
) -> RankingArtifact:
    """Crée un répertoire de modèle immuable et actualise seulement ``current.json``."""
    directory.mkdir(parents=True, exist_ok=True)
    version = _next_version(directory)
    artifact_directory = directory / version
    artifact_directory.mkdir()
    model_path = artifact_directory / "model.pt"
    result.model.save(model_path)
    metadata_path = artifact_directory / "metadata.json"
    metrics_path = artifact_directory / "metrics.json"
    metadata = {
        "format_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "model_type": result.model.model_type,
        "embedding_model_version": embedding_model_version,
        "embedding_dimension": result.model.embedding_dimension,
        "comparison_count": result.train_metrics.comparison_count + result.validation_metrics.comparison_count,
        "video_count": len(set(result.split.train_video_ids) | set(result.split.validation_video_ids)),
        "settings": asdict(settings),
        "seed": settings.seed,
    }
    metrics = {
        "epochs_completed": result.epochs_completed,
        "best_epoch": result.best_epoch,
        "excluded_cross_split_count": result.split.excluded_cross_split_count,
        "train": asdict(result.train_metrics),
        "validation": asdict(result.validation_metrics),
        "train_video_ids": result.split.train_video_ids,
        "validation_video_ids": result.split.validation_video_ids,
    }
    _write_json(metadata_path, metadata)
    _write_json(metrics_path, metrics)
    _write_json(directory / "current.json", {"version": version, "path": str(artifact_directory)})
    return RankingArtifact(artifact_directory, version, model_path, metadata_path, metrics_path)


def load_current_ranking_model(directory: Path) -> LinearRankingModel:
    """Charge le modèle pointé par le manifeste local, sans écraser aucun historique."""
    try:
        current = json.loads((directory / "current.json").read_text(encoding="utf-8"))
        path = Path(str(current["path"])) / "model.pt"
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise RankingModelError("Aucun modèle personnel courant n'est disponible.") from error
    return LinearRankingModel.load(path)


def pairwise_loss(
    first_scores: Any,
    second_scores: Any,
    choices: Sequence[PreferenceChoice],
    *,
    equal_loss_weight: float,
) -> Any:
    """Calcule les pertes RankNet et d'égalité ; ``SKIP`` est explicitement interdit."""
    if equal_loss_weight < 0:
        raise ValueError("Le poids de perte d'égalité doit être positif ou nul.")
    if len(first_scores) != len(second_scores) or len(first_scores) != len(choices):
        raise ValueError("Scores et préférences doivent avoir la même taille.")
    torch = _torch()
    losses: list[Any] = []
    for index, choice in enumerate(choices):
        difference = first_scores[index] - second_scores[index]
        if choice is PreferenceChoice.FIRST:
            losses.append(-torch.nn.functional.logsigmoid(difference))
        elif choice is PreferenceChoice.SECOND:
            losses.append(-torch.nn.functional.logsigmoid(-difference))
        elif choice is PreferenceChoice.EQUAL:
            losses.append(equal_loss_weight * difference.square())
        else:
            raise ValueError("SKIP ne doit jamais être transmis à la loss pairwise.")
    if not losses:
        raise ValueError("Au moins une préférence utilisable est nécessaire.")
    return torch.stack(losses).mean()


def _tensors(examples: Sequence[RankingExample], device: str) -> tuple[Any, Any, list[PreferenceChoice]]:
    torch = _torch()
    return (
        torch.tensor([item.first_embedding for item in examples], dtype=torch.float32, device=device),
        torch.tensor([item.second_embedding for item in examples], dtype=torch.float32, device=device),
        [item.preference for item in examples],
    )


def _accuracy(values: Sequence[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def _next_version(directory: Path) -> str:
    used = [
        int(path.name.removeprefix("model-"))
        for path in directory.glob("model-*")
        if path.is_dir() and path.name.removeprefix("model-").isdigit()
    ]
    return f"model-{max(used, default=0) + 1:04d}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    except OSError as error:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise RankingTrainingError(f"Impossible d'écrire l'artefact de ranking : {path}") from error


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RankingTrainingError("Installez l'extra : pip install -e '.[embedding]'.") from error
    return torch
