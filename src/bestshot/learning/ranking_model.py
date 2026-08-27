"""Head linéaire entraînable au-dessus d'embeddings visuels explicitement figés."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bestshot.embedding.provider import normalize_embedding


class RankingModelError(RuntimeError):
    """Le head personnel de ranking ne peut pas être utilisé localement."""


class LinearRankingModel:
    """Une couche linéaire : le backbone DINOv2 n'est jamais chargé ni entraîné ici."""

    model_type = "linear"

    def __init__(self, embedding_dimension: int, device: str | None = None) -> None:
        if embedding_dimension <= 0:
            raise RankingModelError("La dimension d'embedding doit être positive.")
        torch = _torch()
        self._torch = torch
        self.embedding_dimension = embedding_dimension
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._head: Any = torch.nn.Linear(embedding_dimension, 1).to(self.device)

    def score(self, embedding: Sequence[float]) -> float:
        """Retourne un score personnel dont l'échelle n'est utile qu'au modèle courant."""
        vector = normalize_embedding(embedding)
        if len(vector) != self.embedding_dimension:
            raise RankingModelError("La dimension de l'embedding ne correspond pas au modèle.")
        with self._torch.inference_mode():
            values = self._torch.tensor(vector, dtype=self._torch.float32, device=self.device).unsqueeze(0)
            return float(self._head(values).squeeze().item())

    def score_tensor(self, values: Any) -> Any:
        """Applique le seul head entraînable aux tenseurs d'embeddings déjà normalisés."""
        return self._head(values).squeeze(-1)

    def train(self) -> None:
        self._head.train()

    def eval(self) -> None:
        self._head.eval()

    def parameters(self) -> Any:
        """Expose uniquement les paramètres du head, jamais ceux de DINOv2."""
        return self._head.parameters()

    def state_dict(self) -> Any:
        return self._head.state_dict()

    def load_state_dict(self, state_dict: Any) -> None:
        self._head.load_state_dict(state_dict)

    def save(self, path: Path) -> None:
        """Écrit exclusivement les poids du head personnel."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self._torch.save(
            {
                "format_version": 1,
                "model_type": self.model_type,
                "embedding_dimension": self.embedding_dimension,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> LinearRankingModel:
        """Recharge un head local sérialisé, sans aucun accès réseau."""
        torch = _torch()
        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            payload = torch.load(path, map_location=selected_device, weights_only=True)
            dimension = int(payload["embedding_dimension"])
            if payload.get("model_type") != cls.model_type:
                raise ValueError("type de modèle inattendu")
            model = cls(dimension, selected_device)
            model.load_state_dict(payload["state_dict"])
            model.eval()
            return model
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
            raise RankingModelError(f"Impossible de charger le modèle de ranking : {path}") from error


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RankingModelError("Installez l'extra : pip install -e '.[embedding]'.") from error
    return torch
