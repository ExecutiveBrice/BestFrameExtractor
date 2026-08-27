"""Provider local DINOv2 ViT-S/14, téléchargé explicitement puis entièrement frozen."""

from __future__ import annotations

from dataclasses import dataclass
from os import environ
from pathlib import Path

from bestshot.domain.preview_image import PreviewImage
from bestshot.embedding.provider import EmbeddingVector, normalize_embedding

MODEL_NAME = "DINOv2 ViT-S/14"


@dataclass(frozen=True, slots=True)
class DINOv2Settings:
    """Localisation et version explicite du premier backbone d'embeddings."""

    huggingface_token_env: str | None
    repo_id: str
    revision: str
    model_version: str
    model_cache_dir: Path
    embedding_cache_dir: Path


@dataclass(frozen=True, slots=True)
class DINOv2ModelStatus:
    """État local des artefacts nécessaires à l'inférence DINOv2."""

    installed: bool
    cache_path: Path
    message: str


class DINOv2ModelError(RuntimeError):
    """Le modèle DINOv2 local est absent, incomplet ou inexploitable."""


class DINOv2ModelManager:
    """Télécharge explicitement les poids DINOv2, sans jamais transmettre d'image."""

    def status(self, settings: DINOv2Settings) -> DINOv2ModelStatus:
        config = settings.model_cache_dir / "config.json"
        preprocessor = settings.model_cache_dir / "preprocessor_config.json"
        weights = tuple(settings.model_cache_dir.glob("*.safetensors")) + tuple(
            settings.model_cache_dir.glob("*.bin")
        )
        installed = config.is_file() and preprocessor.is_file() and bool(weights)
        message = (
            "installé localement"
            if installed
            else "non installé ; lancez `bestshot models download embedding`"
        )
        return DINOv2ModelStatus(installed, settings.model_cache_dir, message)

    def download(self, settings: DINOv2Settings) -> DINOv2ModelStatus:
        """Télécharge explicitement le snapshot des poids dans le cache configuré."""
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise DINOv2ModelError("Installez l'extra : pip install -e '.[embedding]'.") from error
        settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
        token = environ.get(settings.huggingface_token_env) if settings.huggingface_token_env else None
        try:
            snapshot_download(
                repo_id=settings.repo_id,
                revision=settings.revision,
                local_dir=settings.model_cache_dir,
                token=token,
            )
        except Exception as error:
            raise DINOv2ModelError(f"Impossible de télécharger DINOv2 : {error}") from error
        return self.status(settings)


class DINOv2EmbeddingProvider:
    """Infère avec DINOv2 local en mode évaluation, sans gradient ni entraînement."""

    def __init__(self, settings: DINOv2Settings) -> None:
        status = DINOv2ModelManager().status(settings)
        if not status.installed:
            raise DINOv2ModelError(status.message)
        try:
            import torch
            from PIL import Image
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as error:
            raise DINOv2ModelError("Installez l'extra : pip install -e '.[embedding]'.") from error

        self._torch = torch
        self._image_class = Image
        self._settings = settings
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self._processor = AutoImageProcessor.from_pretrained(
                settings.model_cache_dir,
                local_files_only=True,
                trust_remote_code=False,
            )  # type: ignore[no-untyped-call]
            self._backbone = AutoModel.from_pretrained(
                settings.model_cache_dir,
                local_files_only=True,
                trust_remote_code=False,
            ).to(self._device)
        except Exception as error:
            raise DINOv2ModelError(f"Impossible de charger DINOv2 localement : {error}") from error
        for parameter in self._backbone.parameters():
            parameter.requires_grad_(False)
        self._backbone.eval()

    @property
    def device(self) -> str:
        """Périphérique sélectionné, CUDA lorsqu'il est disponible sinon CPU."""
        return self._device

    @property
    def model_name(self) -> str:
        """Nom lisible du backbone figé."""
        return MODEL_NAME

    @property
    def model_version(self) -> str:
        """Version qui invalide automatiquement le cache lors d'un changement de poids."""
        return f"{self._settings.repo_id}@{self._settings.revision}:{self._settings.model_version}"

    def embed(self, image: PreviewImage) -> EmbeddingVector:
        """Retourne le token CLS DINOv2 normalisé, sans construire de graphe de gradients."""
        if image.width <= 0 or image.height <= 0 or len(image.rgb_bytes) != image.width * image.height * 3:
            raise DINOv2ModelError("L'aperçu RGB est invalide pour DINOv2.")
        pil_image = self._image_class.frombytes("RGB", (image.width, image.height), image.rgb_bytes)
        try:
            inputs = self._processor(images=pil_image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self._device)
            with self._torch.inference_mode():
                output = self._backbone(pixel_values=pixel_values)
            values = output.last_hidden_state[0, 0].detach().float().cpu().tolist()
        except Exception as error:
            raise DINOv2ModelError(f"Impossible de calculer l'embedding DINOv2 : {error}") from error
        if not isinstance(values, list):
            raise DINOv2ModelError("DINOv2 n'a pas retourné de vecteur exploitable.")
        return normalize_embedding([float(value) for value in values])
