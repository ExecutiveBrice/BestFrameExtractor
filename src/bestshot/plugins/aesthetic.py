"""Plugin CLIP local pour un prédicteur esthétique optionnel."""

from __future__ import annotations

from dataclasses import dataclass
from os import environ
from pathlib import Path
from time import perf_counter
from typing import Protocol

from bestshot.domain.candidate_frame import PreviewImage
from bestshot.domain.composite_score import AestheticScore


@dataclass(frozen=True, slots=True)
class AestheticModelSettings:
    huggingface_token_env: str | None
    clip_repo_id: str
    model_repo_id: str
    model_filename: str
    cache_dir: Path
    raw_score_min: float
    raw_score_max: float


@dataclass(frozen=True, slots=True)
class AestheticModelStatus:
    installed: bool
    cache_path: Path
    message: str


class AestheticScoreProvider(Protocol):
    """Port pour les modèles esthétiques exécutés entièrement localement."""

    def score(self, preview: PreviewImage) -> AestheticScore:
        """Évalue un aperçu sans accès réseau."""


class AestheticModelManager:
    """Télécharge explicitement et met en cache les poids, jamais les images utilisateur."""

    def status(self, settings: AestheticModelSettings) -> AestheticModelStatus:
        model = settings.cache_dir / settings.model_filename
        processor = settings.cache_dir / "clip" / "preprocessor_config.json"
        config = settings.cache_dir / "clip" / "config.json"
        installed = model.is_file() and processor.is_file() and config.is_file()
        message = "installé localement" if installed else "non installé ; lancez `bestshot models download aesthetic`"
        return AestheticModelStatus(installed, settings.cache_dir, message)

    def download(self, settings: AestheticModelSettings) -> AestheticModelStatus:
        """Télécharge explicitement les modèles dans le cache local configuré."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise RuntimeError("Installez l'extra optionnel : pip install -e '.[aesthetic]'.") from error
        settings.cache_dir.mkdir(parents=True, exist_ok=True)
        token = environ.get(settings.huggingface_token_env) if settings.huggingface_token_env else None
        clip_directory = settings.cache_dir / "clip"
        for filename in ("config.json", "preprocessor_config.json"):
            hf_hub_download(settings.clip_repo_id, filename, local_dir=clip_directory, token=token)
        hf_hub_download(
            settings.model_repo_id,
            settings.model_filename,
            local_dir=settings.cache_dir,
            token=token,
        )
        return self.status(settings)


class UnavailableAestheticScorer:
    """Fallback neutre qui laisse le pipeline fonctionner sans modèle optionnel."""

    def __init__(self, status: str) -> None:
        self._status = status

    def score(self, preview: PreviewImage) -> AestheticScore:
        return AestheticScore(0.5, is_neutral=True, status=self._status)


class RsineAestheticScorer:
    """Adaptateur local de ``rsinema/aesthetic-scorer`` fondé sur CLIP ViT-B/32."""

    def __init__(self, settings: AestheticModelSettings) -> None:
        status = AestheticModelManager().status(settings)
        if not status.installed:
            raise RuntimeError(status.message)
        try:
            import torch
            from PIL import Image
            from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel
        except ImportError as error:
            raise RuntimeError("Installez l'extra optionnel : pip install -e '.[aesthetic]'.") from error
        self._torch = torch
        self._image_class = Image
        self._settings = settings
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = CLIPImageProcessor.from_pretrained(
            settings.cache_dir / "clip", local_files_only=True
        )
        state = torch.load(
            settings.cache_dir / settings.model_filename,
            map_location=self._device,
            weights_only=True,
        )
        if not isinstance(state, dict):
            raise TypeError("Le checkpoint esthétique doit être un dictionnaire de poids.")
        vision_config = CLIPVisionConfig.from_pretrained(settings.cache_dir / "clip", local_files_only=True)
        self._backbone = CLIPVisionModel(vision_config).to(self._device)  # type: ignore[arg-type]
        backbone_state = _weights_with_prefix(state, "backbone.")
        self._backbone.load_state_dict(backbone_state, strict=True)
        self._aesthetic_head = torch.nn.Linear(vision_config.hidden_size, 1).to(self._device)
        self._aesthetic_head.load_state_dict(
            _weights_with_prefix(state, "aesthetic_head.0."), strict=True
        )
        self._backbone.eval()
        self._aesthetic_head.eval()

    def score(self, preview: PreviewImage) -> AestheticScore:
        started = perf_counter()
        image = self._image_class.frombytes("RGB", (preview.width, preview.height), preview.rgb_bytes)
        inputs = self._processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device)
        with self._torch.no_grad():
            features = self._backbone(pixel_values=pixel_values).pooler_output
            raw_score = float(self._aesthetic_head(features).reshape(-1)[0].item())
        normalized = _clamp(
            (raw_score - self._settings.raw_score_min)
            / (self._settings.raw_score_max - self._settings.raw_score_min)
        )
        return AestheticScore(
            normalized,
            model_name=self._settings.model_repo_id,
            inference_ms=(perf_counter() - started) * 1_000,
            status=f"{self._device}",
        )


def create_aesthetic_scorer(settings: AestheticModelSettings) -> RsineAestheticScorer | UnavailableAestheticScorer:
    status = AestheticModelManager().status(settings)
    if not status.installed:
        return UnavailableAestheticScorer(status.message)
    try:
        return RsineAestheticScorer(settings)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return UnavailableAestheticScorer(str(error))


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def _weights_with_prefix(state: dict[object, object], prefix: str) -> dict[str, object]:
    weights = {
        key.removeprefix(prefix): value
        for key, value in state.items()
        if isinstance(key, str) and key.startswith(prefix)
    }
    if not weights:
        raise RuntimeError(f"Poids absents pour {prefix[:-1]}.")
    return weights
