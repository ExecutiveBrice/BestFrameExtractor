"""Plugin CLIP local pour un prédicteur esthétique optionnel."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from bestshot.domain.candidate_frame import PreviewImage
from bestshot.domain.composite_score import AestheticScore


@dataclass(frozen=True, slots=True)
class AestheticModelSettings:
    clip_repo_id: str
    predictor_repo_id: str
    predictor_filename: str
    cache_dir: Path
    embedding_dimension: int
    raw_score_min: float
    raw_score_max: float


@dataclass(frozen=True, slots=True)
class AestheticModelStatus:
    installed: bool
    cache_path: Path
    message: str


class AestheticModelManager:
    """Télécharge explicitement et met en cache les poids, jamais les images utilisateur."""

    def status(self, settings: AestheticModelSettings) -> AestheticModelStatus:
        predictor = settings.cache_dir / settings.predictor_filename
        installed = predictor.is_file() and (settings.cache_dir / "clip").is_dir()
        message = "installé localement" if installed else "non installé ; lancez `bestshot models download aesthetic`"
        return AestheticModelStatus(installed, settings.cache_dir, message)

    def download(self, settings: AestheticModelSettings) -> AestheticModelStatus:
        """Télécharge explicitement les modèles dans le cache local configuré."""
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
        except ImportError as error:
            raise RuntimeError("Installez l'extra optionnel : pip install -e '.[aesthetic]'.") from error
        settings.cache_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(settings.clip_repo_id, local_dir=settings.cache_dir / "clip")
        hf_hub_download(
            settings.predictor_repo_id,
            settings.predictor_filename,
            local_dir=settings.cache_dir,
        )
        return self.status(settings)


class UnavailableAestheticScorer:
    """Fallback neutre qui laisse le pipeline fonctionner sans modèle optionnel."""

    def __init__(self, status: str) -> None:
        self._status = status

    def score(self, preview: PreviewImage) -> AestheticScore:
        return AestheticScore(0.5, is_neutral=True, status=self._status)


class ClipAestheticScorer:
    """CLIP + tête linéaire esthétique, tous deux chargés depuis le cache local."""

    def __init__(self, settings: AestheticModelSettings) -> None:
        status = AestheticModelManager().status(settings)
        if not status.installed:
            raise RuntimeError(status.message)
        try:
            import torch
            from PIL import Image
            from transformers import CLIPImageProcessor, CLIPModel
        except ImportError as error:
            raise RuntimeError("Installez l'extra optionnel : pip install -e '.[aesthetic]'.") from error
        self._torch = torch
        self._image_class = Image
        self._settings = settings
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = CLIPModel.from_pretrained(settings.cache_dir / "clip", local_files_only=True).to(self._device)
        self._processor = CLIPImageProcessor.from_pretrained(settings.cache_dir / "clip", local_files_only=True)
        self._predictor = torch.nn.Linear(settings.embedding_dimension, 1).to(self._device)
        state = torch.load(settings.cache_dir / settings.predictor_filename, map_location=self._device)
        self._predictor.load_state_dict(state.get("state_dict", state))
        self._model.eval()
        self._predictor.eval()

    def score(self, preview: PreviewImage) -> AestheticScore:
        started = perf_counter()
        image = self._image_class.frombytes("RGB", (preview.width, preview.height), preview.rgb_bytes)
        inputs = self._processor(images=image, return_tensors="pt").to(self._device)
        with self._torch.no_grad():
            embedding = self._model.get_image_features(**inputs)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            raw_score = float(self._predictor(embedding).item())
        normalized = _clamp(
            (raw_score - self._settings.raw_score_min)
            / (self._settings.raw_score_max - self._settings.raw_score_min)
        )
        return AestheticScore(
            normalized,
            model_name=self._settings.clip_repo_id,
            inference_ms=(perf_counter() - started) * 1_000,
            status=f"{self._device}",
        )


def create_aesthetic_scorer(settings: AestheticModelSettings) -> ClipAestheticScorer | UnavailableAestheticScorer:
    status = AestheticModelManager().status(settings)
    if not status.installed:
        return UnavailableAestheticScorer(status.message)
    try:
        return ClipAestheticScorer(settings)
    except RuntimeError as error:
        return UnavailableAestheticScorer(str(error))


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))
