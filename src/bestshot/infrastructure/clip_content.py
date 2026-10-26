"""CLIP image/texte local : seul le gestionnaire explicite accède au réseau."""

from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image, ImageOps

from bestshot.domain.candidate_frame import PreviewImage
from bestshot.domain.photo_pool import ClipContentSettings, PoolContent

MODEL_FILES = (
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "pytorch_model.bin",
)


class ClipContentModelManager:
    def installed(self, settings: ClipContentSettings) -> bool:
        return all((settings.cache_dir / filename).is_file() for filename in MODEL_FILES)

    def download(self, settings: ClipContentSettings) -> Path:
        """Ne télécharge que les poids et configurations du dépôt officiel choisi."""
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise RuntimeError("Installez l'extra : pip install -e '.[pool]'.") from error
        snapshot_download(
            settings.repo_id,
            revision=settings.revision,
            local_dir=settings.cache_dir,
            allow_patterns=list(MODEL_FILES),
        )
        return settings.cache_dir


class LocalClipContentScorer:
    """Compare le cadre entier aux descriptions, sans critère d'orientation du visage."""

    def __init__(self, settings: ClipContentSettings) -> None:
        if not ClipContentModelManager().installed(settings):
            raise RuntimeError("Modèle CLIP absent ; lancez `bestshot models download clip`.")
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as error:
            raise RuntimeError("Installez l'extra : pip install -e '.[pool]'.") from error
        self._torch = torch
        self._device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self._processor = CLIPProcessor.from_pretrained(
            settings.cache_dir, local_files_only=True, trust_remote_code=False
        )
        self._model = cast(
            Any,
            CLIPModel.from_pretrained(
                settings.cache_dir,
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=False,
            ),
        )
        self._model.to(device=self._device)
        self._model.eval()
        self._prompts = (
            settings.people_prompts + settings.landscape_prompts + settings.reject_prompts
        )
        self._people_end = len(settings.people_prompts)
        self._landscape_end = self._people_end + len(settings.landscape_prompts)
        tokens = self._processor(text=list(self._prompts), padding=True, return_tensors="pt")
        with torch.inference_mode():
            features = self._model.get_text_features(**tokens.to(self._device))
            # Transformers 5 peut retourner un objet de sortie plutôt qu'un tenseur.
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output
            self._text_features = features / features.norm(dim=-1, keepdim=True)

    def score(self, preview: PreviewImage) -> PoolContent:
        image = Image.frombytes("RGB", (preview.width, preview.height), preview.rgb_bytes)
        # Le recadrage central par défaut pourrait supprimer les personnes sur les bords.
        image = ImageOps.pad(image, (224, 224), color=(127, 127, 127))
        inputs = self._processor(images=image, return_tensors="pt")
        with self._torch.inference_mode():
            features = self._model.get_image_features(**inputs.to(self._device))
            if not isinstance(features, self._torch.Tensor):
                features = features.pooler_output
            features = features / features.norm(dim=-1, keepdim=True)
            similarities = (features @ self._text_features.T)[0].cpu().numpy()
        return content_from_similarities(
            similarities, self._prompts, self._people_end, self._landscape_end
        )


def content_from_similarities(
    values: np.ndarray,
    prompts: tuple[str, ...],
    people_end: int,
    landscape_end: int,
) -> PoolContent:
    """Agrège les groupes sans softmax : ajouter une description ne change pas l'échelle."""
    positive_index = int(np.argmax(values[:landscape_end]))
    return PoolContent(
        people=float(np.max(values[:people_end])),
        landscape=float(np.max(values[people_end:landscape_end])),
        unwanted=float(np.max(values[landscape_end:])),
        best_prompt=prompts[positive_index],
        unwanted_prompt=prompts[landscape_end + int(np.argmax(values[landscape_end:]))],
    )
