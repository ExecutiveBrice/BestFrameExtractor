"""Tests du provider DINOv2 local et entièrement frozen."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from bestshot.domain.preview_image import PreviewImage
from bestshot.embedding.dinov2 import DINOv2EmbeddingProvider, DINOv2ModelManager, DINOv2Settings


def _settings(tmp_path: Path) -> DINOv2Settings:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
        (model_dir / name).touch()
    return DINOv2Settings(
        huggingface_token_env=None,
        repo_id="facebook/dinov2-small",
        revision="main",
        model_version="dino-test-1",
        model_cache_dir=model_dir,
        embedding_cache_dir=tmp_path / "embeddings",
    )


def test_dinov2_provider_freezes_backbone_and_normalizes_its_cls_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    created_models: list[FakeModel] = []

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, path: Path, **kwargs: object) -> FakeProcessor:
            assert path == tmp_path / "model"
            assert kwargs == {"local_files_only": True, "trust_remote_code": False}
            return cls()

        def __call__(self, *, images: object, return_tensors: str) -> dict[str, object]:
            assert return_tensors == "pt"
            assert images.size == (1, 1)
            return {"pixel_values": torch.zeros((1, 3, 1, 1))}

    class FakeModel:
        def __init__(self) -> None:
            self.parameter = torch.nn.Parameter(torch.ones(1), requires_grad=True)
            self.is_eval = False
            self.device: str | None = None

        def to(self, device: str) -> FakeModel:
            self.device = device
            return self

        def parameters(self) -> list[object]:
            return [self.parameter]

        def eval(self) -> FakeModel:
            self.is_eval = True
            return self

        def __call__(self, *, pixel_values: object) -> SimpleNamespace:
            assert self.device == "cpu"
            assert pixel_values is not None
            return SimpleNamespace(last_hidden_state=torch.tensor([[[3.0, 4.0]]]))

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path: Path, **kwargs: object) -> FakeModel:
            assert path == tmp_path / "model"
            assert kwargs == {"local_files_only": True, "trust_remote_code": False}
            model = FakeModel()
            created_models.append(model)
            return model

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoImageProcessor = FakeProcessor
    fake_transformers.AutoModel = FakeAutoModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    provider = DINOv2EmbeddingProvider(_settings(tmp_path))
    embedding = provider.embed(PreviewImage(width=1, height=1, rgb_bytes=b"\x00\x00\x00"))

    assert provider.device == "cpu"
    assert embedding == pytest.approx((0.6, 0.8))
    assert created_models[0].is_eval is True
    assert created_models[0].parameter.requires_grad is False


def test_dinov2_manager_requires_explicit_local_download(tmp_path: Path) -> None:
    settings = DINOv2Settings(
        huggingface_token_env=None,
        repo_id="facebook/dinov2-small",
        revision="main",
        model_version="dino-test-1",
        model_cache_dir=tmp_path / "missing",
        embedding_cache_dir=tmp_path / "embeddings",
    )

    status = DINOv2ModelManager().status(settings)

    assert status.installed is False
    assert "models download embedding" in status.message
