"""Le pool doit accepter les variantes utiles et rejeter sans compensation le flou."""

import json
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from bestshot.domain.candidate_frame import PreviewImage
from bestshot.domain.photo_pool import PhotoPoolPolicy, PoolContent, PoolFrame, PoolQuality
from bestshot.infrastructure.clip_content import (
    ClipContentModelManager,
    LocalClipContentScorer,
    content_from_similarities,
)
from bestshot.infrastructure.config import (
    ConfigurationError,
    load_clip_content_settings,
    load_photo_pool_settings,
)
from bestshot.infrastructure.photo_pool import (
    LocalPoolImageWriter,
    LocalPoolRepository,
    OpenCVPoolQualityScorer,
)
from bestshot.services.photo_pool import PhotoPoolRunner

GOOD_QUALITY = PoolQuality(150.0, 0.10, 80.0, 40.0, 120.0)
PEOPLE = PoolContent(0.32, 0.20, 0.18, "a person seen from behind")
LANDSCAPE = PoolContent(0.18, 0.32, 0.18, "a mountain landscape")
UNCERTAIN = PoolContent(0.24, 0.18, 0.24, "a person")
GROUND = PoolContent(0.16, 0.18, 0.32, "ground")


@pytest.mark.parametrize("content", [PEOPLE, LANDSCAPE])
def test_people_and_landscape_are_alternative_conditions(content) -> None:
    policy = PhotoPoolPolicy(load_photo_pool_settings())
    assert policy.decide(GOOD_QUALITY, content).status == "accepted"


@pytest.mark.parametrize(
    "quality",
    [
        replace(GOOD_QUALITY, sharpness=1),
        replace(GOOD_QUALITY, detail_ratio=0.0001),
        replace(GOOD_QUALITY, native_sharpness=1),
        replace(GOOD_QUALITY, contrast=1),
        replace(GOOD_QUALITY, luminance=0),
        replace(GOOD_QUALITY, luminance=255),
        replace(GOOD_QUALITY, sharpness=float("nan")),
    ],
)
def test_content_never_compensates_for_bad_quality(quality) -> None:
    policy = PhotoPoolPolicy(load_photo_pool_settings())
    assert policy.decide(quality, PEOPLE).status == "rejected"


def test_ambiguous_content_is_separate_from_the_pool() -> None:
    policy = PhotoPoolPolicy(load_photo_pool_settings())
    assert policy.decide(GOOD_QUALITY, UNCERTAIN).status == "review"
    assert policy.decide(GOOD_QUALITY, GROUND).status == "rejected"
    assert policy.decide(GOOD_QUALITY, None).status == "rejected"
    assert policy.decide(GOOD_QUALITY, replace(PEOPLE, people=float("nan"))).status == "rejected"


def test_prompts_explicitly_include_front_profile_and_back() -> None:
    prompts = load_clip_content_settings().people_prompts
    assert any("facing the camera" in prompt for prompt in prompts)
    assert any("behind" in prompt for prompt in prompts)
    assert any("side" in prompt or "profile" in prompt for prompt in prompts)


def test_content_uses_cosine_values_and_negative_margin_not_probabilities() -> None:
    result = content_from_similarities(
        np.array([0.31, 0.34, 0.22, 0.35]), ("front", "back", "landscape", "ground"), 2, 3
    )
    assert result.people == pytest.approx(0.34)
    assert result.margin == pytest.approx(-0.01)
    assert result.best_prompt == "back"
    assert result.unwanted_prompt == "ground"


def test_missing_clip_model_never_triggers_a_download(tmp_path, monkeypatch) -> None:
    settings = replace(load_clip_content_settings(), cache_dir=tmp_path)

    def unexpected_download(*args, **kwargs):
        pytest.fail("Téléchargement implicite interdit")

    monkeypatch.setattr(ClipContentModelManager, "download", unexpected_download)
    with pytest.raises(RuntimeError, match="models download clip"):
        LocalClipContentScorer(settings)


@pytest.mark.parametrize(
    "change",
    [
        {"fps": 0},
        {"fps": float("nan")},
        {"minimum_sharpness": float("inf")},
        {"analysis_width": 1},
        {"jpeg_quality": 101},
        {"review_margin": 0.8},
    ],
)
def test_invalid_pool_settings_are_rejected(change) -> None:
    with pytest.raises(ValueError):
        replace(load_photo_pool_settings(), **change)


def test_missing_config_section_is_explicit(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    with pytest.raises(ConfigurationError, match="photo_pool"):
        load_photo_pool_settings(config)


def _image() -> PreviewImage:
    image = Image.new("RGB", (160, 90), "green")
    return PreviewImage(image.width, image.height, image.tobytes())


class FakeSource:
    def frames(self, video_path, fps, start=0, duration=None, should_stop=None):
        if video_path.stem == "bad":
            raise RuntimeError("vidéo illisible")
        for i in range(6):
            if should_stop and should_stop():
                break
            yield PoolFrame(start + i / fps, i, _image())


class FakeQuality:
    def __init__(self, quality=GOOD_QUALITY):
        self.quality = quality

    def score(self, image):
        return self.quality


class FakeContent:
    def __init__(self, content=PEOPLE):
        self.calls = 0
        self.content = content

    def score(self, preview):
        self.calls += 1
        return self.content


def _runner(quality=None, content=None):
    return PhotoPoolRunner(
        FakeSource(),
        quality or FakeQuality(),
        content or FakeContent(),
        LocalPoolImageWriter(),
        LocalPoolRepository(),
        load_photo_pool_settings(),
        {"repo_id": "test", "revision": "local"},
    )


def test_identical_photos_are_kept_without_a_scene_quota(tmp_path) -> None:
    video = tmp_path / "video.mp4"
    video.touch()
    result = _runner().run(video, None, tmp_path / "photos")
    assert result.exported_count == 6
    assert len(set(result.videos[0].image_paths)) == 6
    assert all(Image.open(p).size == (160, 90) for p in result.videos[0].image_paths)
    manifest = json.loads((result.videos[0].output_directory / "manifest.json").read_text())
    assert manifest["counts"] == {"accepted": 6}
    decisions = (result.videos[0].output_directory / "decisions.jsonl").read_text().splitlines()
    assert json.loads(decisions[1])["timestamp"] == pytest.approx(1 / 3)


def test_blur_is_rejected_before_clip_is_called(tmp_path) -> None:
    video = tmp_path / "video.mp4"
    video.touch()
    content = FakeContent()
    runner = _runner(FakeQuality(replace(GOOD_QUALITY, sharpness=0)), content)
    result = runner.run(video, None, tmp_path / "photos")
    assert result.exported_count == 0
    assert content.calls == 0


def test_uncertain_images_are_exported_outside_the_pool(tmp_path) -> None:
    video = tmp_path / "video.mp4"
    video.touch()
    result = _runner(content=FakeContent(UNCERTAIN)).run(video, None, tmp_path / "photos")
    assert result.exported_count == 0
    destination = result.videos[0].output_directory
    assert len(list((destination / "a_verifier").glob("*.jpg"))) == 6
    assert not (destination / "pool").exists()


def test_stop_preserves_exports_and_marks_manifest_as_partial(tmp_path) -> None:
    video = tmp_path / "video.mp4"
    video.touch()
    stopped = False

    def on_progress(progress):
        nonlocal stopped
        if progress.state == "exported":
            stopped = True

    result = _runner().run(
        video, None, tmp_path / "photos", on_progress=on_progress, should_stop=lambda: stopped
    )
    assert result.cancelled
    assert result.exported_count == 1
    manifest = json.loads((result.videos[0].output_directory / "manifest.json").read_text())
    assert manifest["cancelled"] is True
    assert manifest["counts"] == {"accepted": 1}


def test_unreadable_video_does_not_interrupt_the_batch_and_reruns_preserve_files(tmp_path) -> None:
    source = tmp_path / "videos"
    source.mkdir()
    (source / "bad.mp4").touch()
    (source / "good.mp4").touch()
    runner = _runner()
    first = runner.run(source, None, tmp_path / "photos")
    second = runner.run(source, None, tmp_path / "photos")
    assert len(first.failures) == 1
    assert first.exported_count == 6
    assert second.successes[0].output_directory != first.successes[0].output_directory
    assert first.successes[0].image_paths[0].exists()


def test_pool_rejects_destination_in_source_folder(tmp_path) -> None:
    video = tmp_path / "video.mp4"
    video.touch()
    with pytest.raises(ValueError, match="destination distincte"):
        _runner().run(video, None, tmp_path)


def test_quality_distinguishes_real_detail_from_gaussian_blur() -> None:
    import cv2

    grid = np.indices((240, 320)).sum(axis=0) // 8 % 2 * 255
    sharp = np.repeat(grid[:, :, None], 3, axis=2).astype(np.uint8)
    blur = cv2.GaussianBlur(sharp, (31, 31), 6)
    scorer = OpenCVPoolQualityScorer(960)
    good = scorer.score(PreviewImage(320, 240, sharp.tobytes()))
    bad = scorer.score(PreviewImage(320, 240, blur.tobytes()))
    assert good.sharpness > bad.sharpness * 10
    assert PhotoPoolPolicy(load_photo_pool_settings()).quality_rejection(good) is None
    assert PhotoPoolPolicy(load_photo_pool_settings()).quality_rejection(bad) is not None


def test_native_writer_preserves_dimensions_and_supports_png(tmp_path) -> None:
    target = tmp_path / "photos" / "frame.png"
    image = _image()
    LocalPoolImageWriter().write(image, target, "png", 95)
    with Image.open(target) as saved:
        assert saved.size == (image.width, image.height)
        assert saved.tobytes() == image.rgb_bytes


def test_clip_loading_and_inference_use_only_local_files(tmp_path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from bestshot.infrastructure.clip_content import MODEL_FILES

    for filename in MODEL_FILES:
        (tmp_path / filename).touch()
    settings = replace(
        load_clip_content_settings(),
        cache_dir=tmp_path,
        people_prompts=("front", "profile", "back"),
        landscape_prompts=("landscape",),
        reject_prompts=("ground",),
    )
    calls = []
    processed_images = []

    class Batch(dict):
        def to(self, device):
            return self

    class Processor:
        def __call__(self, text=None, images=None, **kwargs):
            if images is not None:
                processed_images.append(images)
            return Batch()

    class Model:
        def to(self, **kwargs):
            return self

        def eval(self):
            return self

        def get_text_features(self):
            return torch.eye(5)

        def get_image_features(self):
            return torch.tensor([[0.1, 0.2, 0.9, 0.1, 0.0]])

    def processor_load(path, **kwargs):
        calls.append((path, kwargs))
        return Processor()

    def model_load(path, **kwargs):
        calls.append((path, kwargs))
        return Model()

    monkeypatch.setattr(transformers.CLIPProcessor, "from_pretrained", processor_load)
    monkeypatch.setattr(transformers.CLIPModel, "from_pretrained", model_load)
    scorer = LocalClipContentScorer(settings)
    result = scorer.score(_image())
    assert result.best_prompt == "back"
    assert all(path == tmp_path and kwargs["local_files_only"] for path, kwargs in calls)
    assert all(kwargs["trust_remote_code"] is False for _, kwargs in calls)
    assert calls[1][1]["use_safetensors"] is False
    assert processed_images[0].size == (224, 224)
    assert processed_images[0].getpixel((0, 0)) == (127, 127, 127)


def test_explicit_clip_download_is_pinned_and_only_downloads_model_files(
    tmp_path, monkeypatch
) -> None:
    hub = pytest.importorskip("huggingface_hub")
    from bestshot.infrastructure.clip_content import MODEL_FILES

    calls = []

    def download(repo, **kwargs):
        calls.append((repo, kwargs))
        return str(tmp_path)

    monkeypatch.setattr(hub, "snapshot_download", download)
    settings = replace(load_clip_content_settings(), cache_dir=tmp_path)
    assert ClipContentModelManager().download(settings) == tmp_path
    assert calls == [
        (
            settings.repo_id,
            {
                "revision": settings.revision,
                "local_dir": tmp_path,
                "allow_patterns": list(MODEL_FILES),
            },
        )
    ]
