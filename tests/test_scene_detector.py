"""Tests unitaires du cas d'usage de détection de scènes."""

import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

from pytest import MonkeyPatch

from bestshot.video.scene_detector import (
    PySceneDetectBackend,
    SceneDetectionBackend,
    SceneDetector,
    SceneDetectorSettings,
)


class FakeSceneDetectionBackend(SceneDetectionBackend):
    def __init__(self, boundaries: Sequence[tuple[float, float]]) -> None:
        self.boundaries = boundaries
        self.calls: list[tuple[Path, SceneDetectorSettings]] = []

    def detect(
        self, video_path: Path, settings: SceneDetectorSettings
    ) -> Sequence[tuple[float, float]]:
        self.calls.append((video_path, settings))
        return self.boundaries


def test_scene_detector_creates_one_indexed_domain_scenes() -> None:
    settings = SceneDetectorSettings(3.0, 15, 2, 15.0)
    backend = FakeSceneDetectionBackend(((0.0, 1.5), (1.5, 4.0)))
    detector = SceneDetector(backend, settings)

    scenes = detector.detect(Path("family.mp4"))

    assert [(scene.index, scene.duration) for scene in scenes] == [(1, 1.5), (2, 2.5)]
    assert backend.calls == [(Path("family.mp4"), settings)]


def test_scene_detector_never_returns_a_negative_duration() -> None:
    detector = SceneDetector(
        FakeSceneDetectionBackend(((3.0, 1.0),)),
        SceneDetectorSettings(3.0, 15, 2, 15.0),
    )

    assert detector.detect(Path("family.mp4"))[0].duration == 0.0


def test_pyscenedetect_backend_uses_adaptive_detector(monkeypatch: MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeAdaptiveDetector:
        def __init__(self, **parameters: object) -> None:
            captured["parameters"] = parameters

    class FakeTimecode:
        def __init__(self, seconds: float) -> None:
            self._seconds = seconds

        def get_seconds(self) -> float:
            return self._seconds

    class FakeSceneManager:
        def add_detector(self, detector: FakeAdaptiveDetector) -> None:
            captured["detector"] = detector

        def detect_scenes(self, video: str) -> None:
            captured["video"] = video

        def get_scene_list(self) -> list[tuple[FakeTimecode, FakeTimecode]]:
            return [(FakeTimecode(0.0), FakeTimecode(2.0))]

    fake_module = ModuleType("scenedetect")
    fake_module.AdaptiveDetector = FakeAdaptiveDetector
    fake_module.SceneManager = FakeSceneManager
    fake_module.open_video = lambda path: f"video:{path}"
    monkeypatch.setitem(sys.modules, "scenedetect", fake_module)
    settings = SceneDetectorSettings(4.0, 20, 3, 12.0)

    boundaries = PySceneDetectBackend().detect(Path("family.mp4"), settings)

    assert boundaries == [(0.0, 2.0)]
    assert captured["parameters"] == {
        "adaptive_threshold": 4.0,
        "min_scene_len": 20,
        "window_width": 3,
        "min_content_val": 12.0,
    }
    assert captured["video"] == "video:family.mp4"


def test_pyscenedetect_backend_uses_whole_video_when_no_cut_is_detected(
    monkeypatch: MonkeyPatch,
) -> None:
    class FakeTimecode:
        def get_seconds(self) -> float:
            return 4.0

    class FakeVideo:
        duration = FakeTimecode()

    class FakeAdaptiveDetector:
        def __init__(self, **parameters: object) -> None:
            del parameters

    class FakeSceneManager:
        def add_detector(self, detector: FakeAdaptiveDetector) -> None:
            del detector

        def detect_scenes(self, video: FakeVideo) -> None:
            del video

        def get_scene_list(self) -> list[tuple[FakeTimecode, FakeTimecode]]:
            return []

    fake_module = ModuleType("scenedetect")
    fake_module.AdaptiveDetector = FakeAdaptiveDetector
    fake_module.SceneManager = FakeSceneManager
    fake_module.open_video = lambda path: FakeVideo()
    monkeypatch.setitem(sys.modules, "scenedetect", fake_module)

    assert PySceneDetectBackend().detect(Path("family.mp4"), SceneDetectorSettings(3.0, 15, 2, 15.0)) == [
        (0.0, 4.0)
    ]
