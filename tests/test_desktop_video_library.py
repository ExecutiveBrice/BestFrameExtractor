"""Tests de la découverte locale de vidéos pour la fenêtre desktop."""

from __future__ import annotations

from pathlib import Path

import pytest

from bestshot.desktop.photo_library import PhotoLibraryError, discover_photos
from bestshot.desktop.video_library import VideoLibraryError, discover_videos


def test_discover_videos_lists_supported_files_in_stable_order(tmp_path: Path) -> None:
    (tmp_path / "B.mov").touch()
    (tmp_path / "a.MP4").touch()
    (tmp_path / "notes.txt").touch()
    nested = tmp_path / "album"
    nested.mkdir()
    (nested / "hidden.mkv").touch()

    assert discover_videos(tmp_path) == (
        (tmp_path / "a.MP4").resolve(),
        (tmp_path / "B.mov").resolve(),
    )


def test_discover_videos_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(VideoLibraryError, match="introuvable"):
        discover_videos(tmp_path / "unknown")


def test_discover_photos_lists_supported_files_in_stable_order(tmp_path: Path) -> None:
    (tmp_path / "B.png").touch()
    (tmp_path / "a.JPG").touch()
    (tmp_path / "notes.txt").touch()
    nested = tmp_path / "album"
    nested.mkdir()
    (nested / "hidden.webp").touch()

    assert discover_photos(tmp_path) == (
        (tmp_path / "a.JPG").resolve(),
        (tmp_path / "B.png").resolve(),
    )


def test_discover_photos_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(PhotoLibraryError, match="introuvable"):
        discover_photos(tmp_path / "unknown")
