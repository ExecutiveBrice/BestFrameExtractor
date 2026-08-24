"""Tests de fumée du paquet initial."""

from bestshot import __version__
from bestshot.cli import app


def test_package_version() -> None:
    assert __version__ == "0.1.0"


def test_cli_is_initialized() -> None:
    assert app.info.help is not None
