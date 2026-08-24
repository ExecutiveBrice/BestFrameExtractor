from typer.testing import CliRunner

from bestshot.cli import app
from bestshot.plugins.aesthetic import AestheticModelManager


def test_models_download_reports_missing_optional_dependency(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def fail_download(self, settings):  # type: ignore[no-untyped-def]
        del self, settings
        raise RuntimeError("Installez l'extra optionnel : pip install -e '.[aesthetic]'.")

    monkeypatch.setattr(AestheticModelManager, "download", fail_download)

    result = CliRunner().invoke(app, ["models", "download", "aesthetic"])

    assert result.exit_code == 1
    assert "Installez l'extra optionnel" in result.output
