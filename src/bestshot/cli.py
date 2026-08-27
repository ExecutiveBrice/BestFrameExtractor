"""Interface en ligne de commande du pipeline BestShotAI V2."""

from pathlib import Path
from typing import Annotated

import typer

from bestshot.dataset.preview_cache import PreviewCache, PreviewCacheError
from bestshot.dataset.sqlite_repository import DatasetRepositoryError, SQLiteDatasetRepository
from bestshot.embedding.cache import EmbeddingCache
from bestshot.embedding.dinov2 import DINOv2EmbeddingProvider, DINOv2ModelError, DINOv2ModelManager
from bestshot.embedding.provider import EmbeddingError
from bestshot.infrastructure.config import (
    ConfigurationError,
    load_dataset_settings,
    load_embedding_settings,
    load_presampling_settings,
)
from bestshot.infrastructure.embedding_frames import (
    EmbeddingFrameReadError,
    PyAVCandidatePreviewReader,
)
from bestshot.infrastructure.ffprobe import SubprocessFFprobeRunner
from bestshot.infrastructure.temporal_sampling import PyAVTemporalSamplingBackend
from bestshot.sampling.candidate_generator import CandidateGenerationError, CandidateGenerator
from bestshot.sampling.sharpness_ranker import SharpnessRanker
from bestshot.sampling.temporal_sampler import (
    PresamplingSettings,
    TemporalSampler,
    TemporalSamplingError,
)
from bestshot.services.dataset import (
    format_dataset_stats,
    format_dataset_videos,
    get_dataset_stats,
    list_dataset_videos,
    reset_dataset_labels,
)
from bestshot.services.embeddings import VideoEmbeddingRunner, format_embedding_report
from bestshot.services.presampling import format_presampling_report, generate_presampling_report
from bestshot.video.probe import VideoProbe, VideoProbeError

app = typer.Typer(
    add_completion=False,
    help="Présélection locale de candidates vidéo et embeddings visuels V2.",
)
models_app = typer.Typer(help="Gère les poids d'embedding conservés localement.")
dataset_app = typer.Typer(help="Consulte et administre le dataset local de candidates.")
app.add_typer(models_app, name="models")
app.add_typer(dataset_app, name="dataset")


@models_app.callback(invoke_without_command=True)
def models() -> None:
    """Affiche l'état du modèle DINOv2 local."""
    status = DINOv2ModelManager().status(load_embedding_settings())
    typer.echo(f"Modèle embedding : {status.message} ({status.cache_path})")


@models_app.command("download")
def download_model(name: str = typer.Argument(...)) -> None:
    """Télécharge explicitement les poids d'embedding, sans transmettre de vidéo."""
    if name != "embedding":
        raise typer.BadParameter("Seul le modèle embedding est disponible.")
    try:
        status = DINOv2ModelManager().download(load_embedding_settings())
    except (ConfigurationError, DINOv2ModelError, OSError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Modèle embedding : {status.message} ({status.cache_path})")


@app.command()
def presample(
    video: Annotated[
        Path,
        typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True),
    ],
) -> None:
    """Génère les candidates V2 par fenêtres temporelles, sans score esthétique."""
    try:
        settings = load_presampling_settings()
        report = generate_presampling_report(
            video,
            VideoProbe(SubprocessFFprobeRunner()),
            _create_candidate_generator(settings),
        )
    except (
        CandidateGenerationError,
        ConfigurationError,
        TemporalSamplingError,
        VideoProbeError,
        ValueError,
    ) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    typer.echo(format_presampling_report(report))


@app.command()
def embeddings(
    video: Annotated[
        Path,
        typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True),
    ],
) -> None:
    """Calcule localement les embeddings DINOv2 des candidates V2 et les met en cache."""
    try:
        presampling_settings = load_presampling_settings()
        embedding_settings = load_embedding_settings()
        report = VideoEmbeddingRunner(
            _create_candidate_generator(presampling_settings),
            PyAVCandidatePreviewReader(),
            DINOv2EmbeddingProvider(embedding_settings),
            EmbeddingCache(embedding_settings.embedding_cache_dir),
            presampling_settings.analysis_max_width,
            _create_dataset_repository(),
            PreviewCache(load_dataset_settings().preview_cache_dir),
        ).run(video)
    except (
        CandidateGenerationError,
        ConfigurationError,
        DINOv2ModelError,
        EmbeddingError,
        EmbeddingFrameReadError,
        DatasetRepositoryError,
        TemporalSamplingError,
        PreviewCacheError,
        ValueError,
    ) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    typer.echo(format_embedding_report(report))


@app.command("desktop")
def desktop() -> None:
    """Ouvre l'interface locale d'analyse et d'export des candidates."""
    from bestshot.desktop.application import DesktopApplicationError, run_desktop_application

    try:
        raise typer.Exit(code=run_desktop_application())
    except DesktopApplicationError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error


@dataset_app.command("stats")
def dataset_stats() -> None:
    """Affiche les compteurs locaux, en distinguant KEEP, REJECT et SKIP."""
    try:
        typer.echo(format_dataset_stats(get_dataset_stats(_create_dataset_repository())))
    except (ConfigurationError, DatasetRepositoryError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error


@dataset_app.command("reset-labels")
def dataset_reset_labels() -> None:
    """Retire les labels personnels sans supprimer les candidates ni les caches."""
    try:
        count = reset_dataset_labels(_create_dataset_repository())
    except (ConfigurationError, DatasetRepositoryError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"{count} label(s) réinitialisé(s) en SKIP.")


@dataset_app.command("videos")
def dataset_videos() -> None:
    """Liste les vidéos locales déjà ingérées et leurs labels de candidates."""
    try:
        typer.echo(format_dataset_videos(list_dataset_videos(_create_dataset_repository())))
    except (ConfigurationError, DatasetRepositoryError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error


def _create_candidate_generator(settings: PresamplingSettings) -> CandidateGenerator:
    """Assemble les composants du présampling temporel V2."""
    return CandidateGenerator(
        TemporalSampler(PyAVTemporalSamplingBackend(), settings),
        SharpnessRanker(),
        settings,
    )


def _create_dataset_repository() -> SQLiteDatasetRepository:
    """Ouvre et migre le dataset SQLite local défini dans la configuration."""
    return SQLiteDatasetRepository(load_dataset_settings().database_path)
