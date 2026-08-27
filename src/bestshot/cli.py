"""Interface en ligne de commande du pipeline BestShotAI V2."""

from pathlib import Path
from typing import Annotated

import typer

from bestshot.dataset.preview_cache import PreviewCache, PreviewCacheError
from bestshot.dataset.sqlite_repository import DatasetRepositoryError, SQLiteDatasetRepository
from bestshot.domain.preview_image import PreviewImage
from bestshot.embedding.cache import EmbeddingCache
from bestshot.embedding.dinov2 import DINOv2EmbeddingProvider, DINOv2ModelError, DINOv2ModelManager
from bestshot.embedding.provider import EmbeddingError
from bestshot.infrastructure.config import (
    ConfigurationError,
    load_dataset_settings,
    load_embedding_settings,
    load_pair_generation_settings,
    load_personal_ranking_settings,
    load_presampling_settings,
)
from bestshot.infrastructure.embedding_frames import (
    EmbeddingFrameReadError,
    PyAVCandidatePreviewReader,
)
from bestshot.infrastructure.ffprobe import SubprocessFFprobeRunner
from bestshot.infrastructure.temporal_sampling import PyAVTemporalSamplingBackend
from bestshot.learning.ranking_model import RankingModelError
from bestshot.learning.ranking_trainer import (
    RankingTrainingError,
    load_current_ranking_model,
)
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
from bestshot.services.preferences import (
    PreferenceServiceError,
    format_preference_stats,
    generate_video_preferences,
)
from bestshot.services.presampling import format_presampling_report, generate_presampling_report
from bestshot.services.ranking import format_ranking_training_result, train_personal_ranking
from bestshot.video.probe import VideoProbe, VideoProbeError

app = typer.Typer(
    add_completion=False,
    help="Présélection locale de candidates vidéo et embeddings visuels V2.",
)
models_app = typer.Typer(help="Gère les poids d'embedding conservés localement.")
dataset_app = typer.Typer(help="Consulte et administre le dataset local de préférences.")
preferences_app = typer.Typer(help="Génère et consulte les comparaisons pairwise locales.")
app.add_typer(models_app, name="models")
app.add_typer(dataset_app, name="dataset")
app.add_typer(preferences_app, name="preferences")
PERSONAL_MODELS_DIRECTORY = Path(".bestshot/models/personal")


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


@preferences_app.command("generate")
def preferences_generate(
    video: Annotated[
        Path,
        typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True),
    ],
) -> None:
    """Propose des paires locales non encore comparées pour une vidéo ingérée."""
    try:
        pairs = generate_video_preferences(
            _create_dataset_repository(),
            video,
            load_pair_generation_settings(),
        )
    except (ConfigurationError, DatasetRepositoryError, PreferenceServiceError, ValueError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Paires proposées : {len(pairs)}")
    for pair in pairs:
        typer.echo(f"{pair.first_frame_id} vs {pair.second_frame_id} ({pair.reason})")


@preferences_app.command("stats")
def preferences_stats() -> None:
    """Affiche la couverture des réponses pairwise, SKIP compris."""
    try:
        typer.echo(format_preference_stats(_create_dataset_repository().preference_stats()))
    except (ConfigurationError, DatasetRepositoryError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error


@preferences_app.command("review")
def preferences_review(
    video: Annotated[
        Path,
        typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True),
    ],
    reviewed: Annotated[
        bool,
        typer.Option("--reviewed", help="Inclut les paires déjà enregistrées pour modifier une réponse."),
    ] = False,
) -> None:
    """Ouvre l'écran PySide6 local de comparaison des candidates d'une vidéo."""
    from bestshot.desktop.pairwise_review import PreferenceWindowError, run_pairwise_review

    try:
        raise typer.Exit(
            code=run_pairwise_review(
                load_dataset_settings().database_path,
                video,
                load_pair_generation_settings(),
                include_reviewed=reviewed,
            )
        )
    except (ConfigurationError, PreferenceWindowError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error


@app.command("train-ranking")
def train_ranking() -> None:
    """Entraîne localement un head linéaire sur les préférences non-SKIP."""
    try:
        embedding_settings = load_embedding_settings()
        result, artifact = train_personal_ranking(
            _create_dataset_repository(),
            load_personal_ranking_settings(),
            PERSONAL_MODELS_DIRECTORY,
            f"{embedding_settings.repo_id}@{embedding_settings.revision}:"
            f"{embedding_settings.model_version}",
        )
    except (ConfigurationError, DatasetRepositoryError, RankingTrainingError, RuntimeError, ValueError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    typer.echo(format_ranking_training_result(result, artifact))


@app.command("ranking-score")
def ranking_score(
    image: Annotated[
        Path,
        typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True),
    ],
) -> None:
    """Calcule le score du modèle personnel courant pour une image locale."""
    try:
        provider = DINOv2EmbeddingProvider(load_embedding_settings())
        model = load_current_ranking_model(PERSONAL_MODELS_DIRECTORY)
        score = model.score(provider.embed(_preview_from_image(image)))
    except (
        ConfigurationError,
        DINOv2ModelError,
        EmbeddingError,
        OSError,
        RankingModelError,
        RuntimeError,
        ValueError,
    ) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Score personnel : {score:.6f}")


@dataset_app.command("stats")
def dataset_stats() -> None:
    """Affiche les compteurs du dataset local, y compris les labels SKIP."""
    try:
        typer.echo(format_dataset_stats(get_dataset_stats(_create_dataset_repository())))
    except (ConfigurationError, DatasetRepositoryError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error


@dataset_app.command("reset-labels")
def dataset_reset_labels() -> None:
    """Supprime tous les labels KEEP/REJECT et remet les frames à SKIP."""
    try:
        count = reset_dataset_labels(_create_dataset_repository())
    except (ConfigurationError, DatasetRepositoryError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"{count} label(s) réinitialisé(s) en SKIP.")


@dataset_app.command("videos")
def dataset_videos() -> None:
    """Liste les vidéos connues du dataset et leurs compteurs de labels."""
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


def _preview_from_image(image_path: Path) -> PreviewImage:
    """Lit localement une image pour le score, sans appel réseau ni fichier temporaire."""
    try:
        from PIL import Image
    except ImportError as error:
        raise RankingModelError("Installez l'extra : pip install -e '.[embedding]'.") from error
    try:
        with Image.open(image_path) as source:
            rgb = source.convert("RGB")
            return PreviewImage(rgb.width, rgb.height, rgb.tobytes())
    except OSError as error:
        raise RankingModelError(f"Impossible de lire l'image locale : {image_path}") from error
