"""Interface en ligne de commande de BestShotAI."""

from pathlib import Path
from typing import Annotated

import typer

from bestshot.infrastructure.candidate_repository import (
    CandidateRepositoryError,
    LocalCandidatePreviewRepository,
)
from bestshot.infrastructure.config import (
    ConfigurationError,
    load_aesthetic_model_settings,
    load_candidate_extraction_settings,
    load_composite_scoring_settings,
    load_deduplication_settings,
    load_export_settings,
    load_face_scoring_settings,
    load_scene_detector_settings,
    load_selection_settings,
    load_technical_scoring_settings,
)
from bestshot.infrastructure.ffmpeg import FFmpegExportError, FFmpegFrameExporter
from bestshot.infrastructure.ffprobe import SubprocessFFprobeRunner
from bestshot.plugins.aesthetic import AestheticModelManager, create_aesthetic_scorer
from bestshot.scoring.composite import CompositeScorer
from bestshot.scoring.face import FaceScoringError, create_face_scorer
from bestshot.scoring.technical import TechnicalScorer
from bestshot.selection.deduplicate import Deduplicator, PerceptualHashSimilarityScorer
from bestshot.selection.exporter import FinalExporter
from bestshot.selection.selector import BestFrameSelector
from bestshot.services.aesthetic_analysis import format_aesthetic_analysis
from bestshot.services.candidates import (
    extract_candidates,
    format_candidate_repository_result,
    persist_candidate_previews,
)
from bestshot.services.scenes import detect_scenes, format_scenes
from bestshot.services.selection import format_selection_result, rank_candidates, select_best_frames
from bestshot.services.technical_analysis import format_technical_analysis
from bestshot.services.video_info import format_video_info, get_video_info
from bestshot.video.candidate_extractor import (
    CandidateExtractionError,
    CandidateExtractor,
    PyAVCandidateFrameBackend,
)
from bestshot.video.probe import VideoProbe, VideoProbeError
from bestshot.video.scene_detector import PySceneDetectBackend, SceneDetectionError, SceneDetector

app = typer.Typer(
    add_completion=False,
    help="Extraction locale des meilleures images fixes depuis une vidéo.",
)
models_app = typer.Typer(help="Gère les modèles optionnels conservés localement.")
app.add_typer(models_app, name="models")


@models_app.callback(invoke_without_command=True)
def models() -> None:
    """Affiche l'état du modèle esthétique local."""
    settings = load_aesthetic_model_settings()
    status = AestheticModelManager().status(settings)
    typer.echo(f"Modèle esthétique : {status.message} ({status.cache_path})")


@models_app.command("download")
def download_model(name: str = typer.Argument(...)) -> None:
    """Télécharge explicitement un modèle optionnel dans le cache local."""
    if name != "aesthetic":
        raise typer.BadParameter("Seul le modèle aesthetic est disponible.")
    try:
        status = AestheticModelManager().download(load_aesthetic_model_settings())
    except (ConfigurationError, OSError, RuntimeError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Modèle esthétique : {status.message} ({status.cache_path})")


@app.command()
def info(
    video: Annotated[
        Path,
        typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True),
    ],
) -> None:
    """Affiche les métadonnées de VIDEO collectées par ffprobe."""
    try:
        result = get_video_info(video, VideoProbe(SubprocessFFprobeRunner()))
    except VideoProbeError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    typer.echo(format_video_info(result))


@app.command()
def scenes(
    video: Annotated[
        Path,
        typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True),
    ],
) -> None:
    """Affiche les scènes de VIDEO détectées localement."""
    try:
        settings = load_scene_detector_settings()
        detector = SceneDetector(PySceneDetectBackend(), settings)
        result = detect_scenes(video, detector)
    except (ConfigurationError, SceneDetectionError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    typer.echo(format_scenes(result))


@app.command()
def candidates(
    video: Annotated[
        Path,
        typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True),
    ],
) -> None:
    """Crée les aperçus de candidates dans le dépôt local configuré."""
    try:
        scene_settings = load_scene_detector_settings()
        scenes_result = detect_scenes(video, SceneDetector(PySceneDetectBackend(), scene_settings))
        extraction_settings = load_candidate_extraction_settings()
        extractor = CandidateExtractor(PyAVCandidateFrameBackend(), extraction_settings)
        result = persist_candidate_previews(
            video,
            extract_candidates(video, scenes_result, extractor),
            LocalCandidatePreviewRepository(extraction_settings.candidate_repository_dir),
        )
        output = format_candidate_repository_result(
            scenes_result,
            result,
        )
    except (
        CandidateExtractionError,
        CandidateRepositoryError,
        ConfigurationError,
        SceneDetectionError,
    ) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    typer.echo(output)


@app.command()
def analyse(
    video: Annotated[
        Path,
        typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True),
    ],
    technical_only: Annotated[
        bool,
        typer.Option("--technical-only", help="Calcule uniquement les scores techniques."),
    ] = False,
    aesthetic: Annotated[bool, typer.Option("--aesthetic", help="Active le plugin CLIP esthétique local.")] = False,
) -> None:
    """Analyse VIDEO ; utilisez --technical-only pour le scorer technique disponible."""
    if not technical_only and not aesthetic:
        typer.echo("Sélectionnez --technical-only ou --aesthetic.", err=True)
        raise typer.Exit(code=2)
    try:
        scene_settings = load_scene_detector_settings()
        scenes_result = detect_scenes(video, SceneDetector(PySceneDetectBackend(), scene_settings))
        extraction_settings = load_candidate_extraction_settings()
        candidates_result = extract_candidates(
            video,
            scenes_result,
            CandidateExtractor(PyAVCandidateFrameBackend(), extraction_settings),
        )
        output = (
            format_aesthetic_analysis(candidates_result, create_aesthetic_scorer(load_aesthetic_model_settings()))
            if aesthetic
            else format_technical_analysis(
                scenes_result, candidates_result, TechnicalScorer(load_technical_scoring_settings())
            )
        )
    except (CandidateExtractionError, ConfigurationError, SceneDetectionError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    typer.echo(output)


@app.command()
def select(
    video: Annotated[
        Path,
        typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True),
    ],
    count: Annotated[int, typer.Option("--count", min=1, help="Nombre maximal de photos à retenir.")] = 20,
) -> None:
    """Sélectionne les meilleures frames diversifiées de VIDEO, sans les exporter."""
    try:
        scenes_result = detect_scenes(
            video, SceneDetector(PySceneDetectBackend(), load_scene_detector_settings())
        )
        face_settings = load_face_scoring_settings()
        ranked = rank_candidates(
            extract_candidates(
                video,
                scenes_result,
                CandidateExtractor(PyAVCandidateFrameBackend(), load_candidate_extraction_settings()),
            ),
            TechnicalScorer(load_technical_scoring_settings()),
            create_face_scorer(face_settings),
            CompositeScorer(load_composite_scoring_settings()),
        )
        deduplication_settings = load_deduplication_settings()
        result = select_best_frames(
            ranked,
            scenes_result,
            Deduplicator(
                PerceptualHashSimilarityScorer(deduplication_settings.hash_size),
                deduplication_settings,
            ),
            BestFrameSelector(load_selection_settings()),
            count,
        )
    except (CandidateExtractionError, ConfigurationError, FaceScoringError, SceneDetectionError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    typer.echo(format_selection_result(result))


@app.command()
def extract(
    video: Annotated[
        Path,
        typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True),
    ],
    count: Annotated[int, typer.Option("--count", min=1)] = 30,
    output: Annotated[Path, typer.Option("--output", file_okay=False)] = Path("photos"),
    image_format: Annotated[str, typer.Option("--format")] = "jpeg",
) -> None:
    """Sélectionne puis extrait des frames natives dans OUTPUT."""
    try:
        scenes_result = detect_scenes(
            video, SceneDetector(PySceneDetectBackend(), load_scene_detector_settings())
        )
        face_settings = load_face_scoring_settings()
        ranked = rank_candidates(
            extract_candidates(
                video,
                scenes_result,
                CandidateExtractor(PyAVCandidateFrameBackend(), load_candidate_extraction_settings()),
            ),
            TechnicalScorer(load_technical_scoring_settings()),
            create_face_scorer(face_settings),
            CompositeScorer(load_composite_scoring_settings()),
        )
        deduplication_settings = load_deduplication_settings()
        selection = select_best_frames(
            ranked,
            scenes_result,
            Deduplicator(
                PerceptualHashSimilarityScorer(deduplication_settings.hash_size),
                deduplication_settings,
            ),
            BestFrameSelector(load_selection_settings()),
            count,
        )
        result = FinalExporter(FFmpegFrameExporter(), load_export_settings()).export(
            video, selection, output, image_format
        )
    except (
        CandidateExtractionError,
        ConfigurationError,
        FaceScoringError,
        FFmpegExportError,
        SceneDetectionError,
        ValueError,
    ) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"{len(result.image_paths)} image(s) exportée(s) dans {result.output_directory}")
    typer.echo(f"Manifeste : {result.manifest_path}")
