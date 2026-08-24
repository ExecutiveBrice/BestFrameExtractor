"""Interface en ligne de commande de BestShotAI."""

from pathlib import Path

import typer

from bestshot.infrastructure.config import ConfigurationError, load_scene_detector_settings
from bestshot.infrastructure.ffprobe import SubprocessFFprobeRunner
from bestshot.services.scenes import detect_scenes, format_scenes
from bestshot.services.video_info import format_video_info, get_video_info
from bestshot.video.probe import VideoProbe, VideoProbeError
from bestshot.video.scene_detector import PySceneDetectBackend, SceneDetectionError, SceneDetector

app = typer.Typer(
    add_completion=False,
    help="Extraction locale des meilleures images fixes depuis une vidéo.",
)


@app.command()
def info(
    video: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True),
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
    video: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True),
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
