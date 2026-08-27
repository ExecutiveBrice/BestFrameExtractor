"""Cas d'usage de consultation et de remise à zéro du dataset local."""

from bestshot.dataset.repository import DatasetRepository, DatasetStats, VideoDatasetSummary


def get_dataset_stats(repository: DatasetRepository) -> DatasetStats:
    """Retourne les compteurs du dataset sans charger de previews ni d'embeddings."""
    return repository.stats()


def reset_dataset_labels(repository: DatasetRepository) -> int:
    """Supprime les labels utilisateur et conserve toutes les candidates collectées."""
    return repository.reset_labels()


def list_dataset_videos(repository: DatasetRepository) -> list[VideoDatasetSummary]:
    """Liste les vidéos locales connues et la couverture de leurs labels."""
    return repository.list_videos()


def format_dataset_stats(stats: DatasetStats) -> str:
    """Représentation texte stable de la couverture du dataset."""
    return "\n".join(
        (
            f"Vidéos : {stats.video_count}",
            f"Frames : {stats.frame_count}",
            f"KEEP : {stats.keep_count}",
            f"REJECT : {stats.reject_count}",
            f"SKIP : {stats.skip_count}",
            f"Modèles d'entraînement : {stats.training_model_count}",
        )
    )


def format_dataset_videos(videos: list[VideoDatasetSummary]) -> str:
    """Affiche les vidéos sans révéler ni charger leurs contenus visuels."""
    if not videos:
        return "Aucune vidéo dans le dataset."
    return "\n".join(
        (
            f"{summary.video.source_path} | {summary.video.video_hash[:12]} | "
            f"frames={summary.frame_count} KEEP={summary.keep_count} "
            f"REJECT={summary.reject_count} SKIP={summary.skip_count}"
        )
        for summary in videos
    )
