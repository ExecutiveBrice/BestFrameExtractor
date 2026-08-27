"""Implémentation SQLite locale et migrée du dataset de préférences."""

from __future__ import annotations

import hashlib
import math
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from bestshot.dataset.labels import FrameLabel, from_storage_value, to_storage_value
from bestshot.dataset.repository import (
    DatasetStats,
    FrameRecord,
    PreferenceStats,
    TrainingModel,
    VideoDatasetSummary,
    VideoRecord,
)
from bestshot.domain.preferences import (
    PairwisePreference,
    PreferenceChoice,
    canonicalize_preference,
)

SCHEMA_VERSION = 3


class DatasetRepositoryError(RuntimeError):
    """Le dataset SQLite local ne peut pas être lu ou mis à jour."""


class SQLiteDatasetRepository:
    """Dataset local sans blobs image : seulement des métadonnées et références externes."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path
        self.migrate()

    def migrate(self) -> None:
        """Applique de façon idempotente les migrations de schéma connues."""
        try:
            with closing(self._connect()) as connection, connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY)"
                )
                applied = {
                    int(row["version"])
                    for row in connection.execute("SELECT version FROM schema_migrations")
                }
                if 1 not in applied:
                    self._apply_initial_schema(connection)
                    connection.execute("INSERT INTO schema_migrations(version) VALUES (1)")
                if 2 not in applied:
                    self._apply_pairwise_preferences_schema(connection)
                    connection.execute("INSERT INTO schema_migrations(version) VALUES (2)")
                if 3 not in applied:
                    self._apply_learning_state_schema(connection)
                    connection.execute("INSERT INTO schema_migrations(version) VALUES (3)")
        except sqlite3.Error as error:
            raise DatasetRepositoryError(f"Impossible de migrer le dataset : {error}") from error

    def upsert_video(self, record: VideoRecord) -> VideoRecord:
        """Enregistre l'identité locale de la vidéo, indexée par son hash de contenu."""
        if not record.video_hash:
            raise DatasetRepositoryError("Le hash de vidéo est obligatoire.")
        if record.source_size < 0 or record.source_mtime_ns < 0:
            raise DatasetRepositoryError("Les métadonnées de vidéo sont invalides.")
        now = _now()
        try:
            with closing(self._connect()) as connection, connection:
                connection.execute(
                    """
                    INSERT INTO videos(source_path, video_hash, source_size, source_mtime_ns, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(video_hash) DO UPDATE SET
                      source_path = excluded.source_path,
                      source_size = excluded.source_size,
                      source_mtime_ns = excluded.source_mtime_ns,
                      updated_at = excluded.updated_at
                    """,
                    (
                        str(record.source_path),
                        record.video_hash,
                        record.source_size,
                        record.source_mtime_ns,
                        record.created_at or now,
                        now,
                    ),
                )
                row = connection.execute(
                    "SELECT id, source_path, video_hash, source_size, source_mtime_ns, created_at "
                    "FROM videos WHERE video_hash = ?",
                    (record.video_hash,),
                ).fetchone()
        except sqlite3.Error as error:
            raise DatasetRepositoryError(f"Impossible d'enregistrer la vidéo : {error}") from error
        if row is None:
            raise DatasetRepositoryError("La vidéo enregistrée est introuvable.")
        return _video_from_row(row)

    def upsert_frame(self, record: FrameRecord) -> FrameRecord:
        """Enregistre une candidate sans blob de preview ni embedding dans SQLite."""
        _validate_frame(record)
        now = _now()
        try:
            with closing(self._connect()) as connection, connection:
                connection.execute(
                    """
                    INSERT INTO frames(
                      video_id, timestamp, frame_index, preview_reference, sharpness,
                      embedding_reference, label, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(video_id, frame_index) DO UPDATE SET
                      timestamp = excluded.timestamp,
                      preview_reference = excluded.preview_reference,
                      sharpness = excluded.sharpness,
                      embedding_reference = excluded.embedding_reference,
                      label = CASE WHEN excluded.label IS NULL THEN frames.label ELSE excluded.label END,
                      updated_at = excluded.updated_at
                    """,
                    (
                        record.video_id,
                        record.timestamp,
                        record.frame_index,
                        record.preview_reference,
                        record.sharpness,
                        record.embedding_reference,
                        to_storage_value(record.label),
                        record.created_at or now,
                        now,
                    ),
                )
                row = connection.execute(
                    """
                    SELECT id, video_id, timestamp, frame_index, preview_reference, sharpness,
                           embedding_reference, label, created_at
                    FROM frames WHERE video_id = ? AND frame_index = ?
                    """,
                    (record.video_id, record.frame_index),
                ).fetchone()
        except sqlite3.Error as error:
            raise DatasetRepositoryError(f"Impossible d'enregistrer la frame : {error}") from error
        if row is None:
            raise DatasetRepositoryError("La frame enregistrée est introuvable.")
        return _frame_from_row(row)

    def set_frame_label(self, frame_id: int, label: FrameLabel) -> None:
        """Met à jour un label ; SKIP devient NULL dans la base."""
        try:
            with closing(self._connect()) as connection, connection:
                cursor = connection.execute(
                    "UPDATE frames SET label = ?, updated_at = ? WHERE id = ?",
                    (to_storage_value(label), _now(), frame_id),
                )
        except sqlite3.Error as error:
            raise DatasetRepositoryError(f"Impossible de mettre à jour le label : {error}") from error
        if cursor.rowcount != 1:
            raise DatasetRepositoryError(f"Frame inconnue pour le label : {frame_id}")

    def reset_labels(self) -> int:
        """Remplace tous les labels par NULL et retourne le nombre de labels supprimés."""
        try:
            with closing(self._connect()) as connection, connection:
                cursor = connection.execute(
                    "UPDATE frames SET label = NULL, updated_at = ? WHERE label IS NOT NULL",
                    (_now(),),
                )
        except sqlite3.Error as error:
            raise DatasetRepositoryError(f"Impossible de réinitialiser les labels : {error}") from error
        return cursor.rowcount

    def stats(self) -> DatasetStats:
        """Compte les labels en distinguant explicitement l'absence de label."""
        try:
            with closing(self._connect()) as connection, connection:
                row = connection.execute(
                    """
                    SELECT
                      (SELECT COUNT(*) FROM videos) AS video_count,
                      COUNT(*) AS frame_count,
                      COALESCE(SUM(label = 'KEEP'), 0) AS keep_count,
                      COALESCE(SUM(label = 'REJECT'), 0) AS reject_count,
                      COALESCE(SUM(label IS NULL), 0) AS skip_count,
                      (SELECT COUNT(*) FROM training_models) AS training_model_count
                    FROM frames
                    """
                ).fetchone()
        except sqlite3.Error as error:
            raise DatasetRepositoryError(f"Impossible de lire les statistiques : {error}") from error
        if row is None:
            raise DatasetRepositoryError("Les statistiques du dataset sont indisponibles.")
        return DatasetStats(
            video_count=int(row["video_count"]),
            frame_count=int(row["frame_count"]),
            keep_count=int(row["keep_count"]),
            reject_count=int(row["reject_count"]),
            skip_count=int(row["skip_count"]),
            training_model_count=int(row["training_model_count"]),
        )

    def list_videos(self) -> list[VideoDatasetSummary]:
        """Liste les vidéos et les labels associés à leurs candidates."""
        try:
            with closing(self._connect()) as connection, connection:
                rows = connection.execute(
                    """
                    SELECT v.id, v.source_path, v.video_hash, v.source_size, v.source_mtime_ns, v.created_at,
                           COUNT(f.id) AS frame_count,
                           COALESCE(SUM(f.label = 'KEEP'), 0) AS keep_count,
                           COALESCE(SUM(f.label = 'REJECT'), 0) AS reject_count,
                           COALESCE(SUM(f.label IS NULL), 0) AS skip_count
                    FROM videos v
                    LEFT JOIN frames f ON f.video_id = v.id
                    GROUP BY v.id
                    ORDER BY v.source_path
                    """
                ).fetchall()
        except sqlite3.Error as error:
            raise DatasetRepositoryError(f"Impossible de lister les vidéos : {error}") from error
        return [
            VideoDatasetSummary(
                video=_video_from_row(row),
                frame_count=int(row["frame_count"]),
                keep_count=int(row["keep_count"]),
                reject_count=int(row["reject_count"]),
                skip_count=int(row["skip_count"]),
            )
            for row in rows
        ]

    def upsert_training_model(self, model: TrainingModel) -> TrainingModel:
        """Persiste uniquement les métadonnées d'un futur modèle, sans entraînement."""
        if not model.name or not model.version:
            raise DatasetRepositoryError("Le nom et la version du modèle sont obligatoires.")
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO training_models(name, version, metadata_json, created_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(name, version) DO UPDATE SET metadata_json = excluded.metadata_json
                    """,
                    (model.name, model.version, model.metadata_json, model.created_at or _now()),
                )
                row = connection.execute(
                    "SELECT id, name, version, metadata_json, created_at "
                    "FROM training_models WHERE name = ? AND version = ?",
                    (model.name, model.version),
                ).fetchone()
        except sqlite3.Error as error:
            raise DatasetRepositoryError(f"Impossible d'enregistrer le modèle : {error}") from error
        if row is None:
            raise DatasetRepositoryError("Le modèle enregistré est introuvable.")
        return TrainingModel(
            id=int(row["id"]),
            name=str(row["name"]),
            version=str(row["version"]),
            metadata_json=str(row["metadata_json"]),
            created_at=str(row["created_at"]),
        )

    def get_video_by_source_path(self, source_path: Path) -> VideoRecord | None:
        """Retrouve la version la plus récente d'une source locale normalisée."""
        try:
            normalized_path = str(source_path.resolve())
            with closing(self._connect()) as connection:
                row = connection.execute(
                    "SELECT id, source_path, video_hash, source_size, source_mtime_ns, created_at "
                    "FROM videos WHERE source_path = ? ORDER BY updated_at DESC, id DESC LIMIT 1",
                    (normalized_path,),
                ).fetchone()
        except (OSError, sqlite3.Error) as error:
            raise DatasetRepositoryError(f"Impossible de lire la vidéo du dataset : {error}") from error
        return _video_from_row(row) if row is not None else None

    def list_frames_for_video(self, video_id: int) -> list[FrameRecord]:
        """Retourne les candidates légères d'une vidéo dans l'ordre temporel."""
        if video_id <= 0:
            raise DatasetRepositoryError("L'identifiant de vidéo doit être positif.")
        try:
            with closing(self._connect()) as connection:
                rows = connection.execute(
                    """
                    SELECT id, video_id, timestamp, frame_index, preview_reference, sharpness,
                           embedding_reference, label, created_at
                    FROM frames WHERE video_id = ? ORDER BY timestamp, frame_index
                    """,
                    (video_id,),
                ).fetchall()
        except sqlite3.Error as error:
            raise DatasetRepositoryError(f"Impossible de lister les frames : {error}") from error
        return [_frame_from_row(row) for row in rows]

    def get_frames_by_ids(self, frame_ids: set[int]) -> dict[int, FrameRecord]:
        """Charge en une requête les candidates nécessaires à l'entraînement."""
        if not frame_ids:
            return {}
        if any(frame_id <= 0 for frame_id in frame_ids):
            raise DatasetRepositoryError("Les identifiants de frames doivent être positifs.")
        placeholders = ", ".join("?" for _ in frame_ids)
        try:
            with closing(self._connect()) as connection:
                rows = connection.execute(
                    """
                    SELECT id, video_id, timestamp, frame_index, preview_reference, sharpness,
                           embedding_reference, label, created_at
                    FROM frames WHERE id IN ("""
                    + placeholders
                    + ")",
                    tuple(sorted(frame_ids)),
                ).fetchall()
        except sqlite3.Error as error:
            raise DatasetRepositoryError(f"Impossible de charger les frames : {error}") from error
        return {int(row["id"]): _frame_from_row(row) for row in rows}

    def save_preference(self, preference: PairwisePreference) -> PairwisePreference:
        """Insère ou actualise un choix pairwise sans dupliquer la paire inverse."""
        first_id, second_id, choice = canonicalize_preference(
            preference.first_frame_id,
            preference.second_frame_id,
            preference.preference,
        )
        now = _now()
        try:
            with closing(self._connect()) as connection, connection:
                connection.execute(
                    """
                    INSERT INTO pairwise_preferences(
                      first_frame_id, second_frame_id, preference, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(first_frame_id, second_frame_id) DO UPDATE SET
                      preference = excluded.preference,
                      updated_at = excluded.updated_at
                    """,
                    (first_id, second_id, _preference_to_storage(choice), preference.created_at or now, now),
                )
                row = connection.execute(
                    """
                    SELECT id, first_frame_id, second_frame_id, preference, created_at, updated_at
                    FROM pairwise_preferences
                    WHERE first_frame_id = ? AND second_frame_id = ?
                    """,
                    (first_id, second_id),
                ).fetchone()
        except sqlite3.Error as error:
            raise DatasetRepositoryError(f"Impossible d'enregistrer la préférence : {error}") from error
        if row is None:
            raise DatasetRepositoryError("La préférence enregistrée est introuvable.")
        return _preference_from_row(row)

    def delete_preference(self, first_frame_id: int, second_frame_id: int) -> bool:
        """Supprime une préférence quelle que soit l'orientation demandée."""
        first_id, second_id, _ = canonicalize_preference(
            first_frame_id, second_frame_id, PreferenceChoice.SKIP
        )
        try:
            with closing(self._connect()) as connection, connection:
                cursor = connection.execute(
                    "DELETE FROM pairwise_preferences WHERE first_frame_id = ? AND second_frame_id = ?",
                    (first_id, second_id),
                )
        except sqlite3.Error as error:
            raise DatasetRepositoryError(f"Impossible de supprimer la préférence : {error}") from error
        return cursor.rowcount == 1

    def get_preference(self, first_frame_id: int, second_frame_id: int) -> PairwisePreference | None:
        """Retourne une préférence canonique, indépendamment de l'ordre fourni."""
        first_id, second_id, _ = canonicalize_preference(
            first_frame_id, second_frame_id, PreferenceChoice.SKIP
        )
        try:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    """
                    SELECT id, first_frame_id, second_frame_id, preference, created_at, updated_at
                    FROM pairwise_preferences
                    WHERE first_frame_id = ? AND second_frame_id = ?
                    """,
                    (first_id, second_id),
                ).fetchone()
        except sqlite3.Error as error:
            raise DatasetRepositoryError(f"Impossible de lire la préférence : {error}") from error
        return _preference_from_row(row) if row is not None else None

    def list_usable_preferences(self) -> list[PairwisePreference]:
        """Retourne FIRST, SECOND et EQUAL ; les SKIP ne sont jamais entraînés."""
        return self._list_preferences("preference IS NOT NULL", ())

    def list_preferences_for_video(self, video_id: int) -> list[PairwisePreference]:
        """Retourne chaque paire impliquant une candidate de la vidéo demandée."""
        if video_id <= 0:
            raise DatasetRepositoryError("L'identifiant de vidéo doit être positif.")
        return self._list_preferences(
            "first_frame_id IN (SELECT id FROM frames WHERE video_id = ?) "
            "OR second_frame_id IN (SELECT id FROM frames WHERE video_id = ?)",
            (video_id, video_id),
        )

    def preference_stats(self) -> PreferenceStats:
        """Compte séparément les réponses utilisables et les absences de jugement."""
        try:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    """
                    SELECT
                      COUNT(*) AS total_count,
                      COALESCE(SUM(preference = 'FIRST'), 0) AS first_count,
                      COALESCE(SUM(preference = 'SECOND'), 0) AS second_count,
                      COALESCE(SUM(preference = 'EQUAL'), 0) AS equal_count,
                      COALESCE(SUM(preference IS NULL), 0) AS skip_count,
                      (SELECT COUNT(DISTINCT f.video_id) FROM frames f
                       JOIN pairwise_preferences p ON f.id = p.first_frame_id OR f.id = p.second_frame_id
                      ) AS video_count,
                      (SELECT COUNT(*) FROM (
                         SELECT first_frame_id AS frame_id FROM pairwise_preferences
                         UNION SELECT second_frame_id AS frame_id FROM pairwise_preferences
                       )) AS distinct_frame_count
                    FROM pairwise_preferences
                    """
                ).fetchone()
        except sqlite3.Error as error:
            raise DatasetRepositoryError(f"Impossible de lire les statistiques de préférences : {error}") from error
        if row is None:
            raise DatasetRepositoryError("Les statistiques de préférences sont indisponibles.")
        return PreferenceStats(
            total_count=int(row["total_count"]),
            first_count=int(row["first_count"]),
            second_count=int(row["second_count"]),
            equal_count=int(row["equal_count"]),
            skip_count=int(row["skip_count"]),
            video_count=int(row["video_count"]),
            distinct_frame_count=int(row["distinct_frame_count"]),
        )

    def reset_preferences(self) -> int:
        """Supprime les réponses pairwise, sans toucher aux frames ni aux caches."""
        try:
            with closing(self._connect()) as connection, connection:
                cursor = connection.execute("DELETE FROM pairwise_preferences")
        except sqlite3.Error as error:
            raise DatasetRepositoryError(f"Impossible de réinitialiser les préférences : {error}") from error
        return cursor.rowcount

    def set_active_learning_pool(self, directory: Path) -> None:
        """Persiste le dernier dossier de photos réellement importé."""
        try:
            normalized_path = str(directory.resolve())
            with closing(self._connect()) as connection, connection:
                connection.execute(
                    """
                    INSERT INTO learning_state(key, value, updated_at)
                    VALUES ('active_photo_pool', ?, ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                    """,
                    (normalized_path, _now()),
                )
        except (OSError, sqlite3.Error) as error:
            raise DatasetRepositoryError(f"Impossible de mémoriser le pool d'apprentissage : {error}") from error

    def get_active_learning_pool(self) -> Path | None:
        """Lit le dernier dossier de photos importé, sans vérifier ses fichiers."""
        try:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    "SELECT value FROM learning_state WHERE key = 'active_photo_pool'"
                ).fetchone()
        except sqlite3.Error as error:
            raise DatasetRepositoryError(f"Impossible de lire le pool d'apprentissage : {error}") from error
        return Path(str(row["value"])) if row is not None else None

    def _list_preferences(
        self, where_clause: str, parameters: tuple[object, ...]
    ) -> list[PairwisePreference]:
        try:
            with closing(self._connect()) as connection:
                rows = connection.execute(
                    """
                    SELECT id, first_frame_id, second_frame_id, preference, created_at, updated_at
                    FROM pairwise_preferences WHERE """
                    + where_clause
                    + " ORDER BY id",
                    parameters,
                ).fetchall()
        except sqlite3.Error as error:
            raise DatasetRepositoryError(f"Impossible de lister les préférences : {error}") from error
        return [_preference_from_row(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _apply_initial_schema(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE videos (
              id INTEGER PRIMARY KEY,
              source_path TEXT NOT NULL,
              video_hash TEXT NOT NULL UNIQUE,
              source_size INTEGER NOT NULL,
              source_mtime_ns INTEGER NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE frames (
              id INTEGER PRIMARY KEY,
              video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
              timestamp REAL NOT NULL CHECK(timestamp >= 0),
              frame_index INTEGER NOT NULL CHECK(frame_index >= 0),
              preview_reference TEXT NOT NULL,
              sharpness REAL NOT NULL,
              embedding_reference TEXT NOT NULL,
              label TEXT NULL CHECK(label IN ('KEEP', 'REJECT')),
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              UNIQUE(video_id, frame_index)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE training_models (
              id INTEGER PRIMARY KEY,
              name TEXT NOT NULL,
              version TEXT NOT NULL,
              metadata_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              UNIQUE(name, version)
            )
            """
        )

    @staticmethod
    def _apply_pairwise_preferences_schema(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE pairwise_preferences (
              id INTEGER PRIMARY KEY,
              first_frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
              second_frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
              preference TEXT NULL CHECK(preference IN ('FIRST', 'SECOND', 'EQUAL')),
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              CHECK(first_frame_id < second_frame_id),
              UNIQUE(first_frame_id, second_frame_id)
            )
            """
        )
        connection.execute(
            "CREATE INDEX idx_pairwise_preferences_first_frame ON pairwise_preferences(first_frame_id)"
        )
        connection.execute(
            "CREATE INDEX idx_pairwise_preferences_second_frame ON pairwise_preferences(second_frame_id)"
        )
        connection.execute(
            "CREATE INDEX idx_pairwise_preferences_choice ON pairwise_preferences(preference)"
        )

    @staticmethod
    def _apply_learning_state_schema(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE learning_state (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL,
              updated_at TEXT NOT NULL
            )
            """
        )


def hash_video_file(video_path: Path) -> str:
    """Calcule séquentiellement le SHA-256 local d'une vidéo sans charger ses frames."""
    digest = hashlib.sha256()
    try:
        with video_path.open("rb") as video_file:
            for chunk in iter(lambda: video_file.read(1_048_576), b""):
                digest.update(chunk)
    except OSError as error:
        raise DatasetRepositoryError(f"Impossible de hacher la vidéo : {video_path}") from error
    return digest.hexdigest()


def video_record_from_path(video_path: Path) -> VideoRecord:
    """Construit l'identité d'une vidéo à persister dans le dataset."""
    try:
        stat = video_path.stat()
    except OSError as error:
        raise DatasetRepositoryError(f"Impossible d'inspecter la vidéo : {video_path}") from error
    return VideoRecord(
        source_path=video_path.resolve(),
        video_hash=hash_video_file(video_path),
        source_size=stat.st_size,
        source_mtime_ns=stat.st_mtime_ns,
    )


def photo_pool_record_from_paths(directory: Path, photos: tuple[Path, ...]) -> VideoRecord:
    """Construit l'identité d'un pool photo virtuel pour le dataset existant.

    Le schéma conserve une relation parent/candidates déjà utilisée par les vidéos.
    Un dossier photo devient donc un parent local autonome : il n'est jamais une
    candidate vidéo et ne peut pas être sélectionné dans l'export final.
    """
    if not photos:
        raise DatasetRepositoryError("Le pool de photos ne peut pas être vide.")
    normalized_directory = directory.resolve()
    digest = hashlib.sha256(b"bestshot-photo-pool-v1\n")
    source_size = 0
    source_mtime_ns = 0
    for photo_path in photos:
        try:
            normalized_photo = photo_path.resolve()
            relative_path = normalized_photo.relative_to(normalized_directory)
            stat = normalized_photo.stat()
        except (OSError, ValueError) as error:
            raise DatasetRepositoryError(f"Impossible d'identifier la photo du pool : {photo_path}") from error
        if not normalized_photo.is_file():
            raise DatasetRepositoryError(f"La source du pool n'est pas une photo : {photo_path}")
        digest.update(str(relative_path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(hash_video_file(normalized_photo).encode("ascii"))
        digest.update(b"\n")
        source_size += stat.st_size
        source_mtime_ns = max(source_mtime_ns, stat.st_mtime_ns)
    return VideoRecord(
        source_path=normalized_directory,
        video_hash=digest.hexdigest(),
        source_size=source_size,
        source_mtime_ns=source_mtime_ns,
    )


def _validate_frame(record: FrameRecord) -> None:
    if record.video_id <= 0 or record.frame_index < 0 or record.timestamp < 0:
        raise DatasetRepositoryError("Les métadonnées temporelles de la frame sont invalides.")
    if not math.isfinite(record.sharpness):
        raise DatasetRepositoryError("La netteté de la frame doit être finie.")
    if not record.preview_reference or not record.embedding_reference:
        raise DatasetRepositoryError("Les références de preview et d'embedding sont obligatoires.")


def _video_from_row(row: sqlite3.Row) -> VideoRecord:
    return VideoRecord(
        id=int(row["id"]),
        source_path=Path(str(row["source_path"])),
        video_hash=str(row["video_hash"]),
        source_size=int(row["source_size"]),
        source_mtime_ns=int(row["source_mtime_ns"]),
        created_at=str(row["created_at"]),
    )


def _frame_from_row(row: sqlite3.Row) -> FrameRecord:
    return FrameRecord(
        id=int(row["id"]),
        video_id=int(row["video_id"]),
        timestamp=float(row["timestamp"]),
        frame_index=int(row["frame_index"]),
        preview_reference=str(row["preview_reference"]),
        sharpness=float(row["sharpness"]),
        embedding_reference=str(row["embedding_reference"]),
        label=from_storage_value(row["label"]),
        created_at=str(row["created_at"]),
    )


def _preference_to_storage(preference: PreferenceChoice) -> str | None:
    return None if preference is PreferenceChoice.SKIP else preference.value


def _preference_from_row(row: sqlite3.Row) -> PairwisePreference:
    stored_choice = row["preference"]
    return PairwisePreference(
        id=int(row["id"]),
        first_frame_id=int(row["first_frame_id"]),
        second_frame_id=int(row["second_frame_id"]),
        preference=PreferenceChoice(str(stored_choice))
        if stored_choice is not None
        else PreferenceChoice.SKIP,
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _now() -> str:
    return datetime.now(UTC).isoformat()
