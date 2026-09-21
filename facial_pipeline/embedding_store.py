"""Persist validated recognition vectors and their compatibility metadata in SQLite."""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np

from facial_pipeline.config import (
    EMBEDDING_DIMENSION,
    EMBEDDING_DTYPE,
    EMBEDDING_SCHEMA_VERSION,
    MODEL_NAME,
)


@dataclass(frozen=True)
class EmbeddingRecord:
    """One stored unit vector plus the metadata required to compare it safely."""
    embedding_id: str
    subject_id: str
    embedding: np.ndarray
    dimension: int
    dtype: str
    model_name: str
    schema_version: int
    normalized: bool
    created_at: str

    def metadata(self) -> dict[str, object]:
        """Return the vector-free fields that are safe to expose to API clients."""
        return {
            "embedding_id": self.embedding_id,
            "subject_id": self.subject_id,
            "dimension": self.dimension,
            "dtype": self.dtype,
            "model_name": self.model_name,
            "schema_version": self.schema_version,
            "normalized": self.normalized,
            "created_at": self.created_at,
        }


class EmbeddingStore:
    """Provide SQLite-backed CRUD operations for normalized face embeddings."""

    def __init__(self, path: Path) -> None:
        """Keep the database location without opening a connection eagerly."""
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        """Open a database connection configured for named-column row access."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self, connection: sqlite3.Connection) -> None:
        """Create the embedding table and lookup index when the database is new."""
        # WAL lets readers continue while a detection request writes enrollment data.
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                embedding_id TEXT PRIMARY KEY,
                subject_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                dimension INTEGER NOT NULL,
                dtype TEXT NOT NULL,
                model_name TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                normalized INTEGER NOT NULL CHECK (normalized IN (0, 1)),
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS embeddings_subject_id_idx "
            "ON embeddings(subject_id)"
        )

    def save(self, subject_id: str, embedding: np.ndarray) -> EmbeddingRecord:
        """Validate and persist one named embedding, returning its generated record."""
        vector = self._validated_vector(embedding)
        record = EmbeddingRecord(
            embedding_id=str(uuid4()),
            subject_id=subject_id,
            embedding=vector,
            dimension=EMBEDDING_DIMENSION,
            dtype=EMBEDDING_DTYPE,
            model_name=MODEL_NAME,
            schema_version=EMBEDDING_SCHEMA_VERSION,
            normalized=True,
            created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )

        with self._connect() as connection:
            self._initialize(connection)
            connection.execute(
                """
                INSERT INTO embeddings (
                    embedding_id, subject_id, embedding, dimension, dtype,
                    model_name, schema_version, normalized, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.embedding_id,
                    record.subject_id,
                    # Persist an explicit byte order so vectors remain portable across hosts.
                    sqlite3.Binary(record.embedding.astype("<f4", copy=False).tobytes()),
                    record.dimension,
                    record.dtype,
                    record.model_name,
                    record.schema_version,
                    int(record.normalized),
                    record.created_at,
                ),
            )
        return record

    def get(self, embedding_id: str | UUID) -> EmbeddingRecord | None:
        """Look up one record by UUID, returning nothing when it is absent."""
        with self._connect() as connection:
            self._initialize(connection)
            row = connection.execute(
                "SELECT * FROM embeddings WHERE embedding_id = ?",
                (str(embedding_id),),
            ).fetchone()
        if row is None:
            return None

        return self._record_from_row(row)

    def list_all(self) -> list[EmbeddingRecord]:
        """Return every record in deterministic newest-first order for matching."""
        with self._connect() as connection:
            self._initialize(connection)
            rows = connection.execute(
                "SELECT * FROM embeddings "
                "ORDER BY created_at DESC, embedding_id DESC"
            ).fetchall()
        return [self._record_from_row(row) for row in rows]

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> EmbeddingRecord:
        """Reconstruct a detached NumPy vector from SQLite's binary representation."""
        # Copy detaches the vector from SQLite's row buffer before the connection closes.
        embedding = np.frombuffer(row["embedding"], dtype="<f4").copy()
        if embedding.shape != (row["dimension"],):
            raise ValueError("Stored embedding size does not match its metadata")
        return EmbeddingRecord(
            embedding_id=row["embedding_id"],
            subject_id=row["subject_id"],
            embedding=np.ascontiguousarray(embedding, dtype=np.float32),
            dimension=row["dimension"],
            dtype=row["dtype"],
            model_name=row["model_name"],
            schema_version=row["schema_version"],
            normalized=bool(row["normalized"]),
            created_at=row["created_at"],
        )

    def delete(self, embedding_id: str | UUID) -> bool:
        """Delete one record and report whether its UUID existed."""
        with self._connect() as connection:
            self._initialize(connection)
            cursor = connection.execute(
                "DELETE FROM embeddings WHERE embedding_id = ?",
                (str(embedding_id),),
            )
        return cursor.rowcount == 1

    def delete_all(self) -> int:
        """Remove every enrollment record and return the number removed."""
        with self._connect() as connection:
            self._initialize(connection)
            cursor = connection.execute("DELETE FROM embeddings")
        return cursor.rowcount

    @staticmethod
    def _validated_vector(embedding: np.ndarray) -> np.ndarray:
        """Enforce the canonical contiguous float32 unit-vector storage format."""
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.shape != (EMBEDDING_DIMENSION,):
            raise ValueError(f"Embedding must have shape ({EMBEDDING_DIMENSION},)")
        if not np.all(np.isfinite(vector)):
            raise ValueError("Embedding must contain only finite values")
        norm = float(np.linalg.norm(vector))
        if not np.isclose(norm, 1.0, rtol=1e-5, atol=1e-6):
            raise ValueError("Embedding must be L2-normalized")
        return np.ascontiguousarray(vector, dtype=np.float32)
