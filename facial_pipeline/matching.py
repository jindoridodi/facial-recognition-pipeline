from dataclasses import dataclass
from typing import Iterable

import numpy as np

from facial_pipeline.config import (
    EMBEDDING_DIMENSION,
    EMBEDDING_SCHEMA_VERSION,
    MODEL_NAME,
)
from facial_pipeline.embedding_store import EmbeddingRecord


@dataclass(frozen=True)
class EmbeddingMatch:
    record: EmbeddingRecord
    similarity: float

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.record.subject_id,
            "subject_id": self.record.subject_id,
            "embedding_id": self.record.embedding_id,
            "similarity": round(self.similarity, 4),
        }


def find_best_match(
    embedding: np.ndarray,
    candidates: Iterable[EmbeddingRecord],
    threshold: float,
) -> EmbeddingMatch | None:
    """Return the closest compatible stored vector above a cosine threshold."""
    query = np.asarray(embedding, dtype=np.float32)
    if query.shape != (EMBEDDING_DIMENSION,) or not np.all(np.isfinite(query)):
        raise ValueError(
            f"Query embedding is not a valid {EMBEDDING_DIMENSION}D vector"
        )

    best_record: EmbeddingRecord | None = None
    best_similarity = -1.0
    for record in candidates:
        if (
            record.dimension != EMBEDDING_DIMENSION
            or record.model_name != MODEL_NAME
            or record.schema_version != EMBEDDING_SCHEMA_VERSION
            or not record.normalized
        ):
            continue

        similarity = float(np.dot(query, record.embedding))
        if similarity > best_similarity:
            best_record = record
            best_similarity = similarity

    if best_record is None or best_similarity < threshold:
        return None
    return EmbeddingMatch(record=best_record, similarity=best_similarity)
