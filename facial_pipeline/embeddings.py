from typing import Any

import numpy as np

from facial_pipeline.config import EMBEDDING_DIMENSION


def extract_normalized_embedding(face: Any) -> np.ndarray:
    """Return a validated, contiguous ArcFace unit vector for one face."""
    raw_value = getattr(face, "embedding", None)
    if raw_value is None:
        raise ValueError("The recognition model did not produce an embedding")

    raw = np.asarray(raw_value)
    if raw.shape != (EMBEDDING_DIMENSION,):
        raise ValueError(
            f"Expected a {EMBEDDING_DIMENSION}-dimensional embedding, got {raw.shape}"
        )
    if not np.issubdtype(raw.dtype, np.number) or not np.all(np.isfinite(raw)):
        raise ValueError("The recognition model produced a non-finite embedding")

    raw_norm = float(np.linalg.norm(raw))
    if not np.isfinite(raw_norm) or raw_norm <= 0:
        raise ValueError("The recognition model produced a zero-norm embedding")

    normalized_value = getattr(face, "normed_embedding", None)
    if normalized_value is None:
        raise ValueError("The recognition model did not normalize the embedding")

    normalized = np.asarray(normalized_value, dtype=np.float32)
    if normalized.shape != (EMBEDDING_DIMENSION,):
        raise ValueError(
            f"Expected a {EMBEDDING_DIMENSION}-dimensional normalized embedding, "
            f"got {normalized.shape}"
        )
    if not np.all(np.isfinite(normalized)):
        raise ValueError("The recognition model produced a non-finite embedding")

    norm = float(np.linalg.norm(normalized))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("The recognition model produced a zero-norm embedding")

    # Re-normalize after the float32 conversion so the stored representation is
    # the canonical unit vector used by downstream cosine similarity.
    normalized = normalized / np.float32(norm)
    return np.ascontiguousarray(normalized, dtype=np.float32)


def embedding_metadata(embedding: np.ndarray) -> dict[str, object]:
    return {
        "dimension": int(embedding.shape[0]),
        "dtype": str(embedding.dtype),
        "normalized": True,
        "norm": float(np.linalg.norm(embedding)),
    }
