"""Define immutable intermediate values shared by pipeline stages."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProcessedFace:
    """Private pipeline result; API serialization intentionally omits raw embeddings."""
    box: dict[str, int]
    confidence: float
    landmarks: np.ndarray
    aligned_image: np.ndarray
    model_input: np.ndarray
    embedding: np.ndarray


@dataclass(frozen=True)
class ProcessedImage:
    """Processed faces together with the source dimensions needed for overlay scaling."""
    width: int
    height: int
    faces: list[ProcessedFace]
