from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProcessedFace:
    box: dict[str, int]
    confidence: float
    landmarks: np.ndarray
    aligned_image: np.ndarray
    model_input: np.ndarray
    embedding: np.ndarray


@dataclass(frozen=True)
class ProcessedImage:
    width: int
    height: int
    faces: list[ProcessedFace]
