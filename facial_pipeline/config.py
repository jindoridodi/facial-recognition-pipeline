"""Centralize model, vector-format, and environment-backed configuration."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = "buffalo_l"
MODEL_PROVIDERS = ["CPUExecutionProvider"]
DETECTION_INPUT_SIZE = (640, 640)

MODEL_INPUT_SIZE = (112, 112)
MODEL_INPUT_MEAN = 127.5
MODEL_INPUT_STD = 127.5

EMBEDDING_DIMENSION = 512
EMBEDDING_DTYPE = "float32"
EMBEDDING_SCHEMA_VERSION = 1
EMBEDDING_DB_PATH = Path(
    os.environ.get("EMBEDDING_DB_PATH", PROJECT_ROOT / "data" / "embeddings.sqlite3")
)

FACE_MATCH_THRESHOLD = float(os.environ.get("FACE_MATCH_THRESHOLD", "0.4"))
if not -1.0 <= FACE_MATCH_THRESHOLD <= 1.0:
    raise ValueError("FACE_MATCH_THRESHOLD must be between -1.0 and 1.0")
