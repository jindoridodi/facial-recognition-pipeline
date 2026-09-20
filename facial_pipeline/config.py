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
