from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = "buffalo_l"
MODEL_PROVIDERS = ["CPUExecutionProvider"]
DETECTION_INPUT_SIZE = (640, 640)

MODEL_INPUT_SIZE = (112, 112)
MODEL_INPUT_MEAN = 127.5
MODEL_INPUT_STD = 127.5
