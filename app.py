import base64
import logging
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from insightface.app import FaceAnalysis
from insightface.utils import face_align


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
MODEL_INPUT_SIZE = (112, 112)
app = FastAPI(title="Facial Recognition Prototype")
app.mount("/assets", StaticFiles(directory=ROOT / "assets"), name="assets")
face_app: FaceAnalysis | None = None


class ImageRequest(BaseModel):
    image: str


def encode_jpeg(image: np.ndarray) -> str:
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        raise ValueError("The aligned face could not be encoded")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def align_face(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Affine-warp a face into the ArcFace model's 112x112 input space."""
    if landmarks.shape != (5, 2):
        raise ValueError("Face alignment requires exactly five 2D landmarks")

    transform = face_align.estimate_norm(
        landmarks,
        image_size=MODEL_INPUT_SIZE[0],
        mode="arcface",
    )
    return cv2.warpAffine(
        image,
        transform,
        MODEL_INPUT_SIZE,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def get_face_app() -> FaceAnalysis:
    global face_app
    if face_app is None:
        logger.info("Loading InsightFace models...")
        face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("InsightFace ready")
    return face_app


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/detect")
def detect(request: ImageRequest) -> dict:
    try:
        encoded = request.image.split(",", 1)[-1]
        image = cv2.imdecode(
            np.frombuffer(base64.b64decode(encoded), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            raise ValueError("The image could not be decoded")

        faces = get_face_app().get(image)
        results = []
        for face in faces:
            x1, y1, x2, y2 = [round(float(value)) for value in face.bbox]
            landmarks = np.asarray(face.kps, dtype=np.float32)
            aligned = align_face(image, landmarks)
            results.append({
                "box": {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
                "confidence": round(float(face.det_score), 3),
                "landmarks": [
                    {"x": round(float(x), 2), "y": round(float(y), 2)}
                    for x, y in landmarks
                ],
                "aligned": encode_jpeg(aligned),
            })
        return {
            "faces": results,
            "width": image.shape[1],
            "height": image.shape[0],
            "model_input": {"width": MODEL_INPUT_SIZE[0], "height": MODEL_INPUT_SIZE[1]},
        }
    except Exception as error:
        logger.exception("Face detection failed")
        raise HTTPException(status_code=400, detail=str(error)) from error
