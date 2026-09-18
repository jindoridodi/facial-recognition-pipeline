import base64
import logging
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from insightface.app import FaceAnalysis


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="Facial Recognition Prototype")
face_app: FaceAnalysis | None = None


class ImageRequest(BaseModel):
    image: str


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
            results.append({
                "box": {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
                "confidence": round(float(face.det_score), 3),
            })
        return {"faces": results, "width": image.shape[1], "height": image.shape[0]}
    except Exception as error:
        logger.exception("Face detection failed")
        raise HTTPException(status_code=400, detail=str(error)) from error
