import logging
from typing import Any

import numpy as np
from insightface.app import FaceAnalysis

from facial_pipeline.config import (
    DETECTION_INPUT_SIZE,
    MODEL_INPUT_SIZE,
    MODEL_NAME,
    MODEL_PROVIDERS,
)
from facial_pipeline.image_ops import (
    align_face,
    decode_image,
    encode_jpeg,
    prepare_model_input,
)


logger = logging.getLogger(__name__)


class FacialLandmarkPipeline:
    def __init__(self) -> None:
        self._analyzer: FaceAnalysis | None = None

    def _get_analyzer(self) -> FaceAnalysis:
        if self._analyzer is None:
            logger.info("Loading InsightFace models...")
            self._analyzer = FaceAnalysis(
                name=MODEL_NAME,
                providers=MODEL_PROVIDERS,
            )
            self._analyzer.prepare(ctx_id=0, det_size=DETECTION_INPUT_SIZE)
            logger.info("InsightFace ready")
        return self._analyzer

    def process(self, data_url: str) -> dict[str, Any]:
        image = decode_image(data_url)
        faces = self._get_analyzer().get(image)
        results = []

        for face in faces:
            x1, y1, x2, y2 = [round(float(value)) for value in face.bbox]
            landmarks = np.asarray(face.kps, dtype=np.float32)
            aligned = align_face(image, landmarks)
            model_input = prepare_model_input(aligned)
            results.append({
                "box": {
                    "x": x1,
                    "y": y1,
                    "width": x2 - x1,
                    "height": y2 - y1,
                },
                "confidence": round(float(face.det_score), 3),
                "landmarks": [
                    {"x": round(float(x), 2), "y": round(float(y), 2)}
                    for x, y in landmarks
                ],
                "aligned": encode_jpeg(aligned),
                "tensor_shape": list(model_input.shape),
            })

        return {
            "faces": results,
            "width": image.shape[1],
            "height": image.shape[0],
            "model_input": {
                "shape": [1, 3, MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0]],
                "layout": "NCHW",
                "color_order": "RGB",
                "dtype": "float32",
                "normalization": "(pixel - 127.5) / 127.5",
            },
        }
