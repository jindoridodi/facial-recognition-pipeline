import base64

import cv2
import numpy as np
from insightface.utils import face_align

from facial_pipeline.config import (
    MODEL_INPUT_MEAN,
    MODEL_INPUT_SIZE,
    MODEL_INPUT_STD,
)


def decode_image(data_url: str) -> np.ndarray:
    encoded = data_url.split(",", 1)[-1]
    image = cv2.imdecode(
        np.frombuffer(base64.b64decode(encoded), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise ValueError("The image could not be decoded")
    return image


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


def prepare_model_input(aligned_face: np.ndarray) -> np.ndarray:
    """Convert an aligned OpenCV image into a normalized NCHW RGB tensor."""
    if aligned_face.shape[:2] != MODEL_INPUT_SIZE[::-1]:
        aligned_face = cv2.resize(
            aligned_face,
            MODEL_INPUT_SIZE,
            interpolation=cv2.INTER_LINEAR,
        )

    rgb_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
    normalized = (rgb_face.astype(np.float32) - MODEL_INPUT_MEAN) / MODEL_INPUT_STD
    chw = np.transpose(normalized, (2, 0, 1))
    return np.ascontiguousarray(chw[np.newaxis, ...], dtype=np.float32)
