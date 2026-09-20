import logging

from fastapi import APIRouter, HTTPException

from facial_pipeline.schemas import ImageRequest
from facial_pipeline.service import FacialLandmarkPipeline


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")
pipeline = FacialLandmarkPipeline()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/detect")
def detect(request: ImageRequest) -> dict:
    try:
        return pipeline.process(request.image)
    except Exception as error:
        logger.exception("Face detection failed")
        raise HTTPException(status_code=400, detail=str(error)) from error
