"""Expose detection, enrollment, and embedding-deletion HTTP endpoints."""

import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, Response, status

from facial_pipeline.config import EMBEDDING_DB_PATH, FACE_MATCH_THRESHOLD
from facial_pipeline.embedding_store import EmbeddingStore
from facial_pipeline.matching import find_best_match
from facial_pipeline.schemas import EnrollmentRequest, ImageRequest
from facial_pipeline.service import FacialLandmarkPipeline


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")
pipeline = FacialLandmarkPipeline()
embedding_store = EmbeddingStore(EMBEDDING_DB_PATH)


@router.get("/health")
def health() -> dict[str, str]:
    """Report that the API process is responding without loading the model."""
    return {"status": "ok"}


@router.post("/detect")
def detect(request: ImageRequest) -> dict:
    """Detect faces and attach any compatible saved identity to each response face."""
    try:
        result = pipeline.analyze(request.image)
    except Exception as error:
        logger.exception("Face detection failed")
        raise HTTPException(status_code=400, detail=str(error)) from error

    try:
        candidates = embedding_store.list_all()
        response = pipeline.detection_response(result)
        # Keep embeddings inside the service layer; clients receive only match metadata.
        for face, processed_face in zip(response["faces"], result.faces):
            match = find_best_match(
                processed_face.embedding,
                candidates,
                FACE_MATCH_THRESHOLD,
            )
            face["identity"] = match.metadata() if match is not None else None
        return response
    except Exception as error:
        logger.exception("Face recognition failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Saved faces could not be loaded",
        ) from error


@router.post("/enroll", status_code=status.HTTP_201_CREATED)
def enroll(request: EnrollmentRequest) -> dict[str, object]:
    """Store the requested face from one image as a named recognition sample."""
    try:
        result = pipeline.analyze(request.image)
    except Exception as error:
        logger.exception("Face embedding generation failed")
        raise HTTPException(status_code=400, detail=str(error)) from error

    if request.face_index is None and len(result.faces) != 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Enrollment requires exactly one face; detected {len(result.faces)}",
        )

    # An omitted index is unambiguous only for an image containing one face.
    face_index = request.face_index if request.face_index is not None else 0
    if face_index >= len(result.faces):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Face index {face_index} is unavailable; "
                f"detected {len(result.faces)} face(s)"
            ),
        )

    try:
        record = embedding_store.save(
            request.subject_id,
            result.faces[face_index].embedding,
        )
    except Exception as error:
        logger.exception("Embedding persistence failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The embedding could not be stored",
        ) from error
    return {**record.metadata(), "name": record.subject_id}


@router.delete(
    "/embeddings",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_all_embeddings() -> Response:
    """Remove all enrolled samples after the caller has confirmed that action."""
    try:
        embedding_store.delete_all()
    except Exception as error:
        logger.exception("Bulk embedding deletion failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Saved faces could not be cleared",
        ) from error
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete(
    "/embeddings/{embedding_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_embedding(embedding_id: UUID) -> Response:
    """Remove one enrollment sample identified by its UUID."""
    try:
        deleted = embedding_store.delete(embedding_id)
    except Exception as error:
        logger.exception("Embedding deletion failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The embedding could not be deleted",
        ) from error
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Embedding not found",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
