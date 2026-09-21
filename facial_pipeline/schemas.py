"""Declare and validate the image payloads accepted by the HTTP API."""

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints


class ImageRequest(BaseModel):
    """A base64-encoded image supplied to detection or enrollment."""
    image: str


class EnrollmentRequest(ImageRequest):
    """An image plus a display name and optional selected detector result."""
    subject_id: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=128),
    ]
    # This indexes detector output for the submitted image, not a browser track ID.
    face_index: int | None = Field(default=None, ge=0)
