from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints


class ImageRequest(BaseModel):
    image: str


class EnrollmentRequest(ImageRequest):
    subject_id: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=128),
    ]
    face_index: int | None = Field(default=None, ge=0)
