from typing import Annotated

from pydantic import BaseModel, StringConstraints


class ImageRequest(BaseModel):
    image: str


class EnrollmentRequest(ImageRequest):
    subject_id: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=128),
    ]
