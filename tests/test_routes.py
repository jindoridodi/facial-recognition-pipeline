import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException

from facial_pipeline.embedding_store import EmbeddingRecord


# Route tests exercise orchestration only and do not need to load InsightFace.
service_stub = ModuleType("facial_pipeline.service")
service_stub.FacialLandmarkPipeline = lambda: None
sys.modules["facial_pipeline.service"] = service_stub

from facial_pipeline.routes import detect, enroll  # noqa: E402
from facial_pipeline.schemas import EnrollmentRequest, ImageRequest  # noqa: E402


def vector(index: int = 0) -> np.ndarray:
    result = np.zeros(512, dtype=np.float32)
    result[index] = 1.0
    return result


class FakePipeline:
    def __init__(self, embeddings: list[np.ndarray]) -> None:
        self.result = SimpleNamespace(
            faces=[SimpleNamespace(embedding=value) for value in embeddings]
        )

    def analyze(self, _image: str) -> SimpleNamespace:
        return self.result

    @staticmethod
    def detection_response(result: SimpleNamespace) -> dict:
        return {
            "faces": [
                {"embedding": {"dimension": 512, "normalized": True}}
                for _face in result.faces
            ]
        }


class FakeStore:
    def __init__(self, records: list[EmbeddingRecord] | None = None) -> None:
        self.records = records or []
        self.saved: tuple[str, np.ndarray] | None = None

    def list_all(self) -> list[EmbeddingRecord]:
        return self.records

    def save(self, name: str, embedding: np.ndarray) -> EmbeddingRecord:
        self.saved = (name, embedding)
        return make_record(name, embedding)


def make_record(name: str, embedding: np.ndarray) -> EmbeddingRecord:
    return EmbeddingRecord(
        embedding_id="4ae88164-e069-4e5b-8f19-e69f63b08a96",
        subject_id=name,
        embedding=embedding,
        dimension=512,
        dtype="float32",
        model_name="buffalo_l",
        schema_version=1,
        normalized=True,
        created_at="2026-09-20T00:00:00Z",
    )


class RouteTests(unittest.TestCase):
    def test_detect_adds_saved_name_without_exposing_vector(self) -> None:
        stored = make_record("Dorothy Vaughan", vector())
        with (
            patch("facial_pipeline.routes.pipeline", FakePipeline([vector()])),
            patch("facial_pipeline.routes.embedding_store", FakeStore([stored])),
        ):
            response = detect(ImageRequest(image="image"))

        face = response["faces"][0]
        self.assertEqual(face["identity"]["name"], "Dorothy Vaughan")
        self.assertNotIn("vector", face["identity"])
        self.assertNotIn("embedding", face["identity"])

    def test_detect_marks_an_unknown_face_with_null_identity(self) -> None:
        stored = make_record("Mary Jackson", vector(1))
        with (
            patch("facial_pipeline.routes.pipeline", FakePipeline([vector(2)])),
            patch("facial_pipeline.routes.embedding_store", FakeStore([stored])),
        ):
            response = detect(ImageRequest(image="image"))

        self.assertIsNone(response["faces"][0]["identity"])

    def test_enroll_can_select_a_face_and_returns_its_name(self) -> None:
        store = FakeStore()
        with (
            patch("facial_pipeline.routes.pipeline", FakePipeline([vector(1), vector(2)])),
            patch("facial_pipeline.routes.embedding_store", store),
        ):
            response = enroll(
                EnrollmentRequest(
                    image="image",
                    subject_id="  Annie Easley  ",
                    face_index=1,
                )
            )

        self.assertEqual(response["name"], "Annie Easley")
        self.assertEqual(store.saved[0], "Annie Easley")
        np.testing.assert_array_equal(store.saved[1], vector(2))

    def test_enroll_without_index_remains_single_face_only(self) -> None:
        store = FakeStore()
        with (
            patch("facial_pipeline.routes.pipeline", FakePipeline([vector(4)])),
            patch("facial_pipeline.routes.embedding_store", store),
        ):
            enroll(EnrollmentRequest(image="image", subject_id="Solo Person"))
        np.testing.assert_array_equal(store.saved[1], vector(4))

        with patch(
            "facial_pipeline.routes.pipeline",
            FakePipeline([vector(4), vector(5)]),
        ):
            with self.assertRaises(HTTPException) as raised:
                enroll(EnrollmentRequest(image="image", subject_id="Ambiguous"))
        self.assertEqual(raised.exception.status_code, 422)

    def test_enroll_rejects_an_unavailable_face_index(self) -> None:
        with patch("facial_pipeline.routes.pipeline", FakePipeline([vector()])):
            with self.assertRaises(HTTPException) as raised:
                enroll(
                    EnrollmentRequest(
                        image="image",
                        subject_id="Missing",
                        face_index=3,
                    )
                )
        self.assertEqual(raised.exception.status_code, 422)


if __name__ == "__main__":
    unittest.main()
