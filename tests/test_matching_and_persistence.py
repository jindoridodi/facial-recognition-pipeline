import tempfile
import unittest
from pathlib import Path

import numpy as np

from facial_pipeline.embedding_store import EmbeddingStore
from facial_pipeline.matching import find_best_match


def unit_vector(index: int) -> np.ndarray:
    vector = np.zeros(512, dtype=np.float32)
    vector[index] = 1.0
    return vector


class MatchingAndPersistenceTests(unittest.TestCase):
    def test_name_and_vector_survive_a_new_store_instance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.sqlite3"
            saved = EmbeddingStore(path).save("Ada Lovelace", unit_vector(7))

            records = EmbeddingStore(path).list_all()

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].embedding_id, saved.embedding_id)
            self.assertEqual(records[0].subject_id, "Ada Lovelace")
            np.testing.assert_array_equal(records[0].embedding, unit_vector(7))

    def test_best_match_returns_name_without_vector_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = EmbeddingStore(Path(directory) / "embeddings.sqlite3")
            store.save("Grace Hopper", unit_vector(3))
            store.save("Katherine Johnson", unit_vector(8))

            match = find_best_match(unit_vector(3), store.list_all(), threshold=0.4)

            self.assertIsNotNone(match)
            metadata = match.metadata()
            self.assertEqual(metadata["name"], "Grace Hopper")
            self.assertEqual(metadata["subject_id"], "Grace Hopper")
            self.assertEqual(metadata["similarity"], 1.0)
            self.assertNotIn("embedding", metadata)
            self.assertNotIn("vector", metadata)

    def test_match_below_threshold_is_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = EmbeddingStore(Path(directory) / "embeddings.sqlite3")
            store.save("Margaret Hamilton", unit_vector(1))

            match = find_best_match(unit_vector(2), store.list_all(), threshold=0.4)

            self.assertIsNone(match)


if __name__ == "__main__":
    unittest.main()
