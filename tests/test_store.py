"""向量库单测：持久化、检索排序、删除、IDF 落盘。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.store import VectorStore, cosine


def _record(record_id: str, text: str, vector: list[float], source: str = "a.md") -> dict:
    return {
        "id": record_id,
        "text": text,
        "source": source,
        "page": None,
        "section": None,
        "vector": vector,
        "file_hash": "h1",
    }


class TestVectorStore(unittest.TestCase):
    def test_cosine(self):
        self.assertAlmostEqual(cosine([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine([1, 0], [0, 1]), 0.0)
        self.assertEqual(cosine([0, 0], [1, 1]), 0.0)

    def test_add_search_persist(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.json"
            store = VectorStore(path)
            store.scoring = "dot"
            store.add_chunks(
                [
                    _record("1", "试用期三个月", [1.0, 0.0]),
                    _record("2", "年假五天", [0.0, 1.0]),
                ]
            )
            self.assertEqual(store.count(), 2)

            reloaded = VectorStore(path)
            self.assertEqual(reloaded.count(), 2)
            self.assertEqual(reloaded.scoring, "dot")
            hits = reloaded.search([1.0, 0.0], top_k=1)
            self.assertEqual(hits[0]["id"], "1")
            self.assertAlmostEqual(hits[0]["score"], 1.0)

    def test_remove_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = VectorStore(Path(tmp) / "index.json")
            store.add_chunks([_record("1", "a", [1.0]), _record("2", "b", [1.0], source="b.md")])
            removed = store.remove_source("a.md")
            self.assertEqual(removed, 1)
            self.assertEqual(store.count(), 1)
            self.assertEqual(store.sources()["b.md"]["chunks"], 1)

    def test_duplicate_id_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = VectorStore(Path(tmp) / "index.json")
            store.add_chunks([_record("1", "a", [1.0])])
            store.add_chunks([_record("1", "重复", [0.0])])
            self.assertEqual(store.count(), 1)

    def test_empty_store_returns_no_hits(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = VectorStore(Path(tmp) / "index.json")
            self.assertEqual(store.search([1.0, 0.0]), [])


if __name__ == "__main__":
    unittest.main()
