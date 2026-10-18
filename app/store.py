"""本地向量库：JSON 持久化 + 相似度检索。

打分方式（记录在索引文件里，建库时决定）：
- 在线模式：cosine（余弦相似度）
- 离线模式：dot（查询单位向量 · 文档原始词袋向量，避免长文档稀释短查询）

换成 Chroma 时，这一步对应 collection 的 `hnsw:space` 参数，
接口刻意做成和 Chroma 接近（add / search / count / remove_source）。
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from app import config

logger = logging.getLogger(__name__)


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度。"""
    if not a or not b:
        return 0.0
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)


class VectorStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else config.INDEX_FILE
        self.items: list[dict] = []
        self.scoring = "cosine"
        self.idf: dict[str, float] = {}
        self.idf_default = 1.0
        self.stop_grams: list[str] = []
        self._load()

    # ---------- 持久化 ----------
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.items = data.get("items", [])
            self.scoring = data.get("scoring", "cosine")
            self.idf = data.get("idf", {}) or {}
            self.idf_default = float(data.get("idf_default", 1.0))
            self.stop_grams = data.get("stop_grams", []) or []
            logger.info("向量库已加载：%d 块（%s，%s 打分）", len(self.items), self.path, self.scoring)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("向量库文件损坏，按空库处理：%s", exc)
            self.items = []

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dim": config.OFFLINE_EMBED_DIM,
            "scoring": self.scoring,
            "idf": self.idf,
            "idf_default": self.idf_default,
            "stop_grams": self.stop_grams,
            "items": self.items,
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)  # 原子替换，避免写到一半进程被杀导致库损坏

    # ---------- 写 ----------
    def add_chunks(self, records: list[dict]) -> None:
        existing = {item["id"] for item in self.items}
        for record in records:
            if record["id"] in existing:
                logger.warning("重复 id，跳过：%s", record["id"])
                continue
            self.items.append(record)
        self.save()

    def remove_source(self, source: str) -> int:
        before = len(self.items)
        self.items = [item for item in self.items if item.get("source") != source]
        removed = before - len(self.items)
        if removed:
            self.save()
        return removed

    def delete_by_source(self, source: str) -> int:
        """和 remove_source 同义，保留这个别名方便对照 Chroma API。"""
        return self.remove_source(source)

    def reset(self) -> None:
        self.items = []
        self.save()

    # ---------- 读 ----------
    def count(self) -> int:
        return len(self.items)

    def sources(self) -> dict[str, dict]:
        """source → {chunks, file_hash}，用于状态展示与增量更新判断。"""
        summary: dict[str, dict] = {}
        for item in self.items:
            entry = summary.setdefault(item["source"], {"chunks": 0, "file_hash": item.get("file_hash")})
            entry["chunks"] += 1
        return summary

    def _scores(self, query_vector: list[float]) -> list[float]:
        if not self.items:
            return []
        if config.HAS_NUMPY:
            import numpy as np

            matrix = np.asarray([item["vector"] for item in self.items], dtype=np.float32)
            query = np.asarray(query_vector, dtype=np.float32)
            if self.scoring == "dot":
                return [float(x) for x in matrix @ query]
            norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(query) or 1.0)
            norms[norms == 0] = 1.0
            return [float(x) for x in (matrix @ query) / norms]

        if self.scoring == "dot":
            return [sum(x * y for x, y in zip(query_vector, item["vector"])) for item in self.items]
        return [cosine(query_vector, item["vector"]) for item in self.items]

    def search(self, query_vector: list[float], top_k: int | None = None) -> list[dict]:
        if not self.items or not query_vector:
            return []
        k = min(top_k or config.TOP_K, len(self.items))
        scores = self._scores(query_vector)
        ranked = sorted(range(len(self.items)), key=lambda i: scores[i], reverse=True)[:k]
        hits: list[dict] = []
        for index in ranked:
            item = self.items[index]
            hits.append(
                {
                    "id": item["id"],
                    "text": item["text"],
                    "source": item.get("source"),
                    "page": item.get("page"),
                    "section": item.get("section"),
                    "score": round(scores[index], 4),
                }
            )
        return hits
