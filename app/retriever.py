"""检索器（初版 V1）：纯向量 Top-K + 低置信度判断。

低置信度判断是初版就有的抗幻觉设计：检索最高分低于阈值时，
generator 直接拒答，根本不调用大模型——省钱、快、且不可能编。
"""
from __future__ import annotations

import logging

from app import config, llm
from app.store import VectorStore

logger = logging.getLogger(__name__)


def retrieve(
    question: str,
    top_k: int | None = None,
    store: VectorStore | None = None,
    min_score: float | None = None,
) -> dict:
    """返回 {"hits", "best_score", "low_confidence", "top_k"}。

    min_score 只给评测/调参用：不传就用配置里的阈值。
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("问题不能为空")

    store = store or VectorStore()
    if store.count() == 0:
        raise RuntimeError("知识库为空，请先运行：python -m app.cli ingest")

    threshold = config.MIN_SCORE if min_score is None else min_score
    expected_scoring = "dot" if llm.is_offline() else "cosine"
    if store.scoring != expected_scoring:
        raise RuntimeError(
            f"索引打分方式（{store.scoring}）与当前模式（{expected_scoring}）不一致，"
            "向量空间不可比。请重建索引：python -m app.cli ingest --rebuild"
        )
    # 查询侧使用建库时统计的 IDF（离线模式需要；在线模式被忽略）
    llm.set_idf(store.idf, store.idf_default, set(store.stop_grams))
    query_vector = llm.embed([question], query=True)[0]
    hits = store.search(query_vector, top_k or config.TOP_K)
    best_score = hits[0]["score"] if hits else 0.0
    return {
        "hits": hits,
        "best_score": best_score,
        "low_confidence": best_score < threshold,
        "top_k": top_k or config.TOP_K,
        "threshold": threshold,
    }
