"""检索器：向量召回 → (BM25 混合) → (规则精排) → (父块展开)。

## 一条查询经过什么

::

    问题
     ├─ 向量召回：embed → VectorStore.rank(CANDIDATE_K)   →  按相似度排序的候选 A
     └─ 稀疏召回：BM25Index.search(CANDIDATE_K)           →  按词频排序的候选 B
            └─ RRF 融合：score = Σ 1 / (k + rank_i)      →  融合候选
                  └─ 规则精排（覆盖度/标题/邻近度/原始名次） →  最终 Top-K
                        └─ 父块展开：child → parent（整节）→ 交给生成侧

## RRF 为什么比"分数加权平均"好

向量分（离线是 IDF 点积，4~40 量级）和 BM25 分（0~20 量级）**不同量纲**，
直接加权平均必须先做归一化，而归一化方式本身又是个超参。
RRF（Reciprocal Rank Fusion）只用名次、不用分数：

.. math:: \\text{RRF}(d) = \\sum_{r \\in \\text{召回器}} \\frac{1}{k + \\text{rank}_r(d)}

好处是没有量纲问题、对异常分不敏感；代价是丢掉了"第一名领先多少"的信息。
对当前这个规模的语料，丢掉的这点信息不重要，鲁棒性更重要。

## 为什么精排不影响拒答

拒答判定用的是 ``best_score``（向量原始分），不是融合分/精排分。
原因：阈值是 V1 用 ``eval/calibrate.py`` 在向量分上标定的，换成分数体系就必须重标一次，
那样每加一级优化都得重来一遍校准，"一次只改一个变量"就不成立了。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace

from app import config, llm
from app.bm25 import BM25Index
from app.reranker import Reranker
from app.store import VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalOptions:
    """检索流水线的开关组合。消融实验就是换这个对象，业务代码一行不用改。"""

    name: str = "V4"
    top_k: int = 5
    candidate_k: int = 20
    use_bm25: bool = True
    use_rerank: bool = True
    use_parent: bool = True
    rrf_k: int = 60

    def with_top_k(self, top_k: int | None) -> "RetrievalOptions":
        return self if top_k is None else replace(self, top_k=top_k)

    # ---------- 四级消融预设 ----------
    @classmethod
    def v1(cls, top_k: int | None = None) -> "RetrievalOptions":
        """V1 基线：纯向量、固定块、无精排。candidate_k 必须等于 top_k，
        否则连基线都在"看更多候选"，消融就被污染了。"""
        k = top_k or config.TOP_K
        return cls(name="V1", top_k=k, candidate_k=k, use_bm25=False, use_rerank=False, use_parent=False)

    @classmethod
    def v2(cls, top_k: int | None = None) -> "RetrievalOptions":
        """V2：父子块。检索用小块，生成用整节。"""
        k = top_k or config.TOP_K
        return cls(name="V2", top_k=k, candidate_k=k, use_bm25=False, use_rerank=False, use_parent=True)

    @classmethod
    def v3(cls, top_k: int | None = None) -> "RetrievalOptions":
        """V3：父子块 + BM25/RRF 混合召回。"""
        k = top_k or config.TOP_K
        return cls(name="V3", top_k=k, candidate_k=config.CANDIDATE_K, use_bm25=True, use_rerank=False, use_parent=True)

    @classmethod
    def v4(cls, top_k: int | None = None) -> "RetrievalOptions":
        """V4：V3 + 规则精排。"""
        k = top_k or config.TOP_K
        return cls(
            name="V4",
            top_k=k,
            candidate_k=config.CANDIDATE_K,
            use_bm25=True,
            use_rerank=True,
            use_parent=True,
            rrf_k=config.RRF_K,
        )


ALL_PRESETS = ("V1", "V2", "V3", "V4")


def _preset(name: str, top_k: int | None = None) -> RetrievalOptions:
    return getattr(RetrievalOptions, name.lower())(top_k)


class Retriever:
    """把手洗干净后的检索流水线。持有 store 与 BM25 索引，可反复调用。"""

    def __init__(
        self,
        store: VectorStore | None = None,
        options: RetrievalOptions | None = None,
        bm25: BM25Index | None = None,
    ):
        self.store = store or VectorStore()
        self.options = options or RetrievalOptions.v4()
        self._bm25 = bm25
        self._reranker = Reranker()

    @property
    def bm25(self) -> BM25Index | None:
        """懒建稀疏索引：只在 use_bm25 时才花这份 CPU。"""
        if not self.options.use_bm25:
            return None
        if self._bm25 is None:
            self._bm25 = BM25Index.from_texts(self.store.texts())
            logger.debug("BM25 索引已构建：%d 篇文档", len(self.store.items))
        return self._bm25

    # ------------------------------------------------------------------
    def _check(self) -> None:
        if self.store.count() == 0:
            raise RuntimeError("知识库为空，请先运行：python -m app.cli ingest")
        expected = "dot" if llm.is_offline() else "cosine"
        if self.store.scoring != expected:
            raise RuntimeError(
                f"索引打分方式（{self.store.scoring}）与当前模式（{expected}）不一致，"
                "向量空间不可比。请重建索引：python -m app.cli ingest --rebuild"
            )

    def _hit(self, index: int, vector_score: float) -> dict:
        item = self.store.items[index]
        return {
            "id": item["id"],
            "text": item["text"],
            "source": item.get("source"),
            "page": item.get("page"),
            "section": item.get("section"),
            "parent_id": item.get("parent_id"),
            "score": round(float(vector_score), 4),
        }

    def retrieve(self, question: str, min_score: float | None = None) -> dict:
        """返回 {"hits", "best_score", "low_confidence", "threshold", "trace"}。"""
        question = (question or "").strip()
        if not question:
            raise ValueError("问题不能为空")

        self._check()
        opts = self.options
        trace: dict = {"preset": opts.name, "candidate_k": opts.candidate_k}

        # 查询侧要用建库时统计的 IDF（离线模式必需；在线模式被忽略）
        llm.set_idf(self.store.idf, self.store.idf_default, set(self.store.stop_grams))

        started = time.perf_counter()
        query_vector = llm.embed([question], query=True)[0]
        vector_ranked = self.store.rank(query_vector, opts.candidate_k)
        trace["vector_ms"] = round((time.perf_counter() - started) * 1000, 2)
        trace["vector_candidates"] = len(vector_ranked)

        # 拒答只看向量原始分（见模块 docstring 的说明）
        best_score = max((score for _, score in vector_ranked), default=0.0)

        # ---------- RRF 融合 ----------
        started = time.perf_counter()
        fused: dict[int, float] = {}
        rrf_k = opts.rrf_k
        for rank, (index, _) in enumerate(vector_ranked):
            fused[index] = fused.get(index, 0.0) + 1.0 / (rrf_k + rank + 1)
        bm25_hits = 0
        if self.bm25 is not None:
            for rank, (index, _) in enumerate(self.bm25.search(question, opts.candidate_k)):
                fused[index] = fused.get(index, 0.0) + 1.0 / (rrf_k + rank + 1)
                bm25_hits += 1
        trace["bm25_candidates"] = bm25_hits
        trace["fuse_ms"] = round((time.perf_counter() - started) * 1000, 2)

        raw_scores = dict(vector_ranked)
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        hits = [dict(self._hit(index, raw_scores.get(index, 0.0)), fused_score=round(value, 6)) for index, value in ordered]

        # ---------- 精排 ----------
        started = time.perf_counter()
        if opts.use_rerank:
            hits = self._reranker.rerank(question, hits[: config.RERANK_TOP_N], opts.top_k)
        else:
            hits = hits[: opts.top_k]
        trace["rerank_ms"] = round((time.perf_counter() - started) * 1000, 2)

        # ---------- 父块展开 ----------
        started = time.perf_counter()
        if opts.use_parent:
            for hit in hits:
                parent = self.store.get_parent(hit.get("parent_id"))
                if parent:
                    context = parent["text"]
                    hit["context"] = context[: config.PARENT_MAX_CHARS]
                    hit["context_truncated"] = len(context) > config.PARENT_MAX_CHARS
        trace["expand_ms"] = round((time.perf_counter() - started) * 1000, 2)
        trace["returned"] = len(hits)
        trace["context_chars"] = sum(len(hit.get("context") or hit["text"]) for hit in hits)

        threshold = config.MIN_SCORE if min_score is None else min_score
        return {
            "hits": hits,
            "best_score": round(float(best_score), 4),
            "low_confidence": best_score < threshold,
            "top_k": opts.top_k,
            "threshold": threshold,
            "trace": trace,
        }


def retrieve(
    question: str,
    top_k: int | None = None,
    store: VectorStore | None = None,
    min_score: float | None = None,
    options: RetrievalOptions | None = None,
) -> dict:
    """一次性检索（向后兼容的入口）。

    需要连续跑很多查询（评测、批量）时，直接用 ``Retriever`` 实例：
    BM25 索引只建一次，实测 40 题评测快约 4 倍。
    """
    opts = (options or RetrievalOptions.v4()).with_top_k(top_k)
    return Retriever(store=store, options=opts).retrieve(question, min_score=min_score)
