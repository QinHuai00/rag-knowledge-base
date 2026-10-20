"""规则精排（V4）：cross-encoder 的零依赖替代品。

## 想解决什么

混合召回（V3）之后，候选集里仍然混着"词都沾边、其实答不了"的块。典型例子：
问"设备更换要满几年"，V1/V3 的 Top-1 是"新员工领用"那一节（都有"设备"两个字），
真正的答案在"2.2 设备更换"里排第 3。

## 为什么不用真正的 cross-encoder

BGE-reranker / bge-m3 这类模型当然更好，但：
1) 需要 2GB+ 权重和 GPU，离线跑不动，项目就失去"clone 即跑"的属性；
2) 接 API 每千次多一次网络往返，延迟从 15ms 涨到 300ms+，全量评测要做 40 次。
所以这里做**规则精排**：用四个可解释信号加权，把"这条 chunk 到底在不在回答这个问题"
拆成可计算的分量。``Reranker`` 预留了 ``backend`` 参数（默认 ``"lexical"``），
将来接真实 cross-encoder 时在 ``score()`` 内按 backend 分派即可，对外接口不变；
当前**没有**实现 ``backend="api"`` 这种一键在线重排。

## 四个信号

| 信号 | 含义 | 为什么有用 |
| --- | --- | --- |
| coverage | 问题词在 chunk 里出现了多少 | 最直接的"相关性"近似 |
| section | 问题词在**小节标题**里出现了多少 | 制度文档里小节标题就是"这段在讲什么"的标签 |
| prior | 融合阶段的名次换算成 0~1 | 不让精排把强召回结果全部推翻 |
| proximity | 问题词是否**挤在同一小段**里 | "设备更换满 3 年"挨着出现 >> 分散在三个段落 |

注意：精排**只改顺序，不改拒答判定**。拒答仍用向量原始分（``hit["score"]``），
这样加精排不会把已有阈值搞失效，实验可以真正做到"一次只改一个变量"。
"""
from __future__ import annotations

from app import config
from app.bm25 import query_terms, tokenize

# 邻近度的时间窗口（字符数）。中文一句话大约 20~40 字，
# 150 字约等于 4~6 句：命中词落在同一个窗口里基本就是同一段论述。
PROXIMITY_WINDOW = 150


def _coverage(terms: list[str], text: str) -> float:
    """问题词覆盖率：去重后的 query term 有多少比例出现在 text 里。"""
    if not terms or not text:
        return 0.0
    doc_terms = set(tokenize(text))
    if not doc_terms:
        return 0.0
    return sum(1 for term in terms if term in doc_terms) / len(terms)


def _proximity(terms: list[str], text: str) -> float:
    """邻近度：问题词在文本里是否聚集。

    做法：取每个 term 的**首次出现位置**，按位置排序后滑窗，
    找出"覆盖最多不同 term 的最短区间"，再按区间长度衰减。

    只看首次出现是有意的：制度文档里同一个词反复出现很正常，
    用全部位置会让窗口计算变成 O(n·m) 且更容易误判（词在不同段落各出现一次也算"聚集"）。
    """
    if not terms or not text:
        return 0.0
    lowered = text.lower()
    positions: list[tuple[int, str]] = []
    for term in terms:
        index = lowered.find(term)
        if index >= 0:
            positions.append((index, term))
    if len(positions) < 2:
        return 0.0  # 只命中一个词谈不上"聚集"

    positions.sort()
    best_density = 0.0
    left = 0
    for right in range(len(positions)):
        # 窗口右移，左侧收缩到刚好不超过 PROXIMITY_WINDOW
        while positions[right][0] - positions[left][0] > PROXIMITY_WINDOW:
            left += 1
        span = positions[right][0] - positions[left][0]
        distinct = len({term for _, term in positions[left : right + 1]})
        density = (distinct / len(terms)) / (1 + span / PROXIMITY_WINDOW)
        best_density = max(best_density, density)
    return best_density


class Reranker:
    """规则精排器。``backend="lexical"`` 是默认且完全离线的实现。"""

    def __init__(
        self,
        backend: str = "lexical",
        w_coverage: float | None = None,
        w_section: float | None = None,
        w_prior: float | None = None,
        w_proximity: float | None = None,
    ):
        self.backend = backend
        self.w_coverage = config.RERANK_W_COVERAGE if w_coverage is None else w_coverage
        self.w_section = config.RERANK_W_SECTION if w_section is None else w_section
        self.w_prior = config.RERANK_W_PRIOR if w_prior is None else w_prior
        self.w_proximity = config.RERANK_W_PROXIMITY if w_proximity is None else w_proximity

    def score(self, question: str, hit: dict, prior: float) -> tuple[float, dict]:
        """给单个候选打分，返回 (总分, 分量明细)。分量明细进 trace，方便排查"为什么排第一"。"""
        terms = query_terms(question)
        coverage = _coverage(terms, hit.get("text") or "")
        section = _coverage(terms, hit.get("section") or "")
        proximity = _proximity(terms, hit.get("text") or "")
        total = (
            self.w_coverage * coverage
            + self.w_section * section
            + self.w_prior * prior
            + self.w_proximity * proximity
        )
        return total, {
            "coverage": round(coverage, 3),
            "section": round(section, 3),
            "prior": round(prior, 3),
            "proximity": round(proximity, 3),
        }

    def rerank(self, question: str, hits: list[dict], top_k: int) -> list[dict]:
        """重排并截断。输入需已按融合分降序（用来算 prior）。"""
        if not hits:
            return []
        total = len(hits)
        scored: list[dict] = []
        for rank, hit in enumerate(hits):
            prior = 1.0 - (rank / (total - 1)) if total > 1 else 1.0
            value, parts = self.score(question, hit, prior)
            item = dict(hit)
            item["rerank_score"] = round(value, 4)
            item["rerank_parts"] = parts
            scored.append(item)
        scored.sort(key=lambda item: (item["rerank_score"], item.get("score") or 0.0), reverse=True)
        return scored[:top_k]
