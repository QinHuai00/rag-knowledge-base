"""BM25 稀疏检索：纯 Python 实现，零第三方依赖。

## 为什么 V1 需要它

V1 只有向量一路召回。离线哈希向量是"字符 n-gram 词袋"，本质上是**模糊的**字面匹配：
它能容忍"报销要多久"和"报销款多久到账"这种措辞差异，但对
``HR-2024-07``、``vpn.example.com``、``5000 元`` 这类**精确编号/专有名词**很不敏感——
哈希把字符串打散到 2048 维里，编号的区分度被稀释掉了。

BM25 恰好相反：它数精确的词频，编号出现就是出现，不出现就是 0 分。
两条路互补，所以做混合召回（RRF 融合，见 ``app/retriever.py``）。

## 为什么不用 jieba

1) 项目纪律是"下载即能跑"，多一个依赖就多一道劝退；
2) 中文 BM25 用**字符 2-gram** 代替分词，召回率通常只差 1~3 个点，但零依赖、零词典；
3) 制度类文档里有大量英文缩写和编号，字符级切分对它们反而更稳（不会被词典切错）。

## 打分公式

.. math::

    \\text{score}(q, d) = \\sum_{t \\in q} \\text{idf}(t) \\cdot
    \\frac{f(t,d)\\,(k_1 + 1)}{f(t,d) + k_1 \\left(1 - b + b \\frac{|d|}{\\text{avgdl}}\\right)}

``k1`` 控制词频饱和速度（出现 3 次和出现 10 次的差距不应是 3 倍多），
``b`` 控制文档长度归一强度（长文档天然更容易命中，要压一压）。
两个参数都在 ``app/config.py`` 里，默认取 k1=1.5 / b=0.75（BM25 的常规取值）。
"""
from __future__ import annotations

import math
import re
from collections import Counter

from app import config

# 英文/数字/编号：连字符、下划线、点、斜杠都算"同一个 token 的一部分"，
# 这样 "HR-2024-07" 不会被切成 "hr" + "2024" + "07" 三个无意义片段。
_LATIN_RE = re.compile(r"[a-z0-9]+(?:[-_./][a-z0-9]+)*")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_PART_RE = re.compile(r"[-_./]")

# 查询侧停用词：问句里的虚词、泛化词对 BM25 没有区分度，
# 留着只会让"什么/怎么/多久"去匹配到所有文档。
QUERY_STOP = {
    "什么", "怎么", "怎样", "如何", "哪些", "哪个", "哪里", "多少", "多久", "几天", "几次",
    "是否", "可以", "能否", "请问", "一下", "需要", "这个", "那个", "介绍", "说说", "告诉",
    "具体", "一般", "公司", "员工", "我们", "规定", "要求", "情况",
}


def tokenize(text: str) -> list[str]:
    """文本 → token 列表（保留重复，BM25 要用词频）。

    规则：
    - 英文/数字/编号整体保留，同时**额外**拆出片段（"hr-2024-07" → 整体 + hr + 2024 + 07），
      这样查询写全称能命中，只写 "2024" 也能命中；
    - 中文按字符 2-gram 切；单字串保留单字（否则"病"这种短标题会没有任何 token）。
    """
    if not text:
        return []
    lowered = text.lower().replace("\u3000", " ")
    tokens: list[str] = []
    for match in _LATIN_RE.finditer(lowered):
        whole = match.group()
        tokens.append(whole)
        parts = [part for part in _PART_RE.split(whole) if part]
        if len(parts) > 1:
            tokens.extend(parts)
    for segment in _CJK_RE.findall(lowered):
        if len(segment) == 1:
            tokens.append(segment)
            continue
        for index in range(len(segment) - 1):
            tokens.append(segment[index : index + 2])
    return tokens


def query_terms(question: str) -> list[str]:
    """查询侧 token，去停用词、去重、保序（用于覆盖率类指标）。"""
    seen: set[str] = set()
    terms: list[str] = []
    for token in tokenize(question):
        if token in QUERY_STOP or token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return terms


class BM25Index:
    """内存倒排索引。语料几百~几千块时构建只需几毫秒，不需要持久化。"""

    def __init__(self, docs: list[list[str]], k1: float | None = None, b: float | None = None):
        self.k1 = config.BM25_K1 if k1 is None else k1
        self.b = config.BM25_B if b is None else b
        self.size = len(docs)
        self.tfs = [Counter(doc) for doc in docs]
        self.lengths = [len(doc) for doc in docs]
        self.avgdl = (sum(self.lengths) / self.size) if self.size else 0.0
        # IDF：用 +0.5 平滑，保证只出现一次的词也有正分（否则专有名词会被判 0）
        df: Counter = Counter()
        for doc in docs:
            df.update(set(doc))
        self.idf = {
            term: math.log(1 + (self.size - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    @classmethod
    def from_texts(cls, texts: list[str], **kwargs) -> "BM25Index":
        return cls([tokenize(text) for text in texts], **kwargs)

    def scores(self, question: str) -> list[float]:
        """对全部文档打分，返回与语料等长的分数列表。"""
        if not self.size:
            return []
        terms = query_terms(question)
        if not terms:
            return [0.0] * self.size
        scores = [0.0] * self.size
        avgdl = self.avgdl or 1.0
        for term in terms:
            idf = self.idf.get(term)
            if idf is None:  # 语料里没出现过的词，对排序没贡献
                continue
            for index in range(self.size):
                freq = self.tfs[index].get(term)
                if not freq:
                    continue
                norm = self.k1 * (1 - self.b + self.b * self.lengths[index] / avgdl)
                scores[index] += idf * freq * (self.k1 + 1) / (freq + norm)
        return scores

    def search(self, question: str, top_n: int = 20) -> list[tuple[int, float]]:
        """返回 [(文档下标, 分数)]，按分数降序，只保留正分。"""
        scores = self.scores(question)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(index, scores[index]) for index in ranked[:top_n] if scores[index] > 0]
