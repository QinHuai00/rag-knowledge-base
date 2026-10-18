"""LLM / Embedding 接入层（Provider 抽象）。

两个实现：
- 在线：OpenAI 兼容协议（硅基流动、DeepSeek、百炼……只换 base_url/model）
- 离线：字符 n-gram 哈希向量 + 抽取式回答（不联网、不装包）

为什么要有离线实现：
1) 项目初版必须"下载即能跑"，否则学习者卡在配 Key 这一步就放弃了；
2) 离线实现是确定性的，评测数字可复现，便于做前后对比实验；
3) 它诚实地暴露 RAG 的骨架：检索 → 组装上下文 → 生成，与是否用大模型无关。
注意：哈希向量只做"字面相似"，不代表真实语义检索效果，不能用它写简历数字。
"""
from __future__ import annotations

import hashlib
import logging
import math
import re

from app import config

logger = logging.getLogger(__name__)


class OfflineLLMError(RuntimeError):
    """离线模式下调用了真实 LLM。调用方应降级为抽取式回答。"""


_client = None
_mode = "offline"

if not config.OFFLINE:
    try:
        from openai import OpenAI

        _client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL, timeout=120)
        _mode = "online"
    except Exception as exc:  # 初始化失败也降级，不让整个应用起不来
        logger.warning("OpenAI 客户端初始化失败，回退离线模式：%s", exc)


def is_offline() -> bool:
    return _mode == "offline"


def provider_info() -> dict:
    if _mode == "offline":
        return {
            "mode": "offline",
            "embedding": f"hash-ngram-{config.OFFLINE_EMBED_DIM}d",
            "chat": "extractive",
            "note": "未配置 API Key 或未安装 openai，使用本地哈希向量 + 抽取式回答",
        }
    return {
        "mode": "online",
        "embedding": config.EMBED_MODEL,
        "chat": config.CHAT_MODEL,
        "base_url": config.BASE_URL,
        "note": "真实模型模式",
    }


# --------------------------------------------------------------------------
# 离线 embedding：字符 n-gram 哈希 → 定长向量 → L2 归一化
# --------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


def _normalize(text: str) -> str:
    return text.lower().replace("\u3000", " ")


def _grams(text: str) -> list[tuple[str, float]]:
    """返回 (gram, 权重)。英文/数字按词，中文按 2-3 字滑窗（单字兜底）。"""
    grams: list[tuple[str, float]] = []
    lowered = _normalize(text)
    for word in _TOKEN_RE.findall(lowered):
        grams.append((word, 1.2))
    for seg in _CJK_RE.findall(lowered):
        if len(seg) == 1:
            grams.append((seg, 0.5))
            continue
        for n, weight in ((2, 1.0), (3, 0.6)):
            for i in range(len(seg) - n + 1):
                grams.append((seg[i : i + n], weight))
    return grams


_idf: dict[str, float] = {}
_idf_default = 1.0
_stop_grams: set[str] = set()

# 疑问/泛化词只出现在提问侧，对定位原文没有区分度（"怎么处理"里的"怎么"）。
# 只在查询向量里过滤，不影响文档向量。
_QUERY_STOP = {
    "什么", "怎么", "如何", "哪些", "哪个", "多少", "多久", "几天", "几次",
    "是否", "可以", "能否", "请问", "一下", "需要", "这个", "那个", "介绍",
    "说说", "告诉", "具体", "一般", "公司", "员工",
}


def build_idf(texts: list[str], stop_idf: float | None = None) -> tuple[dict[str, float], float, set[str]]:
    """在语料上统计 IDF：idf = ln((N+1)/(df+1)) + 1（平滑，避免除零）。

    返回 (idf 表, 未见词的默认 idf, 高频停用 gram 集合)。
    为什么离线模式要做 IDF：
    哈希词袋向量若不做加权，"公司/的/是"这类高频词会让任何一句带"公司"的问题
    都拿到高分，拒答形同虚设。IDF 把常见词压下去、把专有名词顶上来，
    这也是后面 BM25 的核心思想之一（V3 会显式换成 BM25）。
    """
    stop_idf = config.STOP_IDF if stop_idf is None else stop_idf
    total = len(texts)
    df: dict[str, int] = {}
    for text in texts:
        for gram in {gram for gram, _ in _grams(text)}:
            df[gram] = df.get(gram, 0) + 1
    idf = {gram: math.log((total + 1) / (count + 1)) + 1 for gram, count in df.items()}
    stop = {gram for gram, value in idf.items() if value <= stop_idf}
    default = math.log(total + 1) + 1 if total else 1.0
    return idf, default, stop


def set_idf(mapping: dict[str, float] | None, default: float = 1.0, stop: set[str] | None = None) -> None:
    """检索/入库前注入当前语料的 IDF 统计。"""
    global _idf, _idf_default, _stop_grams
    _idf = mapping or {}
    _idf_default = default or 1.0
    _stop_grams = stop or set()


def _gram_weight(gram: str, base_weight: float, query: bool) -> float:
    if gram in _stop_grams:
        return 0.0
    if query:
        if gram in _QUERY_STOP:
            return 0.0
        # 查询里的未登录词（语料没出现过）对检索没有帮助，直接丢掉，
        # 否则它只会把查询向量模长撑大、压低所有真实命中的分数。
        return base_weight * _idf.get(gram, 0.0)
    return base_weight * _idf.get(gram, _idf_default)


def _hash_embed_one(text: str, normalize: bool = True, query: bool = False) -> list[float]:
    dim = config.OFFLINE_EMBED_DIM
    vec = [0.0] * dim
    for gram, base_weight in _grams(text):
        weight = _gram_weight(gram, base_weight, query)
        if weight == 0.0:
            continue
        digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vec[idx] += sign * weight
    if not normalize:
        return vec
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


def _hash_embed(texts: list[str], normalize: bool = True, query: bool = False) -> list[list[float]]:
    return [_hash_embed_one(t, normalize=normalize, query=query) for t in texts]


def query_units(text: str) -> set[str]:
    """问题里的"有区分度"的 gram（去掉疑问词等查询停用词）。

    生成侧也复用它来判断"某个句子是否在回答这个问题"，避免"怎么/哪些"这类词
    把无关句子算成命中。
    """
    return {
        gram
        for gram, _ in _grams(text)
        if gram not in _QUERY_STOP and gram not in _stop_grams
    }


# --------------------------------------------------------------------------
# 对外统一接口
# --------------------------------------------------------------------------
def embed(texts: list[str], batch_size: int = 32, query: bool = False) -> list[list[float]]:
    """文本 → 向量。query=True 表示"查询侧向量"。

    离线模式下查询向量归一化、文档向量不归一化：
    因为短查询 × 长文档的余弦会被文档长度稀释到接近 0（哈希向量尤其严重），
    改成"查询单位向量 · 文档原始词袋向量"，分数量级才稳定、阈值才好定。
    在线模式（BGE）仍用余弦，检索库会记录建库时使用的打分方式。
    """
    if not texts:
        return []
    if _mode == "offline":
        return _hash_embed(texts, normalize=query, query=query)

    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        try:
            resp = _client.embeddings.create(model=config.EMBED_MODEL, input=batch)
            vectors.extend([item.embedding for item in resp.data])
        except Exception as exc:
            # 不静默：记录后降级，保证演示不中断；但日志里能看出这次结果不可信
            logger.error("在线 embedding 失败（第 %d 批），本批降级为哈希向量：%s", start // batch_size + 1, exc)
            vectors.extend(_hash_embed(batch, normalize=True))
    return vectors


def chat(messages: list[dict], temperature: float = 0.2) -> str:
    """单轮对话。离线模式抛 OfflineLLMError，由 generator 降级。"""
    if _mode == "offline":
        raise OfflineLLMError("当前为离线模式，无法调用大模型")
    resp = _client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()
