"""回答生成：在线走大模型，离线走抽取式；两种模式共用同一套拒答逻辑。

设计要点：
- low_confidence 时根本不调用 chat()，直接返回固定话术（省钱 + 不可能幻觉）；
- 在线模式失败（超时/限流）自动降级为抽取式回答，保证接口不 500；
- 返回结构里带 mode，前端/CLI 必须显示，避免把离线结果误当成大模型能力。
"""
from __future__ import annotations

import logging
import re

from app import llm, prompt

logger = logging.getLogger(__name__)

NO_ANSWER = "知识库中未找到相关内容。"
MAX_EXTRACT_HITS = 3
MAX_SENTENCES_PER_HIT = 2
MAX_EXTRACT_LINES = 4
MIN_SENTENCE_OVERLAP = 0.25   # 句子与问题的 gram 重叠率低于它就不摘
MIN_HIT_RATIO = 0.5           # 低于最高分一半的命中块不参与抽取，避免拉进噪音

_SENTENCE_RE = re.compile(r"[^。！？；!?;\n]+[。！？；!?;]?")


def build_sources(hits: list[dict]) -> list[dict]:
    return [
        {
            "n": index,
            "source": hit.get("source"),
            "page": hit.get("page"),
            "section": hit.get("section"),
            "score": hit.get("score"),
            "label": prompt.format_source(hit),
        }
        for index, hit in enumerate(hits, start=1)
    ]


def _sentences(text: str) -> list[str]:
    sentences = [match.group().strip() for match in _SENTENCE_RE.finditer(text)]
    return [sentence for sentence in sentences if len(sentence) >= 6]


def _overlap(question_units: set[str], text: str) -> float:
    if not question_units:
        return 0.0
    return len(question_units & llm.query_units(text)) / len(question_units)


def _pick_sentences(
    question: str,
    text: str,
    limit: int = MAX_SENTENCES_PER_HIT,
    min_overlap: float = MIN_SENTENCE_OVERLAP,
) -> list[str]:
    """按"与问题的 gram 重叠率"挑句子，保持原文顺序返回。"""
    query_units = llm.query_units(question)
    if not query_units:
        return []
    scored: list[tuple[int, float, str]] = []
    for index, sentence in enumerate(_sentences(text)):
        if sentence.lstrip().startswith("#"):  # Markdown 标题不是答案句
            continue
        overlap = _overlap(query_units, sentence)
        if overlap >= min_overlap:
            scored.append((index, overlap, sentence))
    top = sorted(scored, key=lambda item: item[1], reverse=True)[:limit]
    return [sentence for _, _, sentence in sorted(top, key=lambda item: item[0])]


def _section_lines(hit: dict, limit: int = 2) -> list[str]:
    """小节标题命中问题时，用该节的前几行正文兜底（常见于"需要哪些材料"这类清单问题）。"""
    lines = [
        line.strip()
        for line in (hit.get("text") or "").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    picked = []
    for line in lines:
        if len(line) >= 6:
            picked.append(line if len(line) <= 160 else line[:160] + "…")
        if len(picked) >= limit:
            break
    return picked


def _truncate(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


def _extractive(question: str, hits: list[dict]) -> str:
    """不调模型，从命中的原文里摘句子并标引用编号。"""
    if not hits:
        return NO_ANSWER
    query_units = llm.query_units(question)
    best_score = hits[0].get("score") or 0.0

    # 小节标题与问题高度相关的块优先抽，避免"材料"问题被"流程"块抢走
    candidates = []
    for index, hit in enumerate(hits[:MAX_EXTRACT_HITS], start=1):
        if best_score > 0 and (hit.get("score") or 0.0) < best_score * MIN_HIT_RATIO:
            break
        section_overlap = _overlap(query_units, hit.get("section") or "")
        candidates.append((section_overlap, hit.get("score") or 0.0, index, hit))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

    # 只要有小节标题命中，就只从这些块里抽；否则退化为按分数抽
    strong_sections = [item for item in candidates if item[0] >= 0.3]
    if strong_sections:
        candidates = strong_sections

    lines: list[str] = []
    seen: set[str] = set()
    for _, _, index, hit in candidates:
        sentences = _pick_sentences(question, hit["text"])
        if not sentences:
            sentences = _section_lines(hit)
        for sentence in sentences:
            key = sentence[:40]
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- {sentence} [{index}]")
            if len(lines) >= MAX_EXTRACT_LINES:
                break
        if len(lines) >= MAX_EXTRACT_LINES:
            break
    if not lines:
        return NO_ANSWER
    return "根据知识库资料：\n" + "\n".join(lines)


def _fallback(question: str, hits: list[dict], reason: str) -> dict:
    return {
        "text": _extractive(question, hits),
        "sources": build_sources(hits),
        "mode": "offline-extractive",
        "degraded": True,
        "degraded_reason": reason,
    }


def answer(question: str, hits: list[dict], low_confidence: bool) -> dict:
    """返回 {"text", "sources", "mode", "low_confidence", "degraded"}。"""
    sources = build_sources(hits)
    if low_confidence or not hits:
        return {
            "text": NO_ANSWER,
            "sources": [],
            "mode": "refuse",
            "low_confidence": True,
            "degraded": False,
        }

    if llm.is_offline():
        return {
            "text": _extractive(question, hits),
            "sources": sources,
            "mode": "offline-extractive",
            "low_confidence": False,
            "degraded": False,
        }

    try:
        text = llm.chat(prompt.build_messages(question, hits), temperature=0.2)
    except Exception as exc:
        logger.warning("大模型调用失败，降级为抽取式回答：%s", exc)
        return _fallback(question, hits, str(exc))

    if not text:
        return _fallback(question, hits, "大模型返回空内容")
    return {
        "text": text,
        "sources": sources,
        "mode": "online-llm",
        "low_confidence": False,
        "degraded": False,
    }
