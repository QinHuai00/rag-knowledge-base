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


def hit_text(hit: dict) -> str:
    """生成侧读到的正文：优先父块（整节），老索引没有父块时退回子块。

    这是父子块优化的"最后一公里"——检索用小块拿精度，这里换成大块拿完整性。
    V1 时 ``context`` 字段不存在，行为与从前完全一致。
    """
    return hit.get("context") or hit.get("text") or ""


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
        for line in hit_text(hit).splitlines()
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
    """不调模型，从命中的原文里摘句子并标引用编号。

    V2 起有两处关键改动：
    1) 摘句用 ``hit_text``（父块/整节），而不是 220 字的子块——
       子块经常只包含"条件"或只有"结论"，整节才能摘出完整答案；
    2) 按 ``parent_id`` 去重：父子块模式下同一节会命中多个子块，
       不去重会输出三遍同一句话。
    """
    if not hits:
        return NO_ANSWER
    query_units = llm.query_units(question)
    best_score = hits[0].get("score") or 0.0

    # 小节标题与问题高度相关的块优先抽，避免"材料"问题被"流程"块抢走
    candidates = []
    seen_parents: set[str] = set()
    for index, hit in enumerate(hits[:MAX_EXTRACT_HITS], start=1):
        if best_score > 0 and (hit.get("score") or 0.0) < best_score * MIN_HIT_RATIO:
            break
        parent_key = hit.get("parent_id") or hit["id"]
        if parent_key in seen_parents:
            continue
        seen_parents.add(parent_key)
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
        sentences = _pick_sentences(question, hit_text(hit))
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


_REF_RE = re.compile(r"\[(\d{1,2})\]")


def sanitize_citations(text: str, n_hits: int) -> tuple[str, list[int]]:
    """删掉指向不存在资料的引用编号（模型幻觉出来的 [7]）。

    大模型偶尔会编出超出候选数量的编号，前端点开是空链接。
    这里不重试、不改写正文，只把越界编号摘掉并回报——**能修的就修，
    不能修的如实记录**，比假装没发生强。
    """
    invalid: list[int] = []

    def _replace(match: re.Match) -> str:
        number = int(match.group(1))
        if 1 <= number <= n_hits:
            return match.group(0)
        invalid.append(number)
        return ""

    cleaned = _REF_RE.sub(_replace, text)
    return cleaned, sorted(set(invalid))


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

    text, invalid_refs = sanitize_citations(text, len(hits))
    if invalid_refs:
        logger.warning("模型引用了不存在的资料编号 %s，已从回答中移除", invalid_refs)
    return {
        "text": text,
        "sources": sources,
        "mode": "online-llm",
        "low_confidence": False,
        "degraded": False,
        "invalid_refs": invalid_refs,
    }
