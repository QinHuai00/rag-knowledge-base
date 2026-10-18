"""切分：长文本 → 语义块（初版 V1：固定长度 + 段落/句子边界回退）。

设计取舍：
- 按段落优先拼块：句子不会被拦腰截断，答案引用更自然；
- 段落超长时退到句子，句子仍超长才硬切；
- overlap 以"段/句"为单位保留，不按字符硬重叠（避免切出半句话）；
- 保留 start/end 字符偏移，后面做父子块、命中高亮时要用。

初版故意不做父子块和标题感知切分——这两项是《优化过程详解》V2 的内容。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app import config


@dataclass
class Chunk:
    text: str
    start: int
    end: int


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in re.finditer(r"\n\s*\n", text):
        start, end = _trim(text, cursor, match.start())
        if end > start:
            spans.append((start, end))
        cursor = match.end()
    start, end = _trim(text, cursor, len(text))
    if end > start:
        spans.append((start, end))
    return spans


def _sentence_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """在 [start, end) 内按中文句末标点切句；切不动就硬切。"""
    spans: list[tuple[int, int]] = []
    cursor = start
    for match in re.finditer(r"[^。！？；!?;\n]+[。！？；!?;]?", text[start:end]):
        seg_start, seg_end = start + match.start(), start + match.end()
        seg_start, seg_end = _trim(text, seg_start, seg_end)
        if seg_end > seg_start:
            spans.append((seg_start, seg_end))
        cursor = seg_end
    if not spans:
        spans = [(start, end)]
    return spans


def _units(text: str, size: int) -> list[tuple[int, int]]:
    """把文本拆成"不切断语义"的最小单元：段落，超长段落拆句。"""
    units: list[tuple[int, int]] = []
    for start, end in _paragraph_spans(text):
        if end - start <= int(size * 1.2):
            units.append((start, end))
            continue
        for seg_start, seg_end in _sentence_spans(text, start, end):
            if seg_end - seg_start <= size:
                units.append((seg_start, seg_end))
                continue
            for offset in range(seg_start, seg_end, size):  # 极端长句：硬切
                units.append((offset, min(offset + size, seg_end)))
    return units


def split_text(
    text: str,
    size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """按字符数为目标大小切块，尽量对齐段落/句子边界。"""
    size = size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    text = text.replace("\r\n", "\n")
    units = _units(text, size)
    if not units:
        return []

    chunks: list[Chunk] = []
    buffer: list[tuple[int, int]] = []
    buffer_len = 0

    def emit() -> None:
        nonlocal buffer, buffer_len
        if not buffer:
            return
        start, end = buffer[0][0], buffer[-1][1]
        chunk_text = text[start:end].strip()
        if len(chunk_text) >= 20:
            chunks.append(Chunk(text=chunk_text, start=start, end=end))
        # 保留尾部不超过 overlap 的单元作为下一块开头，避免上下文断裂
        kept: list[tuple[int, int]] = []
        kept_len = 0
        for unit in reversed(buffer):
            unit_len = unit[1] - unit[0]
            if kept_len + unit_len <= overlap:
                kept.insert(0, unit)
                kept_len += unit_len
            else:
                break
        buffer = kept
        buffer_len = kept_len

    for unit in units:
        unit_len = unit[1] - unit[0]
        if buffer and buffer_len + unit_len + 1 > size:
            emit()
        buffer.append(unit)
        buffer_len += unit_len + 1
    emit()
    return chunks


def split_pages(pages, file_hash: str) -> list[dict]:
    """页面列表 → 带全局唯一 id 的 chunk 记录（id 里带文件哈希/页码/序号）。"""
    records: list[dict] = []
    for page_index, page in enumerate(pages, start=1):
        for chunk_index, chunk in enumerate(split_text(page.text)):
            page_no = page.page if page.page is not None else page_index
            records.append(
                {
                    "id": f"{file_hash}-p{page_no}-c{chunk_index}",
                    "text": chunk.text,
                    "source": page.source,
                    "page": page.page,
                    "section": page.section,
                    "start": chunk.start,
                    "end": chunk.end,
                }
            )
    return records
