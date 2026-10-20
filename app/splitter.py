"""切分：长文本 → 语义块。

两级切分（V2）：
- **child（子块）**：short，用于检索。语义集中，向量不被稀释，BM25 也更容易命中；
- **parent（父块）**：长，用于生成。命中子块后按 ``parent_id`` 回溯整节内容。

为什么这么做：V1 用 600 字固定块，一个块里常混着"条件 + 流程 + 例外"三段内容。
查询"转正条件"时块的前半段在讲试用期工资，向量被无关内容稀释；而如果把块切到 200 字，
模型又会因为拿不到完整条款而答不全。父子块把"检索精度"和"上下文完整性"解耦。

设计取舍：
- 按段落优先拼块：句子不会被拦腰截断，答案引用更自然；
- 段落超长时退到句子，句子仍超长才硬切；
- overlap 以"段/句"为单位保留，不按字符硬重叠（避免切出半句话）；
- 保留 start/end 字符偏移，命中高亮时要用。
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


# --------------------------------------------------------------------------
# V2：父子块切分
# --------------------------------------------------------------------------
def split_pages_hierarchical(
    pages,
    file_hash: str,
    child_size: int | None = None,
    child_overlap: int | None = None,
    parent_max: int | None = None,
) -> tuple[list[dict], dict[str, dict]]:
    """页面 → (子块记录列表, 父块字典)。

    父块字典结构：``{parent_id: {"text", "source", "page", "section"}}``，
    和子块一起写进同一个索引文件，避免再维护一套存储。

    为什么父块也要再切一次：一节可能长达三千字（"报销标准"带表格），
    整节塞进 Prompt 会挤掉别的候选，所以超过 ``parent_max`` 的节按句边界二次切分，
    保证每个父块都是"可独立阅读的一段"。
    """
    child_size = child_size or config.CHILD_SIZE
    child_overlap = child_overlap if child_overlap is not None else config.CHILD_OVERLAP
    parent_max = parent_max or config.PARENT_MAX_CHARS

    records: list[dict] = []
    parents: dict[str, dict] = {}

    for page_index, page in enumerate(pages, start=1):
        page_no = page.page if page.page is not None else page_index
        # 父块：整节；超长节按句边界切成若干段
        parent_spans = [
            (chunk.text, chunk.start, chunk.end) for chunk in split_text(page.text, size=parent_max, overlap=0)
        ] or [(page.text, 0, len(page.text))]

        for parent_index, (parent_text, p_start, p_end) in enumerate(parent_spans):
            children = split_text(parent_text, size=child_size, overlap=child_overlap)
            if not children:
                # 只有标题、没有正文的"伪小节"（如 "## 2 设备申领与更换"）不登记父块，
                # 否则索引里会堆一堆没人引用的孤儿父块
                continue
            parent_id = f"{file_hash}-p{page_no}-s{parent_index}"
            parents[parent_id] = {
                "text": parent_text,
                "source": page.source,
                "page": page.page,
                "section": page.section,
            }
            # 子块：在父块内部再切，偏移换算回整节坐标系，方便以后做高亮
            for chunk_index, chunk in enumerate(children):
                records.append(
                    {
                        "id": f"{parent_id}-c{chunk_index}",
                        "text": chunk.text,
                        "source": page.source,
                        "page": page.page,
                        "section": page.section,
                        "start": p_start + chunk.start,
                        "end": p_start + chunk.end,
                        "parent_id": parent_id,
                    }
                )
    return records, parents
