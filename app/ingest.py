"""入库：文件 → 解析 → 切分（父子块）→ 向量化 → 落库。

三个工程细节：
1) 内容哈希做 id 前缀：同名不同内容不会冲突，也能判断"这个文件改没改"；
2) 重新上传同一文件先按 source 删旧块再写新块，避免知识库越滚越脏；
3) 父块（整节正文）和子块（检索小块）写在同一个索引文件里，
   保证"删文件时父子块一起走"，不会留下孤儿父块。

离线模式的额外一步：先在全语料上统计 IDF，再向量化。
否则高频词（"公司""员工"）会让任何问题都显得"相关"，拒答阈值失效。
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from app import config, llm
from app.loader import SUPPORTED_SUFFIXES, load
from app.splitter import split_pages_hierarchical
from app.store import VectorStore

logger = logging.getLogger(__name__)


def file_hash(path: str | Path) -> str:
    digest = hashlib.md5(Path(path).read_bytes()).hexdigest()
    return digest[:10]


def _prepare_store(store: VectorStore) -> VectorStore:
    """建库前确认打分方式与当前模式一致，否则向量空间不可比。"""
    expected = "dot" if llm.is_offline() else "cosine"
    if store.count() > 0 and store.scoring != expected:
        raise RuntimeError(
            f"索引打分方式（{store.scoring}）与当前模式（{expected}）不一致，"
            "请重建索引：python -m app.cli ingest --rebuild"
        )
    store.scoring = expected
    return store


def _ensure_idf(store: VectorStore, extra_texts: list[str]) -> None:
    """首次建库时统计 IDF；之后增量入库沿用旧统计（新词用默认 IDF）。"""
    if not llm.is_offline():
        llm.set_idf({}, 1.0)
        return
    if store.idf:
        llm.set_idf(store.idf, store.idf_default, set(store.stop_grams))
        return
    corpus = [item["text"] for item in store.items] + extra_texts
    idf, default, stop = llm.build_idf(corpus)
    store.idf, store.idf_default, store.stop_grams = idf, default, sorted(stop)
    llm.set_idf(idf, default, stop)
    logger.info("已统计 IDF：%d 个 gram，过滤高频 gram %d 个（语料 %d 块）", len(idf), len(stop), len(corpus))


def _embed_records(records: list[dict]) -> None:
    vectors = llm.embed([record["text"] for record in records])
    if len(vectors) != len(records):
        raise RuntimeError("向量数量与 chunk 数量不一致，入库中止")
    for record, vector in zip(records, vectors):
        record["vector"] = [round(float(x), 6) for x in vector]


def _build_records(path: Path) -> tuple[list[dict], dict[str, dict]]:
    """解析 + 父子块切分，返回 (子块记录, 父块字典)。"""
    content_hash = file_hash(path)
    pages = load(path)
    records, parents = split_pages_hierarchical(pages, content_hash)
    if not records:
        raise ValueError(f"{path.name} 切分后没有有效内容")
    for record in records:
        record["file_hash"] = content_hash
    return records, parents


def ingest_file(path: str | Path, store: VectorStore | None = None, force: bool = False) -> dict:
    """入库单个文件，返回 {"file","chunks","hash","status"}。"""
    file_path = Path(path)
    store = _prepare_store(store or VectorStore())
    content_hash = file_hash(file_path)
    source = file_path.name

    known = store.sources().get(source, {})
    if not force and known.get("file_hash") == content_hash:
        logger.info("内容未变化，跳过：%s", source)
        return {"file": source, "chunks": known.get("chunks", 0), "hash": content_hash, "status": "unchanged"}

    records, parents = _build_records(file_path)
    _ensure_idf(store, [record["text"] for record in records])
    _embed_records(records)
    removed = store.remove_source(source)
    store.add_chunks(records, parents)
    logger.info("入库完成：%s（%d 块 / %d 父块，替换旧块 %d）", source, len(records), len(parents), removed)
    return {"file": source, "chunks": len(records), "hash": content_hash, "status": "ingested"}


def ingest_dir(
    raw_dir: str | Path | None = None,
    store_path: str | Path | None = None,
    rebuild: bool = False,
    force: bool = False,
) -> dict:
    """扫描目录批量入库。返回汇总结果，供 CLI / server 展示。"""
    raw = Path(raw_dir) if raw_dir else config.RAW_DIR
    if not raw.exists():
        raise FileNotFoundError(f"原始文档目录不存在：{raw}")

    store = _prepare_store(VectorStore(store_path))
    if rebuild:
        store.reset()
        logger.info("已清空向量库，重新入库")

    result: dict = {"ingested": [], "skipped": [], "errors": [], "total_chunks": 0, "total_parents": 0}
    pending: list[tuple[Path, list[dict], dict[str, dict], str]] = []

    for path in sorted(raw.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        content_hash = file_hash(path)
        known = store.sources().get(path.name, {})
        if not force and known.get("file_hash") == content_hash:
            result["skipped"].append(path.name)
            continue
        try:
            records, parents = _build_records(path)
        except Exception as exc:  # 单个文件失败不能中断整批
            logger.error("解析失败：%s（%s）", path.name, exc)
            result["errors"].append({"file": path.name, "error": str(exc)})
            continue
        pending.append((path, records, parents, content_hash))

    # 先统计 IDF（只在新库时），再统一向量化
    _ensure_idf(store, [record["text"] for _, records, _, _ in pending for record in records])

    for path, records, parents, content_hash in pending:
        try:
            _embed_records(records)
            removed = store.remove_source(path.name)
            store.add_chunks(records, parents)
        except Exception as exc:
            logger.error("入库失败：%s（%s）", path.name, exc)
            result["errors"].append({"file": path.name, "error": str(exc)})
            continue
        logger.info("入库完成：%s（%d 块 / %d 父块，替换旧块 %d）", path.name, len(records), len(parents), removed)
        result["ingested"].append(
            {"file": path.name, "chunks": len(records), "hash": content_hash, "status": "ingested"}
        )

    result["total_chunks"] = store.count()
    result["total_parents"] = len(store.parents)
    return result
