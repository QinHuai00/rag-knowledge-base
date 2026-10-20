"""参数扫描：子块大小取多少最合适？

## 为什么不能拍一个数就走

父子块里 ``CHILD_SIZE`` 是最关键的超参：

- 太小 → 子块只剩半句话，"条件"和"结论"被拆开，检索到的块答不了问题；
- 太大 → 一个子块接近一整节，父子块退化成"什么都没做"，白增加复杂度。

所以这个值必须**扫出来**，而不是抄别人的 512。本脚本在临时目录里按不同
``CHILD_SIZE`` 重建索引（不动线上索引），跑同一份评测集，输出对比表。

## 用法

在 rag-kb 目录下::

    python -m eval.sweep
    python -m eval.sweep --sizes 80,110,140,170,220,300
"""
from __future__ import annotations

import argparse
import tempfile
from dataclasses import replace
from pathlib import Path

from app import config, llm
from app.bm25 import BM25Index
from app.ingest import file_hash
from app.loader import SUPPORTED_SUFFIXES, load
from app.retriever import RetrievalOptions, Retriever
from app.splitter import split_pages_hierarchical
from app.store import VectorStore
from eval.run_eval import evaluate

DEFAULT_SIZES = (80, 110, 140, 170, 220, 300)


def build_store(path: Path, child_size: int, child_overlap: int | None = None) -> VectorStore:
    """按指定子块大小重建一份索引（写进临时路径，不覆盖 data/index）。"""
    store = VectorStore(path)
    store.scoring = "dot" if llm.is_offline() else "cosine"
    child_overlap = child_size // 5 if child_overlap is None else child_overlap

    records: list[dict] = []
    parents: dict[str, dict] = {}
    for file in sorted(config.RAW_DIR.iterdir()):
        if not file.is_file() or file.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        content_hash = file_hash(file)
        file_records, file_parents = split_pages_hierarchical(
            load(file), content_hash, child_size=child_size, child_overlap=child_overlap
        )
        for record in file_records:
            record["file_hash"] = content_hash
        records.extend(file_records)
        parents.update(file_parents)

    if llm.is_offline():
        idf, default, stop = llm.build_idf([record["text"] for record in records])
        store.idf, store.idf_default, store.stop_grams = idf, default, sorted(stop)
        llm.set_idf(idf, default, stop)

    vectors = llm.embed([record["text"] for record in records])
    for record, vector in zip(records, vectors):
        record["vector"] = [round(float(value), 6) for value in vector]
    store.items = records
    store.parents = parents
    store.save()
    return store


def sweep(sizes: tuple[int, ...] = DEFAULT_SIZES, preset: str = "V4", top_k: int = 5) -> list[dict]:
    results: list[dict] = []
    print(f"扫描子块大小：{list(sizes)}｜档位 {preset}｜Top-K {top_k}")
    with tempfile.TemporaryDirectory() as tmp:
        for size in sizes:
            store = build_store(Path(tmp) / f"idx_{size}.json", size)
            options = getattr(RetrievalOptions, preset.lower())(top_k)
            retriever = Retriever(
                store=store,
                options=options,
                bm25=BM25Index.from_texts(store.texts()) if options.use_bm25 else None,
            )
            metrics = evaluate(top_k=top_k, preset=preset, verbose=False, retriever=retriever)
            metrics["child_size"] = size
            metrics["chunks"] = store.count()
            metrics["parents"] = len(store.parents)
            results.append(metrics)
            print(
                f"  child_size={size:3d} 块={store.count():3d} "
                f"@1={metrics['section_recall_at_1']:.1%} MRR={metrics['mrr']:.3f} "
                f"要点={metrics['point_coverage']:.1%} P95={metrics['p95_retrieval_ms']}ms"
            )

    print("\n| 子块大小 | 块数 | 小节命中率@1 | MRR | 要点覆盖率 | P95 检索 |")
    print("| --- | --- | --- | --- | --- | --- |")
    for result in results:
        print(
            f"| {result['child_size']} | {result['chunks']} | {result['section_recall_at_1']:.1%} "
            f"| {result['mrr']:.3f} | {result['point_coverage']:.1%} | {result['p95_retrieval_ms']} ms |"
        )
    best = max(results, key=lambda item: (item["section_recall_at_1"], item["point_coverage"]))
    print(f"\n综合最优：child_size={best['child_size']}（当前配置 {config.CHILD_SIZE}）")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="扫描子块大小")
    parser.add_argument("--sizes", default=",".join(str(size) for size in DEFAULT_SIZES))
    parser.add_argument("--preset", default="V4")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    sweep(tuple(int(size) for size in args.sizes.split(",")), args.preset, args.top_k)
