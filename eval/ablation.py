"""消融实验：V1 / V2 / V3 / V4 四档一起跑，输出对比表。

## 为什么要有这个脚本

"我做了父子块、BM25、精排"这种话谁都会说。面试官真正想听的是：
**你怎么知道它有用？提升是多少？代价是什么？**

回答这三个问题的唯一办法是消融（ablation）：控制其他变量不变，一次只加一级，
用同一份评测集、同一台机器、同一个随机种子跑一遍。这个脚本把这件事变成一条命令。

## V1 档位为什么要单独建库

V1 的切分是"600 字固定块"，V2 之后是"子块 + 整节父块"（子块大小见 config.CHILD_SIZE）。
切分策略变了，索引就变了，所以 V1 必须**用老切分重新建一份库**，
否则"V1 vs V2"比的其实是"要不要展开父块"，而不是"切分策略谁更好"。
四档的对照关系因此是：

| 档位 | 切分 | 父块展开 | BM25/RRF | 精排 |
| --- | --- | --- | --- | --- |
| V1 | 600 字固定块 | 否 | 否 | 否 |
| V2 | 子块 | 是 | 否 | 否 |
| V3 | 子块 | 是 | 是 | 否 |
| V4 | 子块 | 是 | 是 | 是 |

## 用法

在 rag-kb 目录下::

    python -m app.cli ablation
    python -m app.cli ablation --no-write     # 只打印不落盘
"""
from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

from app import config, llm
from app.bm25 import BM25Index
from app.ingest import file_hash
from app.loader import SUPPORTED_SUFFIXES, load
from app.retriever import RetrievalOptions, Retriever
from app.splitter import split_pages
from app.store import VectorStore
from eval.run_eval import evaluate

OUTPUT_FILE = config.EVAL_DIR / "ablation.md"

# 每一档相对上一档改了什么的"人话说明"，直接写进表格的"改动"列
PRESETS = [
    ("V1", "基线：600 字固定块 + 纯向量召回"),
    ("V2", f"父子块：{config.CHILD_SIZE} 字子块检索、整节父块喂生成"),
    ("V3", "+ BM25 稀疏召回 + RRF 融合"),
    ("V4", "+ 规则精排（覆盖度/标题/邻近度/名次）"),
]


def build_v1_store(tmp_dir: Path) -> VectorStore:
    """按 V1 的老切分（600/80 固定块）单独建一份索引，作为严格基线。"""
    store = VectorStore(tmp_dir / "v1_index.json")
    store.scoring = "dot" if llm.is_offline() else "cosine"

    records: list[dict] = []
    for path in sorted(config.RAW_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        content_hash = file_hash(path)
        for record in split_pages(load(path), content_hash):
            record["file_hash"] = content_hash
            records.append(record)

    # 和 ingest 保持一致：先在全语料上统计 IDF，再向量化
    if llm.is_offline():
        idf, default, stop = llm.build_idf([record["text"] for record in records])
        store.idf, store.idf_default, store.stop_grams = idf, default, sorted(stop)
        llm.set_idf(idf, default, stop)

    vectors = llm.embed([record["text"] for record in records])
    for record, vector in zip(records, vectors):
        record["vector"] = [round(float(value), 6) for value in vector]
    store.items = records
    store.save()
    return store


def run_ablation(top_k: int = 5, write: bool = True) -> list[dict]:
    """跑完四档，打印对比表，可选写入 eval/ablation.md。"""
    child_store = VectorStore()
    if child_store.count() == 0:
        raise RuntimeError("知识库为空，请先运行：python -m app.cli ingest --rebuild")

    with tempfile.TemporaryDirectory() as tmp:
        v1_store = build_v1_store(Path(tmp))
        # BM25 只在子块库上建一次，V3/V4 共用，保证两档的唯一差别就是"有没有精排"
        bm25 = BM25Index.from_texts(child_store.texts())

        stores = {"V1": v1_store, "V2": child_store, "V3": child_store, "V4": child_store}
        ndcg_key = f"ndcg@{top_k}"
        results: list[dict] = []
        for name, desc in PRESETS:
            options = getattr(RetrievalOptions, name.lower())(top_k)
            retriever = Retriever(
                store=stores[name],
                options=options,
                bm25=bm25 if options.use_bm25 else None,
            )
            metrics = evaluate(top_k=top_k, preset=name, verbose=False, retriever=retriever)
            metrics["desc"] = desc
            metrics["avg_context_chars"] = round(
                sum(item["context_chars"] for item in metrics["details"]) / max(len(metrics["details"]), 1)
            )
            results.append(metrics)
            print(
                f"  {name} 完成：section_recall@{top_k}={metrics['section_recall']:.1%} "
                f"nDCG@{top_k}={metrics[ndcg_key]:.3f} "
                f"覆盖率={metrics['point_coverage']:.1%} P95={metrics['p95_retrieval_ms']}ms"
            )

    markdown = render(results, top_k)
    print("\n" + markdown)
    if write:
        OUTPUT_FILE.write_text(markdown + "\n", encoding="utf-8")
        print(f"\n对比表已写入：{OUTPUT_FILE}")
    return results


def render(results: list[dict], top_k: int) -> str:
    """生成 markdown 对比表（同时用于终端输出和落盘）。"""
    base = results[0]
    ndcg_key = f"ndcg@{top_k}"
    lines = [
        f"# 消融实验（{datetime.now().strftime('%Y-%m-%d %H:%M')}）",
        "",
        f"- 评测集：`eval/qa_set.jsonl`（{base['total']} 条，可答 {base['answerable']} / 不可答 {base['unanswerable']}）",
        f"- 模式：{base['mode']}｜Top-K：{top_k}｜拒答阈值：{base['threshold']:.2f}",
        "- 语料：`data/raw` 下 3 份制度文档",
        "",
        f"| 档位 | 改动 | 小节命中率@K | 小节命中率@1 | MRR | nDCG@{top_k} | 要点覆盖率 | 拒答正确率 | P95 检索 | 上下文均字符 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        lines.append(
            f"| **{result['preset']}** | {result['desc']} "
            f"| {result['section_recall']:.1%} "
            f"| {result['section_recall_at_1']:.1%} "
            f"| {result['mrr']:.3f} "
            f"| {result[ndcg_key]:.3f} "
            f"| {result['point_coverage']:.1%} "
            f"| {result['refusal_accuracy']:.1%} "
            f"| {result['p95_retrieval_ms']} ms "
            f"| {result['avg_context_chars']} |"
        )

    lines += [
        "",
        "## 增量对比（相对上一档）",
        "",
        f"| 档位 | 小节命中率@K 变化 | nDCG@{top_k} 变化 | 要点覆盖率 变化 | P95 检索 变化 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for previous, current in zip(results, results[1:]):
        lines.append(
            f"| {previous['preset']} → {current['preset']} "
            f"| {_delta(previous['section_recall'], current['section_recall'], True)} "
            f"| {current[ndcg_key] - previous[ndcg_key]:+.3f} "
            f"| {_delta(previous['point_coverage'], current['point_coverage'], True)} "
            f"| {_delta(previous['p95_retrieval_ms'], current['p95_retrieval_ms'], False)} |"
        )

    lines += [
        "",
        "## 分类表现（V4）",
        "",
        "| 类别 | 题数 | 小节命中率 | 要点覆盖率 |",
        "| --- | --- | --- | --- |",
    ]
    best = results[-1]
    for category, bucket in sorted(best["by_category"].items()):
        if category == "negative":
            lines.append(f"| {category} | {bucket['n']} | — | — |")
            continue
        section_rate = bucket["section_hit"] / bucket["n"] if bucket["n"] else 0.0
        point_rate = bucket["point_hit"] / bucket["point_total"] if bucket["point_total"] else 0.0
        lines.append(f"| {category} | {bucket['n']} | {section_rate:.1%} | {point_rate:.1%} |")

    lines += [
        "",
        "> 口径说明：`小节命中率@K` 要求 Top-K 里出现**正确的那一小节**（不是整个文件），",
        "> 比 V1 的 `source_recall` 严格得多；V1 的 source_recall 在本语料上恒为 100%，",
        "> 正是因为它太松、无法衡量优化，才需要换口径。",
        "> 离线模式（本地哈希向量 + 抽取式回答）的数字只反映**工程链路与检索策略**的相对差异，",
        "> 接入真实 BGE / 大模型后必须用同一脚本重跑。",
    ]
    return "\n".join(lines)


def _delta(before: float, after: float, percent: bool) -> str:
    diff = after - before
    if percent:
        return f"{diff:+.1%}"
    return f"{diff:+.1f} ms"


if __name__ == "__main__":
    run_ablation()
