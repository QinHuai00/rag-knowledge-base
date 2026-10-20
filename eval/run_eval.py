"""评测集与评测脚本（V2 起升级版）。

## 为什么必须升级指标

V1 只报 Recall@5（"正确**文档**在不在 Top-5 里"）。这个口径太松：一个文件里
有十几个小节，只要 5 个候选里出现这个文件名就算命中，于是 V1 的 Recall@5 直接
**满分 100%**——满分意味着它没法再衡量任何优化。

所以这一版把口径收紧，并且拆成能分别归因的几个指标：

| 指标 | 口径 | 能回答什么问题 |
| --- | --- | --- |
| source_recall@K | 正确**文件**是否出现在 Top-K | 有没有找对文档（V1 的老指标，保留做对照） |
| section_recall@K | 正确**小节**是否出现在 Top-K | 检索精度，V2/V3/V4 的主战场 |
| section_recall@1 | Top-1 是不是正确小节 | 用户点开第一个结果就能看到答案的比例 |
| mrr | 正确小节名次倒数的平均 | 不只看"进没进 Top-K"，还看排得靠不靠前 |
| point_coverage | 答案**要点**被覆盖的比例 | 最终回答答全了没有（生成 + 父块完整性的综合效果） |
| refusal_accuracy | 库里没有的问题是否拒答 | 抗幻觉 |
| answer_rate | 库里有的问题是否被误拒 | 别把该答的题也拒了 |

## 用法

在 rag-kb 目录下::

    python -m app.cli eval                    # 当前档位（V4）
    python -m app.cli eval --preset V1        # 指定档位
    python -m app.cli ablation                # V1/V2/V3/V4 一起跑，输出对比表
"""
from __future__ import annotations

import json
import math
import re
import statistics
import time
from pathlib import Path

from app import config
from app.generator import NO_ANSWER, answer
from app.retriever import RetrievalOptions, Retriever

_SPACE_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------
# 评测集
# --------------------------------------------------------------------------
def load_qa_set(path: str | Path | None = None) -> list[dict]:
    qa_path = Path(path) if path else config.QA_SET_FILE
    rows = []
    for line in qa_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _as_list(value) -> list[str]:
    """gold_source / gold_section 允许写字符串或数组（跨节/跨文档的题写数组）。"""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [item for item in value if item]


def _norm(text: str) -> str:
    """去掉所有空白再比对：文档里写"3 个月"，要点写"3个月"，必须能对上。"""
    return _SPACE_RE.sub("", text or "")


def _point_coverage(points: list[str], text: str) -> float:
    """要点覆盖率：gold_points 里有多少条在回答原文中出现（微平均用）。"""
    if not points:
        return 0.0
    haystack = _norm(text)
    return sum(1 for point in points if _norm(point) in haystack) / len(points)


def _percentile(values: list[float], pct: float) -> float:
    """线性插值分位数（不引 numpy，避免离线环境多一个依赖）。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(pct * len(ordered)) - 1))
    return round(ordered[index], 1)


# --------------------------------------------------------------------------
# 主评测
# --------------------------------------------------------------------------
def evaluate(
    top_k: int = 5,
    preset: str = "V4",
    verbose: bool = True,
    retriever: Retriever | None = None,
) -> dict:
    """跑一遍评测集。``retriever`` 可由外面传进来复用（消融要跑四遍，省掉重复建索引）。"""
    rows = load_qa_set()
    options = getattr(RetrievalOptions, preset.lower())(top_k)
    if retriever is None:
        retriever = Retriever(options=options)
    else:
        retriever.options = options

    answerable = [row for row in rows if row.get("answerable")]
    unanswerable = [row for row in rows if not row.get("answerable")]

    source_hits = section_hits = top1_hits = 0
    reciprocal_ranks: list[float] = []
    point_total = point_hit = 0
    answered = refused_correct = 0
    retrieval_ms: list[float] = []
    total_ms: list[float] = []
    details: list[dict] = []
    by_category: dict[str, dict] = {}

    for row in rows:
        started = time.perf_counter()
        result = retriever.retrieve(row["q"])
        retrieval_elapsed = (time.perf_counter() - started) * 1000
        reply = answer(row["q"], result["hits"], result["low_confidence"])
        elapsed = (time.perf_counter() - started) * 1000
        retrieval_ms.append(retrieval_elapsed)
        total_ms.append(elapsed)

        gold_sources = _as_list(row.get("gold_source"))
        gold_sections = _as_list(row.get("gold_section"))
        hits = result["hits"]
        hit_sources = [hit.get("source") for hit in hits]
        hit_sections = [hit.get("section") for hit in hits]

        category = row.get("category") or ("negative" if not row.get("answerable") else "basic")
        bucket = by_category.setdefault(
            category, {"n": 0, "section_hit": 0, "point_hit": 0, "point_total": 0, "refused": 0}
        )
        bucket["n"] += 1

        source_found = any(source in hit_sources for source in gold_sources)
        # 小节级命中：Top-K 里出现任一 gold 小节
        section_found = any(section in hit_sections for section in gold_sections) if gold_sections else False
        # 名次：第一个命中 gold 小节的候选位置
        rank = next((i + 1 for i, section in enumerate(hit_sections) if section in gold_sections), None)

        coverage = 0.0
        if row.get("answerable"):
            source_hits += int(source_found)
            section_hits += int(section_found)
            top1_hits += int(rank == 1)
            reciprocal_ranks.append(1.0 / rank if rank else 0.0)
            coverage = _point_coverage(row.get("gold_points") or [], reply["text"])
            points = row.get("gold_points") or []
            point_total += len(points)
            point_hit += round(coverage * len(points))
            answered += int(not result["low_confidence"] and reply["text"] != NO_ANSWER)
            bucket["section_hit"] += int(section_found)
            bucket["point_hit"] += round(coverage * len(points))
            bucket["point_total"] += len(points)
        else:
            refused_correct += int(result["low_confidence"] and reply["text"] == NO_ANSWER)
            bucket["refused"] += int(result["low_confidence"] and reply["text"] == NO_ANSWER)

        details.append(
            {
                "q": row["q"],
                "category": category,
                "answerable": row.get("answerable"),
                "gold_section": "/".join(gold_sections) or None,
                "top1_section": hit_sections[0] if hit_sections else None,
                "rank": rank,
                "best_score": result["best_score"],
                "refused": result["low_confidence"],
                "coverage": round(coverage, 3),
                "mode": reply["mode"],
                "retrieval_ms": round(retrieval_elapsed, 1),
                "context_chars": result["trace"].get("context_chars", 0),
            }
        )

    n_answerable = len(answerable) or 1
    n_unanswerable = len(unanswerable) or 1
    metrics = {
        "preset": options.name,
        "mode": "offline" if config.OFFLINE else "online",
        "total": len(rows),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "top_k": top_k,
        "source_recall": round(source_hits / n_answerable, 3),
        "section_recall": round(section_hits / n_answerable, 3),
        "section_recall_at_1": round(top1_hits / n_answerable, 3),
        "mrr": round(statistics.mean(reciprocal_ranks), 3) if reciprocal_ranks else 0.0,
        "point_coverage": round(point_hit / point_total, 3) if point_total else 0.0,
        "answer_rate": round(answered / n_answerable, 3),
        "refusal_accuracy": round(refused_correct / n_unanswerable, 3),
        "avg_retrieval_ms": round(statistics.mean(retrieval_ms), 1),
        "p50_retrieval_ms": _percentile(retrieval_ms, 0.50),
        "p95_retrieval_ms": _percentile(retrieval_ms, 0.95),
        "avg_total_ms": round(statistics.mean(total_ms), 1),
        "p95_total_ms": _percentile(total_ms, 0.95),
        "threshold": config.MIN_SCORE,
        "by_category": by_category,
        "details": details,
    }

    if verbose:
        _print_report(metrics)
    return metrics


def _print_report(metrics: dict) -> None:
    print(f"\n评测档位：{metrics['preset']}｜模式：{metrics['mode']}（阈值 {metrics['threshold']:.2f}，Top-K {metrics['top_k']}）")
    print(f"样本：{metrics['total']} 条（可答 {metrics['answerable']} / 不可答 {metrics['unanswerable']}）")
    print("| 指标 | 数值 | 说明 |")
    print("| --- | --- | --- |")
    print(f"| source_recall@{metrics['top_k']} | {metrics['source_recall']:.1%} | 正确文件是否进 Top-K（V1 口径） |")
    print(f"| section_recall@{metrics['top_k']} | {metrics['section_recall']:.1%} | 正确小节是否进 Top-K |")
    print(f"| section_recall@1 | {metrics['section_recall_at_1']:.1%} | Top-1 就是正确小节 |")
    print(f"| MRR | {metrics['mrr']:.3f} | 正确小节的倒数名次均值 |")
    print(f"| 要点覆盖率 | {metrics['point_coverage']:.1%} | 答案覆盖了多少个标注要点 |")
    print(f"| 可回答率 | {metrics['answer_rate']:.1%} | 有答案的问题没有被误拒 |")
    print(f"| 拒答正确率 | {metrics['refusal_accuracy']:.1%} | 库外问题全部拒答 |")
    print(f"| 检索耗时 | 均值 {metrics['avg_retrieval_ms']}ms / P50 {metrics['p50_retrieval_ms']}ms / P95 {metrics['p95_retrieval_ms']}ms | 检索阶段 |")
    print(f"| 端到端耗时 | 均值 {metrics['avg_total_ms']}ms / P95 {metrics['p95_total_ms']}ms | 含回答生成 |")

    print("\n分类表现：")
    print("| 类别 | 题数 | 小节命中率 | 要点覆盖率 | 拒答正确率 |")
    print("| --- | --- | --- | --- | --- |")
    for category, bucket in sorted(metrics["by_category"].items()):
        section_rate = bucket["section_hit"] / bucket["n"] if bucket["n"] else 0.0
        point_rate = bucket["point_hit"] / bucket["point_total"] if bucket["point_total"] else 0.0
        if category == "negative":
            print(f"| {category} | {bucket['n']} | — | — | {bucket['refused'] / bucket['n']:.1%} |")
        else:
            print(f"| {category} | {bucket['n']} | {section_rate:.1%} | {point_rate:.1%} | — |")

    missed = [item for item in metrics["details"] if item["answerable"] and item["rank"] != 1]
    if missed:
        print(f"\nTop-1 未命中的 {len(missed)} 题（这些就是下一步要优化的点）：")
        print("| 问题 | 期望小节 | Top-1 小节 | 名次 |")
        print("| --- | --- | --- | --- |")
        for item in missed:
            rank = item["rank"] if item["rank"] else "未进 Top-K"
            print(f"| {item['q']} | {item['gold_section'] or '-'} | {item['top1_section'] or '-'} | {rank} |")


if __name__ == "__main__":
    evaluate()
