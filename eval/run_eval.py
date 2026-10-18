"""评测集与评测脚本。

初版指标（都能自动算）：
- Recall@K：正确来源文档是否出现在 Top-K 里（回答"检索对不对"）
- 拒答正确率：库里没有的问题，系统是否正确拒答（回答"会不会瞎编"）
- 可回答率：库里有的问题，系统是否给出了答案而不是误拒
- 检索耗时 / 端到端耗时：离线抽取式 ≈ 端到端；真实模型另有 TTFT

用法（在 rag-kb 目录下）：
    python -m eval.run_eval
    python -m eval.calibrate      # 扫描拒答阈值，看它怎么影响指标
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from app import config
from app.generator import NO_ANSWER, answer
from app.retriever import retrieve


def load_qa_set(path: str | Path | None = None) -> list[dict]:
    qa_path = Path(path) if path else config.QA_SET_FILE
    rows = []
    for line in qa_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _hit(sources: list[str], gold_source: str | None) -> bool:
    return bool(gold_source) and gold_source in sources


def evaluate(top_k: int = 5, verbose: bool = True) -> dict:
    rows = load_qa_set()
    answerable = [row for row in rows if row.get("answerable")]
    unanswerable = [row for row in rows if not row.get("answerable")]

    hits_at_k = 0
    answered = 0
    refused_correct = 0
    retrieval_ms: list[float] = []
    total_ms: list[float] = []
    details: list[dict] = []

    for row in rows:
        started = time.perf_counter()
        result = retrieve(row["q"], top_k=top_k)
        retrieval_elapsed = (time.perf_counter() - started) * 1000
        reply = answer(row["q"], result["hits"], result["low_confidence"])
        elapsed = (time.perf_counter() - started) * 1000
        retrieval_ms.append(retrieval_elapsed)
        total_ms.append(elapsed)

        sources = [hit.get("source") for hit in result["hits"]]
        found = _hit(sources, row.get("gold_source"))
        if row.get("answerable"):
            hits_at_k += int(found)
            answered += int(not result["low_confidence"] and reply["text"] != NO_ANSWER)
        else:
            refused_correct += int(result["low_confidence"] and reply["text"] == NO_ANSWER)

        details.append(
            {
                "q": row["q"],
                "answerable": row.get("answerable"),
                "gold_source": row.get("gold_source"),
                "top1": sources[0] if sources else None,
                "recall_hit": found,
                "best_score": result["best_score"],
                "refused": result["low_confidence"],
                "mode": reply["mode"],
                "retrieval_ms": round(retrieval_elapsed, 1),
            }
        )

    metrics = {
        "total": len(rows),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "top_k": top_k,
        "recall": round(hits_at_k / len(answerable), 3) if answerable else 0.0,
        "answer_rate": round(answered / len(answerable), 3) if answerable else 0.0,
        "refusal_accuracy": round(refused_correct / len(unanswerable), 3) if unanswerable else 0.0,
        "avg_retrieval_ms": round(statistics.mean(retrieval_ms), 1),
        "avg_total_ms": round(statistics.mean(total_ms), 1),
        "threshold": config.MIN_SCORE,
        "mode": "offline" if config.OFFLINE else "online",
    }

    if verbose:
        print(f"\n评测模式：{metrics['mode']}（阈值 {metrics['threshold']:.2f}，Top-K {top_k}）")
        print("| 指标 | 数值 |")
        print("| --- | --- |")
        print(f"| Recall@{top_k}（可回答问题命中正确来源） | {metrics['recall']:.1%} |")
        print(f"| 可回答率（未被误拒） | {metrics['answer_rate']:.1%} |")
        print(f"| 拒答正确率（无答案问题被拒） | {metrics['refusal_accuracy']:.1%} |")
        print(f"| 平均检索耗时 | {metrics['avg_retrieval_ms']} ms |")
        print(f"| 平均端到端耗时 | {metrics['avg_total_ms']} ms |")
        print("\n逐题结果：")
        print("| 问题 | 期望来源 | Top1 | 命中 | 最高分 | 拒答 |")
        print("| --- | --- | --- | --- | --- | --- |")
        for item in details:
            print(
                f"| {item['q']} | {item['gold_source'] or '（无答案）'} | {item['top1'] or '-'} | "
                f"{'✅' if item['recall_hit'] else ('—' if not item['answerable'] else '❌')} | "
                f"{item['best_score']:.3f} | {'是' if item['refused'] else '否'} |"
            )
    metrics["details"] = details
    return metrics


if __name__ == "__main__":
    evaluate()
