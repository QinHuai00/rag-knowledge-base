"""拒答阈值校准工具。

问题：MIN_SCORE 设多少合适？拍脑袋设 0.3 可能把该答的题也拒了。
做法：先把每题的最高相似度算出来，再扫描一串阈值，看「可回答率」和「拒答正确率」
的权衡曲线，取平衡点。

用法（在 rag-kb 目录下）：
    python -m eval.calibrate
"""
from __future__ import annotations

from app import config
from app.retriever import retrieve
from eval.run_eval import load_qa_set


def calibrate(candidates: list[float] | None = None) -> list[dict]:
    rows = load_qa_set()
    scored: list[tuple[dict, float]] = []
    for row in rows:
        result = retrieve(row["q"], min_score=0.0)
        scored.append((row, result["best_score"]))

    answerable = [(row, score) for row, score in scored if row.get("answerable")]
    unanswerable = [(row, score) for row, score in scored if not row.get("answerable")]

    # 不同打分方式（cosine / dot）分数量级差很多，阈值范围按实际最高分自适应
    if candidates is None:
        max_score = max((score for _, score in scored), default=1.0) or 1.0
        step = max_score / 25
        candidates = [round(step * i, 3) for i in range(1, 26)]

    print("阈值扫描（离线/在线模式不同，结论不能跨模式套用）")
    print("| 阈值 | 可回答率 | 拒答正确率 | 平衡分 |")
    print("| --- | --- | --- | --- |")
    table: list[dict] = []
    for threshold in candidates:
        answer_rate = sum(1 for _, score in answerable if score >= threshold) / max(len(answerable), 1)
        refusal_rate = sum(1 for _, score in unanswerable if score < threshold) / max(len(unanswerable), 1)
        balanced = round((answer_rate + refusal_rate) / 2, 3)
        table.append(
            {
                "threshold": threshold,
                "answer_rate": round(answer_rate, 3),
                "refusal_accuracy": round(refusal_rate, 3),
                "balanced": balanced,
            }
        )
        print(f"| {threshold:.2f} | {answer_rate:.1%} | {refusal_rate:.1%} | {balanced} |")

    best = max(table, key=lambda item: item["balanced"])
    print(f"\n当前配置阈值：{config.MIN_SCORE:.2f}；平衡分最高：{best['threshold']:.2f}"
          f"（可回答率 {best['answer_rate']:.1%}，拒答正确率 {best['refusal_accuracy']:.1%}）")
    print("注意：阈值应结合业务偏好定——客服场景宁可多拒答也别编，阈值取高一点；")
    print("     效率工具场景宁可多答，阈值取低一点。这只影响拒答，不影响 Recall。")
    return table


if __name__ == "__main__":
    calibrate()
