"""规则精排单测：四路分量、截断、prior 单调性、不碰拒答字段。

核心约定：精排只改顺序，不改 ``hit["score"]``（拒答判定用的向量原始分）。
这里逐条钉死，防止以后"优化精排"时顺手把拒答阈值搞失效。
"""
from __future__ import annotations

import unittest

from app.reranker import Reranker, _coverage, _proximity


def _hit(text: str, section: str = "", score: float = 10.0, **extra) -> dict:
    hit = {
        "id": "x",
        "text": text,
        "source": "员工手册.md",
        "section": section,
        "parent_id": "p1",
        "score": score,
    }
    hit.update(extra)
    return hit


class TestComponents(unittest.TestCase):
    def test_coverage_counts_query_terms(self):
        # "试用期多久" → 试用/用期/期多（"多久"是查询停用词被去掉）→ 文本命中 2/3
        self.assertAlmostEqual(_coverage(["试用", "用期", "期多"], "试用期为 3 个月"), 2 / 3, places=6)

    def test_coverage_edge_cases(self):
        self.assertEqual(_coverage([], "任意文本"), 0.0)
        self.assertEqual(_coverage(["试用"], ""), 0.0)
        self.assertEqual(_coverage(["完全不存在"], "试用期为 3 个月"), 0.0)

    def test_proximity_prefers_clustered_terms(self):
        dense = "设备更换年限为 3 年。"
        sparse = "设备" + "甲" * 300 + "更换年限"
        self.assertGreater(_proximity(["设备", "更换", "年限"], dense), _proximity(["设备", "更换", "年限"], sparse))

    def test_proximity_single_term_is_zero(self):
        self.assertEqual(_proximity(["试用"], "试用期为 3 个月"), 0.0)
        self.assertEqual(_proximity([], "任意文本"), 0.0)

    def test_score_returns_parts(self):
        reranker = Reranker()
        total, parts = reranker.score("试用期多久", _hit("试用期为 3 个月", section="1.2 试用期与转正"), 0.5)
        self.assertGreater(total, 0.0)
        self.assertEqual(set(parts), {"coverage", "section", "prior", "proximity"})
        self.assertAlmostEqual(parts["prior"], 0.5, places=3)

    def test_weights_isolate_one_signal(self):
        """单个权重置 1、其余置 0，就能单独验证某一分量的贡献。"""
        reranker = Reranker(w_coverage=0.0, w_section=1.0, w_prior=0.0, w_proximity=0.0)
        _total, parts = reranker.score("试用期多久", _hit("无关文本", section="1.2 试用期与转正"), 0.0)
        self.assertGreater(parts["section"], 0.0)


class TestRerank(unittest.TestCase):
    def test_empty_hits(self):
        self.assertEqual(Reranker().rerank("试用期多久", [], top_k=5), [])

    def test_truncates_to_top_k(self):
        hits = [_hit(f"候选文本 {index}") for index in range(6)]
        self.assertEqual(len(Reranker().rerank("试用期", hits, top_k=3)), 3)

    def test_prior_is_monotonic(self):
        reranker = Reranker(w_coverage=0.0, w_section=0.0, w_prior=1.0, w_proximity=0.0)
        low, _ = reranker.score("试用期", _hit("同一段文本"), 0.25)
        high, _ = reranker.score("试用期", _hit("同一段文本"), 1.0)
        self.assertGreater(high, low)
        self.assertAlmostEqual(high, 1.0)

    def test_rerank_keeps_prior_order_when_only_prior_matters(self):
        reranker = Reranker(w_coverage=0.0, w_section=0.0, w_prior=1.0, w_proximity=0.0)
        hits = [_hit("甲", score=10.0), _hit("乙", score=9.0), _hit("丙", score=8.0)]
        reranked = reranker.rerank("试用期", hits, top_k=3)
        self.assertEqual([item["text"] for item in reranked], ["甲", "乙", "丙"])

    def test_does_not_rewrite_vector_score(self):
        """拒答用 hit["score"]，精排永远不能覆盖它（只新增 rerank_score）。"""
        hit = _hit("试用期为 3 个月", section="1.2 试用期与转正", score=12.345)
        reranked = Reranker().rerank("试用期多久", [hit], top_k=5)[0]
        self.assertEqual(reranked["score"], 12.345)
        self.assertIn("rerank_score", reranked)
        self.assertIn("rerank_parts", reranked)
        # 输入 dict 不被原地修改
        self.assertNotIn("rerank_score", hit)

    def test_keeps_other_hit_fields(self):
        hit = _hit("试用期为 3 个月", section="1.2 试用期与转正", score=9.9, fused_score=0.032, context="整节父块")
        reranked = Reranker().rerank("试用期多久", [hit], top_k=5)[0]
        for key in ("id", "source", "section", "parent_id", "fused_score", "context"):
            self.assertEqual(reranked[key], hit[key])


if __name__ == "__main__":
    unittest.main()
