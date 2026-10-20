"""评测指标单测：nDCG@K 与其余纯函数。

nDCG 是新加的纯评估指标，必须钉死口径（二元相关性、重复小节只算一次、
IDCG 按 min(len(gold), k) 计算），否则 README 里的数字不可复现。
"""
from __future__ import annotations

import math
import unittest

from eval.run_eval import _as_list, _ndcg_at_k, _percentile, _point_coverage


class TestNdcgAtK(unittest.TestCase):
    def test_gold_at_top_is_perfect(self):
        self.assertAlmostEqual(_ndcg_at_k(["A", "B", "C"], ["A"], 5), 1.0)

    def test_gold_at_second_position_is_discounted(self):
        expected = 1.0 / math.log2(3)
        self.assertAlmostEqual(_ndcg_at_k(["B", "A"], ["A"], 5), expected)

    def test_multiple_gold_all_ranked_is_perfect(self):
        self.assertAlmostEqual(_ndcg_at_k(["A", "B"], ["A", "B"], 5), 1.0)
        self.assertAlmostEqual(_ndcg_at_k(["B", "A"], ["A", "B"], 5), 1.0)

    def test_multiple_gold_partial_match(self):
        dcg = 1.0
        idcg = 1.0 + 1.0 / math.log2(3)
        self.assertAlmostEqual(_ndcg_at_k(["A", "C"], ["A", "B"], 2), dcg / idcg)

    def test_duplicate_hit_counts_once(self):
        # 同一小节被多个子块命中时不能重复加分，否则 nDCG 会超过 1
        self.assertAlmostEqual(_ndcg_at_k(["A", "A", "B"], ["A"], 5), 1.0)

    def test_no_gold_returns_zero(self):
        self.assertEqual(_ndcg_at_k(["A", "B"], [], 5), 0.0)
        self.assertEqual(_ndcg_at_k([], ["A"], 5), 0.0)

    def test_k_cutoff(self):
        self.assertEqual(_ndcg_at_k(["B", "B", "A"], ["A"], 2), 0.0)
        # 第三个唯一命中位于第 3 名：1/log2(4) = 0.5
        self.assertAlmostEqual(_ndcg_at_k(["B", "B", "A"], ["A"], 3), 0.5)

    def test_non_positive_k(self):
        self.assertEqual(_ndcg_at_k(["A"], ["A"], 0), 0.0)
        self.assertEqual(_ndcg_at_k(["A"], ["A"], -1), 0.0)


class TestEvalHelpers(unittest.TestCase):
    def test_as_list_accepts_str_and_list(self):
        self.assertEqual(_as_list("a.md"), ["a.md"])
        self.assertEqual(_as_list(["a", "b"]), ["a", "b"])
        self.assertEqual(_as_list(None), [])

    def test_point_coverage_ignores_whitespace(self):
        self.assertEqual(_point_coverage(["3个月", "转正"], "试用期为 3 个月，可申请转正。"), 1.0)
        self.assertEqual(_point_coverage(["不存在"], "任意回答"), 0.0)
        self.assertEqual(_point_coverage([], "任意回答"), 0.0)

    def test_percentile(self):
        values = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(_percentile(values, 0.5), 2.0)
        self.assertEqual(_percentile(values, 0.95), 4.0)
        self.assertEqual(_percentile([], 0.5), 0.0)


if __name__ == "__main__":
    unittest.main()
