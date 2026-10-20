"""BM25 稀疏检索单测：分词规则、排序、空输入边界。

BM25 是 V3 新增的模块，之前只有端到端测试间接覆盖；
这里直接测 tokenize / BM25Index，保证"精确编号查询能排第一"这条核心假设可回归。
"""
from __future__ import annotations

import unittest

from app.bm25 import BM25Index, query_terms, tokenize


class TestTokenize(unittest.TestCase):
    def test_chinese_uses_two_grams(self):
        self.assertEqual(tokenize("员工手册"), ["员工", "工手", "手册"])

    def test_single_chinese_char_is_kept(self):
        # 单字标题（如 "病"）如果不保留单字，就会一个 token 都没有
        self.assertEqual(tokenize("病"), ["病"])

    def test_identifier_is_kept_whole_and_as_parts(self):
        # "HR-2024-07" 整体保留，同时拆出片段：写全称能命中，只写 "2024" 也能命中
        self.assertEqual(tokenize("HR-2024-07"), ["hr-2024-07", "hr", "2024", "07"])

    def test_domain_like_token(self):
        tokens = tokenize("vpn.example.com")
        self.assertIn("vpn.example.com", tokens)
        self.assertIn("vpn", tokens)
        self.assertIn("example", tokens)

    def test_lowercases(self):
        self.assertEqual(tokenize("VPN"), tokenize("vpn"))

    def test_keeps_duplicates_for_term_frequency(self):
        tokens = tokenize("试用期试用期")
        self.assertGreaterEqual(tokens.count("试用"), 2)

    def test_empty_text_returns_empty(self):
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize(None), [])  # type: ignore[arg-type]

    def test_query_terms_drop_stopwords_and_dedup(self):
        terms = query_terms("请问试用期是多久？")
        self.assertNotIn("请问", terms)
        self.assertNotIn("多久", terms)
        self.assertEqual(len(terms), len(set(terms)))
        self.assertIn("试用", terms)


class TestBM25Ranking(unittest.TestCase):
    """用两份"编号不同、其余用词几乎一样"的文档验证稀疏召回的精确性。"""

    DOCS = [
        "转正需提交《转正申请表》，表单编号 HR-2024-07。",
        "年假申请需提前 3 个工作日，表单编号 HR-2024-11。",
        "公司每年组织一次团建活动，具体时间另行通知。",
    ]

    def _index(self) -> BM25Index:
        return BM25Index.from_texts(self.DOCS)

    def test_exact_identifier_ranks_first(self):
        ranked = self._index().search("HR-2024-07")
        self.assertTrue(ranked)
        self.assertEqual(ranked[0][0], 0, f"精确编号查询应命中第一份文档，实际：{ranked}")
        self.assertGreater(ranked[0][1], 0.0)

    def test_identifier_score_differs_between_documents(self):
        scores = self._index().scores("HR-2024-07")
        self.assertGreater(scores[0], scores[1], "编号不同，BM25 分数必须能区分")

    def test_numeric_part_still_matches(self):
        # 只写年份片段时，两份带编号的文档都应命中，无关文档不应进结果
        ranked = self._index().search("2024")
        indexes = {index for index, _ in ranked}
        self.assertEqual(indexes, {0, 1})

    def test_unrelated_query_returns_nothing(self):
        self.assertEqual(self._index().search("股票期权"), [])

    def test_stopword_only_query_returns_nothing(self):
        self.assertEqual(self._index().search("什么"), [])

    def test_empty_query_returns_nothing(self):
        self.assertEqual(self._index().search(""), [])
        self.assertEqual(self._index().scores(""), [0.0, 0.0, 0.0])


class TestBM25EdgeCases(unittest.TestCase):
    def test_empty_corpus(self):
        index = BM25Index.from_texts([])
        self.assertEqual(index.size, 0)
        self.assertEqual(index.scores("试用期"), [])
        self.assertEqual(index.search("试用期"), [])

    def test_search_respects_top_n(self):
        index = BM25Index.from_texts(
            ["试用期为 3 个月。", "试用期工资为 80%。", "试用期可提前转正。"]
        )
        self.assertEqual(len(index.search("试用期", top_n=2)), 2)

    def test_results_are_sorted_descending(self):
        index = BM25Index.from_texts(
            ["试用期三个月，转正需申请。", "试用期工资为转正工资的 80%，试用期不买社保。"]
        )
        values = [score for _, score in index.search("试用期", top_n=2)]
        self.assertEqual(values, sorted(values, reverse=True))


if __name__ == "__main__":
    unittest.main()
