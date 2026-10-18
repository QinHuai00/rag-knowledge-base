"""切分器单测：保证不切断句子、块大小受控、偏移量可用。"""
from __future__ import annotations

import unittest

from app.splitter import split_pages, split_text
from app.loader import Page


class TestSplitText(unittest.TestCase):
    def test_short_text_is_single_chunk(self):
        text = "试用期为 3 个月，转正需要提交《转正申请表》，由直属主管评估后审批。"
        chunks = split_text(text, size=600, overlap=80)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, text)

    def test_respects_size_and_boundaries(self):
        paragraphs = [f"第{i}段：这是用来测试切分边界的一段文字，长度大概三十个字左右，需要拼到不超过目标大小。" for i in range(12)]
        text = "\n\n".join(paragraphs)
        chunks = split_text(text, size=120, overlap=20)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            # 允许超出单个超长段落，但拼包结果不应远超 size
            self.assertLessEqual(len(chunk.text), 160, msg=f"块过大：{len(chunk.text)}")
            self.assertTrue(chunk.text[:2].strip())

    def test_offsets_match_text(self):
        text = "第一段内容。\n\n第二段内容，稍微长一点，用来验证偏移量是否正确对应原文。\n\n第三段。"
        for chunk in split_text(text, size=40, overlap=10):
            self.assertEqual(chunk.text, text[chunk.start : chunk.end].strip())

    def test_long_sentence_is_hard_split(self):
        sentence = "这是一个非常长的句子" * 60 + "。"
        chunks = split_text(sentence, size=100, overlap=10)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), 110)

    def test_empty_text_returns_empty(self):
        self.assertEqual(split_text("   \n\n   "), [])
        self.assertEqual(split_text(""), [])


class TestSplitPages(unittest.TestCase):
    def test_ids_are_unique_and_traceable(self):
        pages = [
            Page(text="第一页内容，足够长足够长足够长足够长足够长。", source="a.md", section="一"),
            Page(text="第二页内容，足够长足够长足够长足够长足够长。", source="a.md", section="二"),
        ]
        records = split_pages(pages, "abc123")
        ids = [record["id"] for record in records]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all("abc123" in item for item in ids))
        self.assertTrue(all(record["source"] == "a.md" for record in records))


if __name__ == "__main__":
    unittest.main()
