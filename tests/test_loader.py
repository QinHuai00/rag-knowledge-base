"""解析器单测：Markdown 按标题切节、纯文本兜底、不支持格式报错。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.loader import load


class TestLoader(unittest.TestCase):
    def test_markdown_sections(self):
        content = "# 人工手册\n\n总则内容。\n\n## 1 试用期\n\n试用期三个月，够长了吧够长了吧。\n\n## 2 年假\n\n年假五天，提前申请，够长了吧。\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "手册.md"
            path.write_text(content, encoding="utf-8")
            pages = load(path)
        sections = [page.section for page in pages]
        self.assertIn("1 试用期", sections)
        self.assertIn("2 年假", sections)
        # 正文里必须包含条款内容，不能只剩标题
        joined = "\n".join(page.text for page in pages)
        self.assertIn("试用期三个月", joined)
        self.assertIn("年假五天", joined)
        self.assertTrue(all(page.source == "手册.md" for page in pages))

    def test_plain_text_without_headings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "说明.txt"
            path.write_text("这是一段没有标题的纯文本，用来验证兜底逻辑，长度要够十个字。", encoding="utf-8")
            pages = load(path)
        self.assertEqual(len(pages), 1)
        self.assertIsNone(pages[0].section)
        self.assertIn("没有标题", pages[0].text)

    def test_unsupported_suffix_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.exe"
            path.write_bytes(b"binary")
            with self.assertRaises(ValueError):
                load(path)

    def test_empty_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.md"
            path.write_text("   \n\n  ", encoding="utf-8")
            with self.assertRaises(ValueError):
                load(path)


if __name__ == "__main__":
    unittest.main()
