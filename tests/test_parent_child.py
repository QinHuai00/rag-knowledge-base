"""父子块单测：parent_id 完整性、父子文本包含关系、父块展开与截断。

V2 的核心不变量是"每个子块都能回溯到父块，父块内容覆盖子块"。
这里用临时语料真实入库一次，确保切分 / 落库 / 检索展开三段都守住这个不变量。
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app import config
from app.ingest import ingest_dir
from app.loader import Page
from app.retriever import retrieve
from app.splitter import split_pages_hierarchical
from app.store import VectorStore

DOC = """# 员工手册

## 1 试用期与转正

试用期为 3 个月，表现优秀可申请提前转正，最短 2 个月。

转正需提交《转正申请表》，表单编号 HR-2024-07，由直属主管评估后审批。

试用期工资为转正工资的 80%，转正后按岗位薪资标准执行。

## 2 年假

入职满 1 年享受 5 天年假，需提前 3 个工作日在系统申请。

年假可以跨年累计，最多累计 10 天，逾期未休视为放弃。
"""


def _build_store(tmp: str) -> VectorStore:
    raw = Path(tmp) / "raw"
    raw.mkdir()
    (raw / "员工手册.md").write_text(DOC, encoding="utf-8")
    index = Path(tmp) / "index.json"
    ingest_dir(raw_dir=raw, store_path=index, rebuild=True)
    return VectorStore(index)


class TestParentChildInvariants(unittest.TestCase):
    def test_every_child_has_existing_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _build_store(tmp)
            self.assertGreater(store.count(), 0)
            self.assertGreater(len(store.parents), 0)
            for item in store.items:
                parent_id = item.get("parent_id")
                self.assertTrue(parent_id, f"子块缺少 parent_id：{item['id']}")
                self.assertIn(parent_id, store.parents, f"父块缺失：{parent_id}")

    def test_parent_text_contains_child_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _build_store(tmp)
            for item in store.items:
                parent = store.parents[item["parent_id"]]
                self.assertIn(item["text"], parent["text"], f"父块未覆盖子块：{item['id']}")

    def test_hierarchical_split_direct(self):
        page = Page(
            text="## 1 制度正文\n\n" + "这是一段用于测试父子块切分的正文内容，需要足够长以便切出多个子块。" * 6,
            source="a.md",
            section="1 制度正文",
        )
        records, parents = split_pages_hierarchical([page], "hash123", child_size=80, child_overlap=10)
        self.assertGreater(len(records), 1)
        self.assertTrue(parents)
        for record in records:
            self.assertIn(record["parent_id"], parents)
            self.assertIn(record["text"], parents[record["parent_id"]]["text"])

    def test_parent_lookup_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _build_store(tmp)
            self.assertIsNone(store.get_parent(None))
            self.assertIsNone(store.get_parent("不存在的-parent-id"))


class TestParentExpansion(unittest.TestCase):
    def test_hits_get_context_from_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _build_store(tmp)
            result = retrieve("试用期多久？", store=store)
            self.assertTrue(result["hits"])
            for hit in result["hits"]:
                self.assertIn("context", hit, "V4 默认开启父块展开，命中必须带 context")
                self.assertTrue(hit["context"])
                self.assertIn(hit["text"], hit["context"])

    def test_context_never_exceeds_parent_max_chars(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _build_store(tmp)
            result = retrieve("试用期多久？", store=store)
            for hit in result["hits"]:
                self.assertLessEqual(len(hit["context"]), config.PARENT_MAX_CHARS)
                self.assertIn("context_truncated", hit)

    def test_oversized_parent_is_truncated(self):
        """超过 PARENT_MAX_CHARS 的父块要截断，避免 Prompt 被单节挤爆。"""
        long_body = "长段落测试关键词" + "补充说明" * 320  # 约 1300 字，单个段落
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            raw.mkdir()
            (raw / "长文档.md").write_text(f"# 长文档\n\n## 1 长段落\n\n{long_body}\n", encoding="utf-8")
            index = Path(tmp) / "index.json"
            ingest_dir(raw_dir=raw, store_path=index, rebuild=True)
            store = VectorStore(index)
            self.assertTrue(any(len(parent["text"]) > config.PARENT_MAX_CHARS for parent in store.parents.values()))

            result = retrieve("长段落测试关键词", store=store)
            self.assertTrue(result["hits"])
            hit = result["hits"][0]
            self.assertEqual(len(hit["context"]), config.PARENT_MAX_CHARS)
            self.assertTrue(hit["context_truncated"])

    def test_old_index_without_parents_falls_back(self):
        """老索引没有 parents 字段时，检索退化为直接用子块文本，不报错。"""
        with tempfile.TemporaryDirectory() as tmp:
            store = _build_store(tmp)
            store.parents = {}
            store.save()
            result = retrieve("试用期多久？", store=store)
            self.assertTrue(result["hits"])
            self.assertNotIn("context", result["hits"][0])


if __name__ == "__main__":
    unittest.main()
