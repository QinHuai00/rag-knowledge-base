"""端到端单测：临时语料入库 → 检索 → 拒答。

不依赖真实模型、不写真实索引，跑得快，可当 CI 的冒烟测试。
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.ingest import ingest_dir
from app.retriever import retrieve
from app.store import VectorStore

EMP = """# 员工手册

## 1 试用期与转正

试用期为 3 个月，表现优秀可申请提前转正，最短 2 个月。转正需提交《转正申请表》，表单编号 HR-2024-07。

## 2 年假

入职满 1 年享受 5 天年假，需提前 3 个工作日在系统申请，表单编号 HR-2024-11。
"""

IT = """# IT 服务指南

## 1 VPN

VPN 地址为 vpn.example.com，使用域账号登录。密码每 90 天更换一次。
"""


class TestIngestAndRetrieve(unittest.TestCase):
    def _build(self, tmp: str) -> tuple[VectorStore, Path]:
        raw = Path(tmp) / "raw"
        raw.mkdir()
        (raw / "员工手册.md").write_text(EMP, encoding="utf-8")
        (raw / "IT服务指南.md").write_text(IT, encoding="utf-8")
        store_path = Path(tmp) / "index.json"
        ingest_dir(raw_dir=raw, store_path=store_path, rebuild=True)
        return VectorStore(store_path), raw

    def test_ingest_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, _ = self._build(tmp)
            self.assertGreater(store.count(), 2)
            self.assertIn("员工手册.md", store.sources())
            self.assertTrue(store.idf, "离线建库应生成 IDF 统计")

    def test_retrieve_hits_right_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, _ = self._build(tmp)
            result = retrieve("试用期多久？", store=store)
            self.assertTrue(result["hits"])
            self.assertEqual(result["hits"][0]["source"], "员工手册.md")
            self.assertFalse(result["low_confidence"])

    def test_unrelated_question_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, _ = self._build(tmp)
            result = retrieve("公司股票期权怎么分配？", store=store)
            self.assertTrue(result["low_confidence"], f"不应答，实际最高分 {result['best_score']}")

    def test_unchanged_file_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, raw = self._build(tmp)
            second = ingest_dir(raw_dir=raw, store_path=store.path)
            self.assertEqual(second["ingested"], [])
            self.assertEqual(len(second["skipped"]), 2)
            self.assertEqual(second["total_chunks"], store.count())


if __name__ == "__main__":
    unittest.main()
