# 优化路线（V1 → V7）

本文件是 README 的配套路线图。初版刻意只做"能跑通、能评测、能复现"的最小完整链路，
后面每一级优化都遵循 **一次只改一个变量 + 用同一份评测集对比** 的原则。

## 进度总览

| 级别 | 优化主题 | 改哪里 | 度量指标 | 状态 |
| --- | --- | --- | --- | --- |
| V1 | 基线：固定切分 + 纯向量召回 | `app/splitter.py`、`app/store.py`、`app/retriever.py` | source_recall / 拒答正确率 / 耗时 | ✅ 已完成 |
| V2 | 切分升级：父子块（small-to-big） | `app/splitter.py`、`app/store.py`、`app/retriever.py`、`app/generator.py` | 小节命中率@K、答案要点覆盖率 | ✅ 已完成 |
| V3 | 召回升级：BM25 + RRF 融合 | 新增 `app/bm25.py`，改 `app/retriever.py` | 小节命中率@K（重点看编号/专有名词题）、nDCG@K | ✅ 已完成 |
| V4 | 精排升级：规则精排（零依赖替代 cross-encoder） | 新增 `app/reranker.py`，改 `app/retriever.py` | 小节命中率@1、MRR、nDCG@K、延迟 | ✅ 已完成 |
| V5 | 查询侧：查询改写 / 多查询 | 计划新增 `app/query_rewrite.py` | 小节命中率@K、误召率 | ⬜ 待做 |
| V6 | 生成侧：引用校验、上下文压缩、忠实度检查 | 计划新增 `app/citation.py`、`app/context.py`，改 `app/generator.py` | 引用合法率、拒答正确率 | ⬜ 待做 |
| V7 | 工程化：增量入库、缓存、并发、SSE、观测、Docker | `app/ingest.py`、`app/server.py`、新增部署文件 | TTFT、QPS、缓存命中率 | ⬜ 待做 |

V1-V4 的实测数字、每一步"现象 → 原因 → 改法 → 踩坑"记录在
[`docs/optimization-log.md`](optimization-log.md)，对比表由 `python -m eval.ablation` 生成到
[`eval/ablation.md`](../eval/ablation.md)。

## 怎么验证每一级真的有效

```bash
# 同一份评测集，改前 / 改后各跑一次
python -m app.cli eval
# 四级消融一起对比，输出 Markdown 表
python -m eval.ablation
# 阈值变化时，重新校准（离线点积与在线余弦分数不可比）
python -m eval.calibrate

# 一次只改一个变量；结果记在同一张表里：
# | 版本 | 改动 | 小节命中率@1 | nDCG@5 | 拒答正确率 | 平均耗时 |
# | V1  | 基线 |   __%    |  __   |    __%     |  __ms    |
# | V2  | 父子块 | __%   |  __   |    __%     |  __ms    |
```

> 诚实约定：离线哈希向量 + 抽取式回答只能证明**工程链路正确**，不能当作真实语义检索效果。
> 接真实 BGE / 大模型后必须用同一份评测集重测，只引用那次测出来的数字。
