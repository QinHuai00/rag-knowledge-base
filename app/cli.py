"""命令行入口：不依赖前端，先把主链路跑通。

用法（在 rag-kb 目录下）：
    python -m app.cli ingest              # 把 data/raw 下的文档入库
    python -m app.cli ingest --rebuild    # 清空重建
    python -m app.cli ask "试用期多久？"   # 问答
    python -m app.cli search "试用期"      # 只看检索结果
    python -m app.cli status              # 库状态 / 当前模式
    python -m app.cli eval                # 跑评测集
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from app import config, llm


def _setup_stdout() -> None:
    """Windows 控制台默认 cp936，中文/符号会炸，这里强制 UTF-8。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:  # 非标准流（测试时）忽略
        pass


def _mode_banner() -> str:
    info = llm.provider_info()
    if info["mode"] == "offline":
        return "[OFFLINE] 本地哈希向量 + 抽取式回答（配置 .env 后自动切换真实模型）"
    return f"[ONLINE] {info['chat']} + {info['embedding']}"


def cmd_ingest(args: argparse.Namespace) -> int:
    from app.ingest import ingest_dir

    config.setup_logging()
    result = ingest_dir(rebuild=args.rebuild, force=args.force)
    print(_mode_banner())
    for item in result["ingested"]:
        print(f"  [入库] {item['file']}: {item['chunks']} 块（hash {item['hash']}）")
    for name in result["skipped"]:
        print(f"  [跳过] {name}: 内容未变化")
    for item in result["errors"]:
        print(f"  [失败] {item['file']}: {item['error']}")
    print(f"向量库共 {result['total_chunks']} 块 → {config.INDEX_FILE}")
    return 0 if not result["errors"] else 1


def cmd_search(args: argparse.Namespace) -> int:
    from app.retriever import retrieve

    result = retrieve(args.question, top_k=args.top_k)
    print(_mode_banner())
    print(f"问题：{args.question}")
    print(f"最高相似度：{result['best_score']:.3f}（拒答阈值 {result['threshold']:.2f}）")
    for index, hit in enumerate(result["hits"], start=1):
        label = hit.get("section") or (f"第 {hit['page']} 页" if hit.get("page") else "无页码")
        print(f"  [{index}] {hit['score']:.3f} | {hit['source']} · {label} | {hit['text'][:60].replace(chr(10), ' ')}…")
    if result["low_confidence"]:
        print("→ 置信度低于阈值，ask 时会直接拒答")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from app.generator import answer
    from app.retriever import retrieve

    started = time.perf_counter()
    result = retrieve(args.question, top_k=args.top_k)
    reply = answer(args.question, result["hits"], result["low_confidence"])
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

    if args.json:
        print(
            json.dumps(
                {
                    "question": args.question,
                    "answer": reply["text"],
                    "sources": reply["sources"],
                    "low_confidence": result["low_confidence"],
                    "best_score": result["best_score"],
                    "mode": reply["mode"],
                    "elapsed_ms": elapsed_ms,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    print(_mode_banner())
    print(f"问题：{args.question}")
    print(f"回答（{reply['mode']}）：")
    print(reply["text"])
    if reply.get("degraded"):
        print(f"注意：在线模式调用失败，已降级（{reply.get('degraded_reason')}）")
    if reply["sources"]:
        print("来源：")
        for source in reply["sources"]:
            print(f"  [{source['n']}] {source['label']}（相似度 {source['score']}）")
    if result["low_confidence"]:
        print(f"提示：最高相似度 {result['best_score']:.3f} < 阈值 {result['threshold']:.2f}，已拒答")
    print(f"耗时：{elapsed_ms} ms")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from app.store import VectorStore

    store = VectorStore()
    info = llm.provider_info()
    print(_mode_banner())
    print(f"向量库文件：{config.INDEX_FILE}")
    print(f"块数：{store.count()}")
    print(f"拒答阈值：{config.MIN_SCORE:.2f} | Chunk：{config.CHUNK_SIZE}/{config.CHUNK_OVERLAP} | Top-K：{config.TOP_K}")
    for source, meta in sorted(store.sources().items()):
        print(f"  - {source}: {meta['chunks']} 块（hash {meta['file_hash']}）")
    print(f"provider：{info}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from eval.run_eval import evaluate

    evaluate(top_k=args.top_k)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description="RAG 知识库问答（初版）")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="把 data/raw 下的文档入库")
    p_ingest.add_argument("--rebuild", action="store_true", help="清空向量库后重建")
    p_ingest.add_argument("--force", action="store_true", help="忽略内容哈希，强制重新入库")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="提问")
    p_ask.add_argument("question")
    p_ask.add_argument("--top-k", type=int, default=None)
    p_ask.add_argument("--json", action="store_true", help="输出 JSON（给脚本/前端用）")
    p_ask.set_defaults(func=cmd_ask)

    p_search = sub.add_parser("search", help="只检索不生成")
    p_search.add_argument("question")
    p_search.add_argument("--top-k", type=int, default=None)
    p_search.set_defaults(func=cmd_search)

    p_status = sub.add_parser("status", help="查看库状态与当前模式")
    p_status.set_defaults(func=cmd_status)

    p_eval = sub.add_parser("eval", help="跑评测集")
    p_eval.add_argument("--top-k", type=int, default=5)
    p_eval.set_defaults(func=cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    _setup_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
