"""HTTP 服务（可选依赖 FastAPI）。

初版只提供三个接口，够前端/Postman 调通：
    GET  /health
    POST /ingest   上传文档入库
    POST /chat     问答（普通 JSON，不做流式）

流式 SSE 放在《优化过程详解》V7，因为初版先把"检索→生成"跑对更重要。
没装 FastAPI 时运行本模块会打印安装提示而不是抛栈。
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from app import config
from app.generator import answer
from app.ingest import ingest_file
from app.retriever import retrieve
from app.store import VectorStore

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from pydantic import BaseModel, Field

    class ChatIn(BaseModel):
        """请求体模型必须定义在模块级：文件开了 future annotations，
        函数内定义的局部模型在 FastAPI 解析类型时拿不到，会被当成 query 参数。"""

        question: str = Field(min_length=1, max_length=500)
        top_k: int = config.TOP_K

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


def create_app():
    if not HAS_FASTAPI:
        raise RuntimeError(
            "未安装 FastAPI。真实服务模式需要：pip install fastapi 'uvicorn[standard]' python-multipart\n"
            "不装也能用命令行：python -m app.cli ask \"...\""
        )

    app = FastAPI(title="RAG 知识库问答（初版）", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "chunks": VectorStore().count(), "provider": config.OFFLINE}

    @app.post("/ingest")
    async def ingest(files: list[UploadFile] = File(...)) -> dict:
        config.RAW_DIR.mkdir(parents=True, exist_ok=True)
        results = []
        for upload in files:
            name = Path(upload.filename or "unnamed").name
            if Path(name).suffix.lower() not in {".md", ".txt", ".docx", ".pdf"}:
                results.append({"file": name, "error": "不支持的文件类型"})
                continue
            destination = config.RAW_DIR / name
            destination.write_bytes(await upload.read())
            try:
                results.append(ingest_file(destination))
            except Exception as exc:  # 单个文件失败不影响其余文件
                logger.exception("上传入库失败：%s", name)
                results.append({"file": name, "error": str(exc)})
        return {"results": results, "chunks": VectorStore().count()}

    @app.post("/chat")
    def chat(body: ChatIn) -> dict:
        started = time.perf_counter()
        try:
            result = retrieve(body.question, top_k=body.top_k)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        reply = answer(body.question, result["hits"], result["low_confidence"])
        return {
            "question": body.question,
            "answer": reply["text"],
            "sources": reply["sources"],
            "low_confidence": result["low_confidence"],
            "best_score": result["best_score"],
            "mode": reply["mode"],
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    return app


app = create_app() if HAS_FASTAPI else None


if __name__ == "__main__":
    if not HAS_FASTAPI:
        print("未安装 FastAPI，无法启动 HTTP 服务。")
        print("安装：pip install fastapi 'uvicorn[standard]' python-multipart")
        print("或者直接用命令行：python -m app.cli ask \"试用期多久？\"")
        raise SystemExit(0)
    import uvicorn

    config.setup_logging()
    uvicorn.run("app.server:app", host="127.0.0.1", port=8000, reload=False)
