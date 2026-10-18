"""Streamlit 页面（可选）：上传文档 + 对话式问答。

先启动后端：python -m app.server
再启动前端：streamlit run ui/app.py

没装 streamlit 时本文件不能运行，但后端和命令行不受影响——
这一版刻意让前端成为"可选壳"，主链路不依赖它。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import requests
    import streamlit as st
except ImportError as exc:  # pragma: no cover - 只在没装依赖时触发
    raise SystemExit(
        "需要先安装前端依赖：pip install streamlit requests\n"
        f"（原始错误：{exc}）"
    )

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="企业知识库问答（初版）", layout="wide")
st.title("企业知识库智能问答 · 初版")

with st.sidebar:
    st.subheader("文档入库")
    uploads = st.file_uploader(
        "上传 PDF / Word / Markdown / TXT",
        type=["pdf", "docx", "md", "txt"],
        accept_multiple_files=True,
    )
    if st.button("开始入库") and uploads:
        payload = [("files", (upload.name, upload.getvalue())) for upload in uploads]
        with st.spinner("解析并向量化中…"):
            try:
                response = requests.post(f"{API}/ingest", files=payload, timeout=600)
                st.json(response.json())
            except requests.RequestException as exc:
                st.error(f"入库失败：{exc}（后端起了吗？python -m app.server）")

    st.divider()
    top_k = st.slider("Top-K", 3, 10, 5)
    if st.button("查看知识库状态"):
        try:
            st.write(requests.get(f"{API}/health", timeout=10).json())
        except requests.RequestException as exc:
            st.error(f"无法连接后端：{exc}")

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

question = st.chat_input("问点什么，比如：试用期是多久？")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            response = requests.post(
                f"{API}/chat",
                json={"question": question, "top_k": top_k},
                timeout=180,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            st.error(f"请求失败：{exc}（后端起了吗？python -m app.server）")
            st.stop()

        st.markdown(data["answer"])
        if data.get("mode") == "offline-extractive":
            st.caption("当前是离线抽取式回答，接入真实模型后会变成大模型生成。")
        if data.get("sources"):
            with st.expander("📎 来源"):
                for source in data["sources"]:
                    st.write(f"[{source['n']}] {source['label']}（相似度 {source['score']}）")
        if data.get("low_confidence"):
            st.caption(f"⚠️ 最高相似度 {data['best_score']:.2f} 低于阈值，已拒答")
        st.caption(f"耗时 {data['elapsed_ms']} ms")

    st.session_state.history.append({"role": "assistant", "content": data["answer"]})
