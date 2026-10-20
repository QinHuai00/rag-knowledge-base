"""Prompt 模板：抗幻觉的第一道防线。

三层防线（初版实现前两层 + 工程层阈值）：
1) 约束层：只依据资料回答，资料里没有就固定话术；
2) 格式层：每条结论后标 [1][2]，编号对应给它的资料序号；
3) 工程层：检索最高分低于阈值直接拒答，不调模型（在 retriever/generator 里）。

V2 起 ``build_messages`` 优先用 ``hit["context"]``（父块/整节）而不是子块：
检索小块是为了精度，喂给模型大块是为了完整性——这是父子块优化的落点。
每个候选都独立限长，避免"一个超长小节挤掉另外四个候选"。
"""
from __future__ import annotations

# 单个候选喂给模型的字符上限。三个候选 × 1200 字 ≈ 3.6K 汉字，
# 加上问题和回答，7B 级别模型的 32K 上下文完全够用，成本也可控。
MAX_BLOCK_CHARS = 1200

SYSTEM = """你是企业知识库问答助手，服务对象是公司员工。必须遵守以下规则：
1. 只能依据【资料】中的内容回答，禁止使用你自己的知识补充或推测；
2. 每个结论后面用 [1] [2] 标注来源编号，编号必须来自【资料】；
3. 如果【资料】不足以回答，只回复：知识库中未找到相关内容。
4. 直接给答案，不要复述问题，不要编造资料里没有的数字、日期、流程。"""

USER_TMPL = """【资料】
{docs}

【问题】{question}"""


def format_source(hit: dict) -> str:
    """人类可读的来源描述：文件名 · 小节 / 第 N 页 / 无页码。"""
    source = hit.get("source") or "未知文件"
    section = hit.get("section")
    page = hit.get("page")
    if section:
        return f"{source} · {section}"
    if page:
        return f"{source} · 第 {page} 页"
    return source


def build_messages(question: str, hits: list[dict], max_block_chars: int = MAX_BLOCK_CHARS) -> list[dict]:
    blocks = []
    for index, hit in enumerate(hits, start=1):
        text = (hit.get("context") or hit.get("text") or "").strip()
        if len(text) > max_block_chars:
            text = text[:max_block_chars] + "…"
        blocks.append(f"[{index}]（{format_source(hit)}）\n{text}")
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_TMPL.format(docs="\n\n".join(blocks), question=question)},
    ]
