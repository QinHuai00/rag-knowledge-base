"""全局配置。

原则：所有"魔法数字"和模型名集中在这里，优化实验只改这一处。
离线模式（无 Key / 无 openai SDK）是默认路径，保证不装任何第三方包也能跑。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

# ---------- 路径：全部以项目根为基准，不写死绝对路径 ----------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "index"
INDEX_FILE = INDEX_DIR / "chunks.json"
EVAL_DIR = BASE_DIR / "eval"
QA_SET_FILE = EVAL_DIR / "qa_set.jsonl"


def _load_dotenv(path: Path = BASE_DIR / ".env") -> None:
    """优先用 python-dotenv；没装就用手写的极简解析，避免多一个硬依赖。"""
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(path, override=False)
        return
    except ImportError:
        pass
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

# ---------- 模型接入（OpenAI 兼容协议，换平台只改这里） ----------
API_KEY = os.getenv("SF_API_KEY", "").strip()
BASE_URL = os.getenv("SF_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")
CHAT_MODEL = os.getenv("CHAT_MODEL", "Qwen/Qwen2.5-7B-Instruct")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

# ---------- 切分与检索参数（初版 V1 的值，优化实验在这里改） ----------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
TOP_K = int(os.getenv("TOP_K", "5"))

# ---------- V2 父子块（small-to-big） ----------
# 检索用小块（child，语义集中、向量不被稀释），生成用大块（parent，上下文完整）。
# 为什么 child 取 110：实测本语料的小节长度是 60~330 字（见 docs/roadmap.md 的实验记录），
# 取 220 时一个子块几乎等于一整节，父子块就退化成了"什么都不做"；
# 取 110 才能真正把"条件句"和"结论句"分开检索，再用父块把结论补回来。
CHILD_SIZE = int(os.getenv("CHILD_SIZE", "110"))
CHILD_OVERLAP = int(os.getenv("CHILD_OVERLAP", "20"))
# 父块喂给生成侧的上限：父块是整节，太长的节（如"报销标准"带表格）要截断，
# 否则在线模式 Token 成本失控、离线抽取也会被噪音带偏。
PARENT_MAX_CHARS = int(os.getenv("PARENT_MAX_CHARS", "1200"))

# ---------- V3 混合召回（BM25 + 向量，RRF 融合） ----------
# 每路各取 CANDIDATE_K 个候选再融合，而不是各自只取 Top-K：
# 融合的收益来自"两路召回集合的差集"，候选太少就没什么可融的。
CANDIDATE_K = int(os.getenv("CANDIDATE_K", "20"))
# RRF 的平滑常数：取 60 是原论文的经验值，作用是压平"第 1 名与第 2 名"的差距，
# 让两路的排序分歧不至于被单路垄断。
RRF_K = int(os.getenv("RRF_K", "60"))
BM25_K1 = float(os.getenv("BM25_K1", "1.5"))   # 词频饱和速度
BM25_B = float(os.getenv("BM25_B", "0.75"))    # 文档长度归一强度

# ---------- V4 规则精排（cross-encoder 的零依赖替代） ----------
# 精排只重排候选顺序，**不参与拒答判定**：拒答仍用向量原始分，
# 否则每加一级优化都要重新校准一次阈值，实验就没法只改一个变量了。
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "20"))
# 四个信号加权求和，权重之和为 1，方便解释"这个 0.82 分怎么来的"
RERANK_W_COVERAGE = float(os.getenv("RERANK_W_COVERAGE", "0.45"))    # 问题词覆盖率
RERANK_W_SECTION = float(os.getenv("RERANK_W_SECTION", "0.20"))      # 小节标题命中
RERANK_W_PRIOR = float(os.getenv("RERANK_W_PRIOR", "0.20"))          # 融合阶段的原始名次
RERANK_W_PROXIMITY = float(os.getenv("RERANK_W_PROXIMITY", "0.15"))  # 问题词是否聚在一起

# 拒答阈值：离线哈希向量的分布与真实 BGE 完全不同，必须分开给两套阈值。
# - 离线默认 4.2：IDF 加权点积的量级。eval/calibrate.py 实测：
#   可回答题最低 4.66（迟到怎么处理），拒答题最高 3.86（带宠物上班）；
# - 在线默认 0.30：BGE 余弦相似度的经验值，接真实模型后必须用自己的评测集重标。
MIN_SCORE_OFFLINE = float(os.getenv("MIN_SCORE_OFFLINE", "4.2"))
MIN_SCORE_ONLINE = float(os.getenv("MIN_SCORE_ONLINE", "0.30"))

# 离线 embedding 维度：纯本地哈希，不需要模型。
# 维度越大哈希冲突越少；512 在短查询上噪音偏高，默认 2048。
# 注意：改这个值必须重建索引（向量维度不一致会检索错乱）。
OFFLINE_EMBED_DIM = int(os.getenv("OFFLINE_EMBED_DIM", "2048"))

# 离线模式的高频词过滤：IDF 低于该值的 gram 视为"太常见、没有区分度"，
# 不参与向量（否则"公司/员工/报销"会让任何问题都看起来相关）。
STOP_IDF = float(os.getenv("STOP_IDF", "1.6"))

FORCE_OFFLINE = os.getenv("FORCE_OFFLINE", "").lower() in ("1", "true", "yes")

try:  # 可选依赖：装了且配了 Key 才走真实模型
    import openai  # noqa: F401

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import numpy  # noqa: F401

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# 离线 = 不调任何远程接口。仅"有 SDK 且配了 Key"才算在线。
OFFLINE = FORCE_OFFLINE or not (HAS_OPENAI and bool(API_KEY))
MIN_SCORE = MIN_SCORE_OFFLINE if OFFLINE else MIN_SCORE_ONLINE

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """统一日志配置：CLI / server / eval 都调用它。"""
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt="%H:%M:%S")
