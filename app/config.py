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
