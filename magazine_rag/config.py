"""
系统全局配置。
集中管理路径、模型、检索参数，方便调优和复现。
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件（如果存在）
load_dotenv()

# ─── 路径 ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCES_DIR = PROJECT_ROOT / "source"                # PDF 源文件目录
QA_PATH = PROJECT_ROOT / "qa_pairs_large.jsonl"

INDEX_DIR = PROJECT_ROOT / "index_cache"
INDEX_DIR.mkdir(exist_ok=True)

# ─── 自动发现源文件 ──────────────────────────────────────
def discover_sources():
    """扫描 source/ 目录，返回所有 PDF 文件路径"""
    return sorted(SOURCES_DIR.glob("*.pdf"))

# ─── 模型 ────────────────────────────────────────────────
EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"       # 轻量中文向量模型 (~92MB)
RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"     # 重排序模型 (~2.2GB)

# ─── 检索参数 ────────────────────────────────────────────
CHUNK_SIZE = 384
CHUNK_OVERLAP = 64
TOP_K_SECTIONS = 3
TOP_K_CHUNKS = 9
TOP_K_FINAL = 5
HYBRID_ALPHA = 0.7
USE_RERANKER = False  # CPU 慢，关闭后检索约 3s/次

# ─── LLM 生成 ────────────────────────────────────────────
# 从 .env 读取（见 .env.example），兼容 OpenAI 格式的 API
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-v4-flash")
LLM_MAX_TOKENS = 1024
LLM_TEMPERATURE = 0.1
