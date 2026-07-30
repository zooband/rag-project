# 数据检索与 RAG 方向实践项目

面向长篇幅 PDF 的双文档 RAG 问答系统。自动扫描 `source/` 目录下的所有PDF文件，构建混合索引，支持单次问答、批量问答和评测。

## 快速开始

```bash
# 1. 安装依赖
uv sync

# 2. 配置 API 密钥（兼容 OpenAI 格式）
cp .env.example .env
# 编辑 .env，填入 LLM_API_KEY=sk-...

# 3. 查看索引概况
uv run python cli.py --inspect

# 4. 单次问答
uv run python cli.py "黑龙江省工信厅在冬奥会期间做了哪些工作？"

# 5. 批量问答
uv run python cli.py --batch qa_pairs_large.jsonl -o results.jsonl

# 6. 评测
uv run python cli.py --eval

# 7. 强制重建索引
uv run python cli.py --rebuild
```

## 项目结构

```
.
├── cli.py                       # CLI 入口
├── pyproject.toml               # 依赖管理
├── qa_pairs.jsonl               # 原始 5 题
├── qa_pairs_large.jsonl         # 自建 100 题
├── source/                      # PDF 源文件
│   ├── 杂志.pdf
│   └── 产品手册.pdf
├── magazine_rag/                # 核心模块
│   ├── types.py                 # Pydantic 数据结构
│   ├── config.py                # 全局配置
│   ├── chunker.py               # PDF → 章节 → 片段
│   ├── indexer.py               # 混合索引（向量 + BM25）
│   ├── retriever.py             # 双层检索 + 混合搜索
│   └── generator.py             # LLM 生成
└── report/                      # 答辩 PPT
    ├── presentation.tex
    └── presentation.pdf
```

## 评测结果

| 指标 | 数值 |
|------|------|
| answerable 正确率 | 96.2%（76/79） |
| unanswerable 正确率 | 85.7%（18/21） |
| **总体准确率** | **94.0%（94/100）** |

## 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `EMBED_MODEL_NAME` | BAAI/bge-small-zh-v1.5 | 向量模型（~92MB） |
| `CHUNK_SIZE` | 384 | 片段大小（tokens） |
| `HYBRID_ALPHA` | 0.7 | 向量 vs BM25 权重 |

LLM 通过 `.env` 配置：
```env
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-v4-flash
```
