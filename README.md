# 数据检索与 RAG 方向实践项目

## 使用

```bash
# 安装依赖
uv sync

# 配置 API 密钥
cp .env.example .env
# 编辑 .env，填入 LLM_API_KEY

# 问答
uv run python cli.py "你的问题"

# 批量评测（读取 jsonl，逐题问答，输出结果文件）
uv run python cli.py --batch qa_pairs_large.jsonl -o results.jsonl
```

## 评测结果

| 指标 | 数值 |
|------|------|
| answerable 正确率 | 96.2%（76/79） |
| unanswerable 正确率 | 85.7%（18/21） |
| **总体准确率** | **94.0%（94/100）** |

## 项目结构

```
magazine_rag/     # 核心模块
cli.py            # 命令行入口
qa_pairs.jsonl    # 原始 5 题
qa_pairs_large.jsonl  # 自建 100 题
source/           # PDF 源文件
```
