"""
答案生成：基于检索结果调用 LLM 生成带引用的答案。
"""

from openai import OpenAI

from . import config
from .retriever import Retriever
from .types import LLMOutput, QAPair, GoldChunk


SYSTEM_PROMPT = """你是一个专业的文档问答助手。请根据提供的证据片段回答问题。

输出 JSON 格式：
{
  "answerable": true,
  "answer": "你的答案",
  "evidence_ids": [1, 3]
}

evidence_ids 列出你在答案中实际使用的证据编号（见上方【证据片段】中的 [N] 标记）。
如果只用了部分证据，只列出用到的编号。
如果证据不足以回答问题，将 answerable 设为 false，answer 设为"文档中没有提供相关信息"，evidence_ids 设为 []。

【约束】
1. 严格基于证据回答，不要添加证据中没有的信息
2. 如果多个片段提供互补信息，综合它们给出完整答案
"""


def build_user_prompt(query: str, context: str) -> str:
    return f"""【证据片段】
{context}

【问题】
{query}"""


class LLMGenerator:
    """通用 OpenAI 兼容 API 生成器"""

    def __init__(self):
        if not config.LLM_API_KEY:
            raise ValueError(
                "未设置 LLM_API_KEY\n"
                "  1. 复制 .env.example 为 .env\n"
                "  2. 在 .env 中填入你的密钥: LLM_API_KEY=..."
            )

    def generate(self, query: str, context: str) -> LLMOutput:
        import json, time
        client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
        for attempt in range(3):
            resp = client.chat.completions.create(
                model=config.LLM_MODEL,
                max_tokens=config.LLM_MAX_TOKENS,
                temperature=config.LLM_TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(query, context)},
                ],
            )
            raw = (resp.choices[0].message.content or "").strip()
            try:
                return LLMOutput(**json.loads(raw))
            except (json.JSONDecodeError, Exception):
                if attempt < 2:
                    time.sleep(1)
                    continue
        return LLMOutput(answerable=False, answer="文档中没有提供相关信息")


def answer_question(query: str, retriever: Retriever,
                    generator: LLMGenerator,
                    source: str = "") -> QAPair:
    """
    完整 RAG 问答管线：检索 -> LLM 生成。
    source：限制检索来源，如 "杂志.pdf" 或 "产品手册.pdf"
    """
    results = retriever.retrieve(query, doc_filter=source)
    context, evidence_map = retriever.format_context(results)
    llm_output = generator.generate(query, context)

    gold_chunks = [GoldChunk(**evidence_map[i].model_dump())
                   for i in llm_output.evidence_ids if i in evidence_map]

    return QAPair(
        question=query,
        answer=llm_output.answer,
        answerable=llm_output.answerable,
        gold_chunks=gold_chunks,
    )


def evaluate_on_qa(retriever: Retriever, generator: LLMGenerator,
                   qa_path: str = "") -> list[QAPair]:
    """在 qa_pairs.jsonl 上批量评测。"""
    import json

    qa_path = qa_path or str(config.QA_PATH)

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_pairs = [json.loads(line) for line in f if line.strip()]

    results = []
    for qa in qa_pairs:
        print(f"\n[{qa['q_id']}] {qa['question'][:60]}...")
        result = answer_question(qa["question"], retriever, generator,
                                 source=qa.get("source", ""))
        result.q_id = qa["q_id"]
        result.query_type = qa["query_type"]
        results.append(result)

        correct_type = (result.answerable == (qa["query_type"] == "answerable"))
        status = "OK" if correct_type else "MISMATCH"
        print(f"  -> {status} answerable={result.answerable} (expected={qa['query_type']})")
        if result.answerable:
            print(f"  答案: {result.answer[:80]}...")

    return results


def question_worker(retriever, item: dict, seq: int) -> tuple[int, QAPair]:
    """处理单题，复用外部 Retriever，每次新建 LLMGenerator。"""
    generator = LLMGenerator()
    q = item.get("question", "")
    source = item.get("source", "")
    try:
        result = answer_question(q, retriever, generator, source=source)
    except Exception as e:
        result = QAPair(q_id=item.get("q_id", f"q_{seq + 1:02d}"),
                        question=q, answer=str(e), answerable=False)
    result.q_id = item.get("q_id", f"q_{seq + 1:02d}")
    return seq, result
