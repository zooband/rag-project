#!/usr/bin/env python3
"""
《中国无线电》杂志 RAG 系统 — CLI 入口。
用法:
  uv run python cli.py                                    # 交互模式
  uv run python cli.py "你的问题"                          # 单次查询
  uv run python cli.py --eval                             # 在 qa_pairs.jsonl 上评测
  uv run python cli.py --inspect                          # 查看索引详情
  uv run python cli.py --rebuild                          # 强制重建索引
"""

import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))


def cmd_inspect():
    """查看索引统计信息"""
    from magazine_rag.chunker import chunk_all
    from magazine_rag import config

    sections, chunks = chunk_all()

    # 按文档名分组统计
    from collections import Counter
    doc_counts = Counter(c.doc_name for c in chunks)
    doc_chars = {}
    for c in chunks:
        doc_chars.setdefault(c.doc_name, 0)
        doc_chars[c.doc_name] += len(c.content)

    total_chunks = len(chunks)
    avg_len = sum(doc_chars.values()) // max(total_chunks, 1)

    print(f"\n{'='*50}")
    print(f"  文档概览")
    print(f"{'='*50}")
    for doc_name in sorted(doc_counts):
        print(f"  {doc_name}: {doc_counts[doc_name]} 片段, "
              f"{doc_chars[doc_name] // max(doc_counts[doc_name],1)} 字符/片段")
    print(f"  合计:          {total_chunks} 片段")
    print(f"  片段平均长度:  {avg_len} 字符")
    print(f"  索引缓存:      {config.INDEX_DIR}")
    print(f"  Embedding模型: {config.EMBED_MODEL_NAME}")
    print(f"\n  检索参数:")
    print(f"    Top-K 最终: {config.TOP_K_FINAL}")
    print(f"    Hybrid α: {config.HYBRID_ALPHA}")
    print(f"  LLM: {config.LLM_MODEL}")
    print(f"  源目录:        {config.SOURCES_DIR}")
    print(f"{'='*50}")
    for md in config.discover_sources():
        print(f"  - {md.name}")


def _build_retriever():
    from magazine_rag.retriever import Retriever
    from magazine_rag.indexer import build_index
    return Retriever(build_index())


def _build_generator():
    from magazine_rag.generator import LLMGenerator
    return LLMGenerator()


def cmd_query(query: str):
    """单次问答"""
    from magazine_rag.generator import answer_question

    retriever = _build_retriever()
    generator = _build_generator()

    print(f"\n问题: {query}")
    print(f"{'='*60}")
    result = answer_question(query, retriever, generator)
    print(f"\n答案: {result.answer}")
    if result.gold_chunks:
        print(f"\n引用来源:")
        for e in result.gold_chunks:
            print(f"  - 第{e.page}页, {e.section}")


def cmd_eval():
    """在 qa_pairs.jsonl 上评测"""
    from magazine_rag.generator import evaluate_on_qa

    retriever = _build_retriever()
    generator = _build_generator()

    results = evaluate_on_qa(retriever, generator)

    total = len(results)
    correct_type = sum(
        1 for r in results
        if r.answerable == (r.query_type == "answerable")
    )
    print(f"\n{'='*50}")
    print(f"  评测完成: {correct_type}/{total} 类型判断正确 "
          f"({correct_type/total*100:.1f}%)")
    print(f"{'='*50}")

    for r in results:
        status = "OK" if r.answerable == (r.query_type == "answerable") else "MISMATCH"
        print(f"  [{r.q_id}] {status} "
              f"answerable={r.answerable} (expected={r.query_type})")
        if r.answerable:
            print(f"     答案: {r.answer[:80]}...")


def cmd_interactive():
    """交互式问答循环"""
    from magazine_rag.generator import answer_question

    retriever = _build_retriever()
    generator = _build_generator()

    print(f"\n{'='*50}")
    print(f"  《中国无线电》杂志 RAG 系统")
    print(f"  输入 'quit' 退出, '!eval' 运行评测")
    print(f"{'='*50}\n")

    while True:
        try:
            query = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            break
        if query == "!eval":
            cmd_eval()
            continue

        result = answer_question(query, retriever, generator)
        print(f"\n答案: {result.answer}")
        if result.gold_chunks:
            for e in result.gold_chunks:
                print(f"  ↳ 第{e.page}页, {e.section}")
        print()


def cmd_batch():
    """从 JSONL 文件批量读取问题，顺序处理，输出结果文件。
    用法: uv run python cli.py --batch input.jsonl -o output.jsonl
    """
    import json, time
    from magazine_rag.generator import question_worker
    from magazine_rag.indexer import build_index

    input_path = None
    output_path = "qa_outputs.jsonl"
    for i, a in enumerate(sys.argv[2:]):
        if a == "-o" and i + 1 < len(sys.argv[2:]):
            output_path = sys.argv[i + 3]
        elif not a.startswith("-") and input_path is None:
            input_path = a

    if not input_path:
        print("用法: uv run python cli.py --batch <input.jsonl> -o <output.jsonl>")
        sys.exit(1)

    from magazine_rag.retriever import Retriever
    index = build_index()
    retriever = Retriever(index)

    with open(input_path, "r", encoding="utf-8") as f:
        items = [json.loads(l.strip()) for l in f if l.strip()]

    t0 = time.time()
    results = []
    for seq, item in enumerate(items):
        _, result = question_worker(retriever, item, seq)
        results.append(result)
        pct = f"{seq+1}/{len(items)}"
        print(f"  [{pct}] q_{seq+1} answerable={result.answerable}")

    elapsed = time.time() - t0
    answerable_count = sum(1 for r in results if r.answerable)
    print(f"\n完成！{len(results)} 题，可回答 {answerable_count}，耗时 {elapsed:.0f}s")

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(r.model_dump_json(ensure_ascii=False) + "\n")
    print(f"结果写入 {output_path}")
    return results


def main():
    if len(sys.argv) < 2:
        cmd_interactive()
    elif sys.argv[1] == "--eval":
        cmd_eval()
    elif sys.argv[1] in ("--batch", "-b"):
        cmd_batch()
    elif sys.argv[1] == "--inspect":
        cmd_inspect()
    elif sys.argv[1] == "--rebuild":
        from magazine_rag.indexer import build_index
        print("强制重建索引...")
        build_index(force_rebuild=True)
        print("完成！")
    else:
        cmd_query(" ".join(sys.argv[1:]))


if __name__ == "__main__":
    main()
