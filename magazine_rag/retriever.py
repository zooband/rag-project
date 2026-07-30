"""
检索管线：混合检索 + 重排序 + 上下文窗口扩展。
"""

import numpy as np
from collections import defaultdict

from . import config
from .chunker import Chunk
from .indexer import HybridIndex
from .types import RetrievedChunk, EvidenceItem
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Retriever:
    """
    检索器：混合检索（向量 + BM25）→ cross-encoder 重排序 → 上下文扩展。
    """

    def __init__(self, index: HybridIndex):
        self.index = index
        self._embed_model: SentenceTransformer | None = None
        self._rerank_model: AutoModelForSequenceClassification | None = None
        self._rerank_tokenizer: AutoTokenizer | None = None

    def _get_embed_model(self):
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer(
                config.EMBED_MODEL_NAME, device="cpu"
            )
        return self._embed_model

    def retrieve(self, query: str, top_k: int | None = None,
                  doc_filter: str = "") -> list[RetrievedChunk]:
        import time
        _t0 = time.time()

        top_k = top_k or config.TOP_K_FINAL
        model = self._get_embed_model()
        import jieba

        q_vec = model.encode([query], normalize_embeddings=True,
                             show_progress_bar=False)
        q_vec = np.array(q_vec, dtype=np.float32)

        sims = (self.index.embeddings @ q_vec.T).flatten()
        query_tokens = list(jieba.cut(query))
        bm25_scores = np.array(self.index.bm25.get_scores(query_tokens))
        bm25_max = bm25_scores.max()
        if bm25_max > 0:
            bm25_scores = bm25_scores / bm25_max
        alpha = config.HYBRID_ALPHA
        hybrid = alpha * sims + (1 - alpha) * bm25_scores

        # Layer 1: 按章节聚合 -> 选 Top-3 章节
        section_scores: dict[str, list[float]] = defaultdict(list)
        section_chunks: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for idx in range(len(self.index.chunks)):
            c = self.index.chunks[idx]
            if doc_filter and c.doc_name != doc_filter:
                continue
            sec_key = f"{c.doc_name}::{c.section_title}"
            section_scores[sec_key].append(hybrid[idx])
            section_chunks[sec_key].append((idx, hybrid[idx]))

        sec_rank = [(sec, max(scores))
                    for sec, scores in section_scores.items()]
        sec_rank.sort(key=lambda x: x[1], reverse=True)
        top_sections = {s for s, _ in sec_rank[:config.TOP_K_SECTIONS]}

        # Layer 2: 只在候选章节内搜索
        candidates: list[tuple[int, float]] = []
        for sec_key, chunk_list in section_chunks.items():
            if sec_key in top_sections:
                candidates.extend(chunk_list)
        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[:top_k * 3]

        # 上下文窗口扩展
        seen: set[str] = set()
        expanded: list[RetrievedChunk] = []
        for idx, score in candidates:
            chunk = self.index.chunks[idx]
            for c in self._expand_context(chunk):
                if doc_filter and c.doc_name != doc_filter:
                    continue
                if c.id not in seen:
                    seen.add(c.id)
                    expanded.append(RetrievedChunk(
                        chunk_id=c.id,
                        doc_name=c.doc_name,
                        page_start=c.page_start,
                        page_end=c.page_end,
                        section_title=c.section_title,
                        section_category=c.section_category,
                        content=c.content,
                        score=float(score),
                        vector_score=float(sims[idx]),
                        bm25_score=float(bm25_scores[idx]),
                    ))

        expanded = self._rerank(query, expanded, top_k)
        expanded.sort(key=lambda x: x.score, reverse=True)
        _t = time.time() - _t0
        if _t > 1:
            print(f"  [timing] retrieve: {_t:.1f}s ({len(expanded)} results)")
        return expanded[:top_k]

    def _expand_context(self, chunk: Chunk, n: int = 1) -> list[Chunk]:
        """返回 chunk 及其前后各 n 个相邻 chunk（同一 section 内）。"""
        if n <= 0:
            return [chunk]

        idx = next((i for i, c in enumerate(self.index.chunks)
                    if c.id == chunk.id), -1)
        if idx < 0:
            return [chunk]

        result = [chunk]
        for offset in range(1, n + 1):
            if idx - offset >= 0 and self.index.chunks[idx - offset].section_title == chunk.section_title:
                result.insert(0, self.index.chunks[idx - offset])
        for offset in range(1, n + 1):
            if idx + offset < len(self.index.chunks) and self.index.chunks[idx + offset].section_title == chunk.section_title:
                result.append(self.index.chunks[idx + offset])
        return result

    def _get_reranker(self):
        """延迟加载 cross-encoder 模型"""
        if self._rerank_model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            self._rerank_tokenizer = AutoTokenizer.from_pretrained(config.RERANK_MODEL_NAME)
            self._rerank_model = AutoModelForSequenceClassification.from_pretrained(
                config.RERANK_MODEL_NAME,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            self._rerank_model.eval()
        return self._rerank_tokenizer, self._rerank_model

    def _rerank(self, query: str,
                candidates: list[RetrievedChunk],
                top_k: int) -> list[RetrievedChunk]:
        """cross-encoder 重排序。CPU 慢时可在 config 中关闭。"""
        if not candidates or not config.USE_RERANKER:
            return candidates

        tokenizer, model = self._get_reranker()
        pairs = [(query, c.content) for c in candidates]
        inputs = tokenizer(pairs, padding=True, truncation=True,
                           max_length=512, return_tensors="pt")

        import torch
        with torch.no_grad():
            scores = model(**inputs).logits.squeeze(-1).tolist()
            if isinstance(scores, float):
                scores = [scores]

        for c, s in zip(candidates, scores):
            c.score = float(s)
            c.vector_score = float(s)
        return candidates

    def format_context(self, results: list[RetrievedChunk]) -> tuple[str, dict[int, EvidenceItem]]:
        """
        将检索结果格式化为 LLM 可读的上下文文本。
        返回 (context_text, evidence_map)
        """
        parts: list[str] = []
        evidence_map: dict[int, EvidenceItem] = {}
        for i, r in enumerate(results, 1):
            page_str = str(r.page_start) if r.page_start == r.page_end else f"{r.page_start}-{r.page_end}"
            label = f"来源：{r.doc_name}，第{page_str}页"
            if r.section_category:
                label += f"，{r.section_category}"
            parts.append(f"[{i}] {label}，{r.section_title}\n{r.content}")
            evidence_map[i] = EvidenceItem(
                page=page_str,
                section=r.section_title,
                content=r.content[:120] + ("..." if len(r.content) > 120 else ""),
            )
        return "\n\n".join(parts), evidence_map
