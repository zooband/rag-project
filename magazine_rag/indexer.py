"""
混合索引构建：向量嵌入 + BM25。
持久化到磁盘，避免每次重新计算。
"""

import json
import pickle
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi

from . import config
from .chunker import Chunk


class HybridIndex:
    """
    混合检索索引。
    包含 chunks 的向量嵌入和 BM25 索引，供 Retriever 使用。
    """

    def __init__(self, chunks: list[Chunk],
                 embeddings: np.ndarray,
                 bm25: BM25Okapi,
                 doc_ids: list[str],
                 doc_texts: list[str]):
        self.chunks = chunks
        self.embeddings = embeddings          # (N, dim) 归一化向量
        self.bm25 = bm25
        self.doc_ids = doc_ids
        self.doc_texts = doc_texts

    def save(self, directory: Path | str) -> None:
        """持久化索引到磁盘"""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)

        chunks_data: list[dict[str, str | int]] = [{
            "id": c.id, "content": c.content,
            "page_start": c.page_start, "page_end": c.page_end,
            "section_title": c.section_title,
            "section_category": c.section_category,
            "doc_name": c.doc_name,
        } for c in self.chunks]

        (d / "chunks.json").write_text(
            json.dumps(chunks_data, ensure_ascii=False), encoding="utf-8")
        np.save(d / "embeddings.npy", self.embeddings)
        with open(d / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)
        (d / "doc_ids.json").write_text(json.dumps(self.doc_ids), encoding="utf-8")
        (d / "doc_texts.json").write_text(
            json.dumps(self.doc_texts, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path | str) -> "HybridIndex | None":
        """从磁盘加载索引，不存在返回 None"""
        d = Path(directory)
        req = ["chunks.json", "embeddings.npy", "bm25.pkl",
               "doc_ids.json", "doc_texts.json"]
        if not all((d / f).exists() for f in req):
            return None

        chunks = [Chunk(**c) for c in json.loads(
            (d / "chunks.json").read_text(encoding="utf-8"))]
        embeddings = np.load(d / "embeddings.npy")
        with open(d / "bm25.pkl", "rb") as f:
            bm25 = pickle.load(f)
        doc_ids = json.loads((d / "doc_ids.json").read_text(encoding="utf-8"))
        doc_texts = json.loads((d / "doc_texts.json").read_text(encoding="utf-8"))

        return cls(chunks, embeddings, bm25, doc_ids, doc_texts)


def _build_bm25(chunks: list[Chunk]) -> tuple[BM25Okapi, list[str], list[str]]:
    """构建 BM25 索引"""
    import warnings
    warnings.filterwarnings("ignore", category=SyntaxWarning, module="jieba")
    import jieba

    doc_ids: list[str] = []
    doc_texts: list[str] = []
    tokenized_corpus: list[list[str]] = []

    for c in chunks:
        doc_ids.append(c.id)
        doc_texts.append(c.content)
        tokenized_corpus.append(list(jieba.cut(c.content)))

    return BM25Okapi(tokenized_corpus), doc_ids, doc_texts


def _build_embeddings(chunks: list[Chunk]) -> np.ndarray:
    """用 sentence-transformers 计算所有 chunk 的向量"""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(config.EMBED_MODEL_NAME, device="cpu")
    embeddings = model.encode(
        [c.content for c in chunks],
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True,
    )
    return np.array(embeddings, dtype=np.float32)


def build_index(force_rebuild: bool = False) -> HybridIndex:
    """
    构建或加载混合索引。
    首次构建时扫描 source/ 下所有 PDF，切分后生成向量 + BM25 索引。
    """
    if not force_rebuild:
        cached = HybridIndex.load(config.INDEX_DIR)
        if cached is not None:
            return cached

    print("\n[build_index] cache not found, building from PDFs (~1-3 min) ...")

    print("[build_index] 1) parsing PDFs and chunking ...")
    from .chunker import chunk_all
    _, all_chunks = chunk_all()
    if not all_chunks:
        raise RuntimeError("no chunks to index")

    print(f"[build_index] 2) embedding {len(all_chunks)} chunks (first run downloads the model) ...")
    embeddings = _build_embeddings(all_chunks)

    print("[build_index] 3) building BM25 keyword index ...")
    bm25, doc_ids, doc_texts = _build_bm25(all_chunks)

    print("[build_index] 4) saving index to disk ...")
    index = HybridIndex(all_chunks, embeddings, bm25, doc_ids, doc_texts)
    index.save(config.INDEX_DIR)
    print("[build_index] done. subsequent runs load the cache.\n")
    return index
