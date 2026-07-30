"""Pydantic 模型，统一系统中所有数据结构。"""

from pydantic import BaseModel


class LLMOutput(BaseModel):
    """LLM 输出的 JSON 结构"""
    answerable: bool
    answer: str
    evidence_ids: list[int] = []


class GoldChunk(BaseModel):
    """单条引用证据"""
    page: str
    section: str = ""
    content: str = ""


class QAPair(BaseModel):
    """标准问答对"""
    q_id: str = ""
    source: str = ""
    query_type: str = ""
    question: str = ""
    answer: str = ""
    answerable: bool = True
    gold_chunks: list[GoldChunk] = []


class RetrievedChunk(BaseModel):
    """检索结果中的一个片段及其分数"""
    chunk_id: str
    doc_name: str
    page_start: int
    page_end: int
    section_title: str
    section_category: str = ""
    content: str
    score: float = 0.0
    vector_score: float = 0.0
    bm25_score: float = 0.0


class EvidenceItem(BaseModel):
    """format_context 返回的证据映射项"""
    page: str
    section: str = ""
    content: str = ""


# ─── chunker 数据结构 ────────────────────────────────────


class Page(BaseModel):
    """PDF 单页"""
    num: int
    text: str


class Section(BaseModel):
    """章节：目录中的一个条目，包含连续页面"""
    title: str = ""
    category: str = ""
    start_page: int = 0
    end_page: int = 0
    pages: list[Page] = []

    @property
    def full_text(self) -> str:
        return "\n".join(p.text for p in self.pages)


class Chunk(BaseModel):
    """检索用的最小片段，保留溯源信息"""
    id: str = ""
    content: str = ""
    page_start: int = 0
    page_end: int = 0
    section_title: str = ""
    section_category: str = ""
    doc_name: str = ""

