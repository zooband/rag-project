"""
PDF → 结构化切分。
自动检测文档类型：带目录的用目录边界，无目录的用页面边界。
"""

import re
from pathlib import Path

from . import config
from .types import Page, Section, Chunk

# ─── 切分阈值（可根据文档特征调整） ──────────────────────
_AD_MIN_CONTENT = 30
_AD_KEYWORD_MIN = 10
_AD_KEYWORD_MAX = 30
_TITLE_SEARCH_RANGE = (10, 5, -1)
_MANUAL_PAGE_MIN_CHARS = 30
_MANUAL_TITLE_KEYWORD_MAX = 50
_MANUAL_TITLE_FALLBACK_MAX = 60
_MANUAL_TITLE_MIN_LEN = 4
_MANUAL_PRODUCT_KEYWORDS = ["设备", "装置", "系统", "接收机", "终端",
                            "产品", "服务", "软件", "天线", "组件"]
_MANUAL_HEADER_STRINGS = ("产品手册V3.0",)


def parse_pdf(path: Path | str) -> list[Page]:
    """直接用 PyMuPDF 从 PDF 提取文本，返回逐页 Page 列表。"""
    import fitz
    doc = fitz.open(str(path))
    pages = [Page(num=i + 1, text=page.get_text().strip()) for i, page in enumerate(doc)]
    doc.close()
    return pages


def parse_directory(pages: list[Page]) -> list[tuple[int, str, str]]:
    """
    从目录页（页 7 左右）解析条目，返回列表：
      [(起始页码, 文章标题, 栏目分类)]
    """
    # 找到目录页——从"目次 Contents"特征判断
    dir_page = None
    for p in pages:
        if "目次" in p.text and "Contents" in p.text:
            dir_page = p
            break

    if not dir_page:
        return []

    entries: list[tuple[int, str, str]] = []
    current_category = ""

    # 行级解析。目录格式示例：
    #   1    国家无线电监测中心北京冬奥会保障工作
    #        总结表彰暨年度考核表彰大会召开
    #   2    黑龙江省工信厅组织学习
    #        冬奥会无线电安全保障黑龙江团队事迹
    #   高考保障专题  NEMT Guarantee
    #   8    维护高考公平公正 "电波卫士"全力出击
    lines = dir_page.text.split("\n")
    page_pattern = re.compile(r"^\s*(\d{1,3})\s{2,}")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 栏目分类行：中文 + 英文，末尾无数字
        # 如 "高考保障专题  NEMT Guarantee"
        # 如 "热点聚焦  Focus on Hot Issues"
        cat_match = re.match(
            r"^([一-鿿]{2,8}[一-鿿\s]*)", line)
        page_match = page_pattern.match(line)

        if page_match:
            page = int(page_match.group(1))
            title = line[page_match.end():].strip()
            # 去掉尾部空格和英文
            title = re.sub(r"\s{2,}.*$", "", title).strip()
            if title:
                entries.append((page, title, current_category))
        elif cat_match and not page_match:
            cat = cat_match.group(1).strip()
            # 过滤掉非栏目行
            if any(kw in cat for kw in ["特别报道", "专题", "聚焦", "依法行政",
                                        "疫情防控", "队伍建设", "管理宣传",
                                        "海外观察", "监测检测", "设施建设",
                                        "干扰排查", "厂商发布", "高考保障"]):
                current_category = cat

    return entries


def _find_article_page(pages: list[Page], title: str, search_from: int = 1) -> int:
    """
    在 PDF 页面中搜索文章标题，返回标题首次出现的实际页码。
    用标题前几个字匹配（目录里的标题和页内标题基本一致）。
    """
    # 用标题前几个字做匹配（从长到短，提高容错）
    for head_len in range(*_TITLE_SEARCH_RANGE):
        if len(title) >= head_len:
            key = title[:head_len]
            for p in pages:
                if p.num < search_from:
                    continue
                if key in p.text:
                    return p.num
    return 0


def assign_sections(pages: list[Page],
                    dir_entries: list[tuple[int, str, str]]) -> list[Section]:
    """
    将目录条目映射到连续页面，生成 Section 列表。
    通过搜索文章标题在PDF中的实际位置来确定章节边界，而不是用固定偏移。
    """
    if not dir_entries:
        return [Section(title=f"第{p.num}页", pages=[p]) for p in pages]

    # 找到目录页位置，从目录页之后开始搜索文章
    dir_page_num = 0
    for p in pages:
        if "目次" in p.text and "Contents" in p.text:
            dir_page_num = p.num
            break

    article_pages: list[tuple[int, str, str]] = []
    scan_from = dir_page_num + 1 if dir_page_num else 1
    for _, title, category in dir_entries:
        pdf_page = _find_article_page(pages, title, scan_from)
        if pdf_page == 0:
            pdf_page = scan_from  # 找不到就接在前一篇后面
        article_pages.append((pdf_page, title, category))
        scan_from = pdf_page + 1

    sections: list[Section] = []
    for i, (pdf_page, title, category) in enumerate(article_pages):
        if i + 1 < len(article_pages):
            end_page = article_pages[i + 1][0] - 1
        else:
            end_page = pages[-1].num if pages else pdf_page + 5

        section_pages = [p for p in pages if pdf_page <= p.num <= end_page]
        sections.append(Section(
            title=title,
            category=category,
            start_page=pdf_page,
            end_page=end_page,
            pages=section_pages,
        ))

    # 目录之前的页面（封面、目录）
    first_article = article_pages[0][0] if article_pages else 1
    leading_pages = [p for p in pages if p.num < first_article]
    if leading_pages:
        content_pages = [p for p in leading_pages
                         if "广告" not in p.text or len(p.text) > _AD_KEYWORD_MIN]
        if content_pages:
            sections.insert(0, Section(
                title="卷首及目录",
                category="卷首",
                start_page=leading_pages[0].num,
                end_page=leading_pages[-1].num,
                pages=content_pages,
            ))

    return sections


def is_ad_page(page: Page) -> bool:
    """判断是否为纯广告页面"""
    text = page.text.strip()
    # 纯广告、封面、空白页
    if not text or text in ("广告", "封面"):
        return True
    if text.startswith("广告") and len(text) < _AD_KEYWORD_MAX:
        return True
    return False


def clean_page_text(text: str) -> str:
    """
    清理单页文本中的噪音：
    - 移除页码标记 (如 "2  CHINA RADIO")
    - 移除页眉页脚
    """
    lines = text.split("\n")
    cleaned: list[str] = []
    for line in lines:
        s = line.strip()
        # 跳过纯页眉
        if re.match(r"^\d+\s+CHINA RADIO$", s):
            continue
        if re.match(r"^CHINA RADIO$", s):
            continue
        if re.match(r"^\d{4}\.\d+$", s):
            continue
        if s.startswith("敬请访问") or s.startswith("敬请关注"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


# ─── 语义切分 ──────────────────────────────────────────


def tokenize_for_chunking(text: str) -> list[str]:
    """
    将文本按句子/子句切分为"逻辑单元"列表，
    用于后续组装成等长 chunk。
    """
    # 按中文句号、问号、感叹号、换行符分割
    units = re.split(r"(?<=[。！？\n])", text)
    units = [u.strip() for u in units if u.strip()]
    return units


def build_chunks(sections: list[Section],
                 chunk_size: int | None = None,
                 overlap: int | None = None) -> list[Chunk]:
    """
    对每个 section，将其文本按 token 数滑窗切分为 Chunk。
    chunk_size/overlap 单位为 token（近似：~1.5 中文字符/token）。
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    chunk_id = 0
    chunks: list[Chunk] = []

    for sec in sections:
        units = tokenize_for_chunking(sec.full_text)
        if not units:
            continue

        # 积累窗口
        window: list[str] = []
        window_chars = 0
        target_chars = chunk_size * 1.5  # 按中文字符估算

        def flush():
            nonlocal chunk_id, window, window_chars
            if not window:
                return
            chunk_id += 1
            content = "".join(window)
            page_nums = [p.num for p in sec.pages]
            chunks.append(Chunk(
                id=f"chunk_{chunk_id:04d}",
                content=content,
                page_start=page_nums[0] if page_nums else 0,
                page_end=page_nums[-1] if page_nums else 0,
                section_title=sec.title,
                section_category=sec.category,
            ))

            # 保留 overlap 部分
            overlap_chars = overlap * 1.5
            kept: list[str] = []
            kept_chars = 0
            for u in reversed(window):
                if kept_chars + len(u) > overlap_chars:
                    break
                kept.insert(0, u)
                kept_chars += len(u)
            window = kept
            window_chars = kept_chars

        for u in units:
            if window_chars + len(u) > target_chars and window:
                flush()
            window.append(u)
            window_chars += len(u)

        # 最后一个窗口
        if window:
            flush()

    # 去重：完全相同的 content 只保留一个
    seen: set[str] = set()
    deduped: list[Chunk] = []
    for c in chunks:
        if c.content not in seen:
            seen.add(c.content)
            deduped.append(c)

    return deduped


# ─── 主入口 ──────────────────────────────────────────────


def detect_doc_type(path: Path | str) -> str:
    """检测文档类型：带目录的返回 'magazine'，否则 'manual'"""
    import fitz
    doc = fitz.open(str(path))
    for page in doc:
        text = page.get_text()
        if "目次" in text and "Contents" in text:
            doc.close()
            return "magazine"
    doc.close()
    return "manual"


def chunk_document(path: Path | str) -> tuple[list[Section], list[Chunk]]:
    """
    自动检测文档类型并执行对应切分流程。
    返回 (sections, chunks)
    """
    path = Path(path)
    doc_name = path.name
    doc_type = detect_doc_type(path)

    if doc_type == "magazine":
        return _chunk_magazine(path, doc_name)
    else:
        return _chunk_manual(path, doc_name)


def _chunk_magazine(path: Path, doc_name: str) -> tuple[list[Section], list[Chunk]]:
    pages = parse_pdf(path)
    print(f"  [chunker] {doc_name}: {len(pages)} 页")

    dir_entries = parse_directory(pages)
    print(f"  [chunker] {doc_name}: {len(dir_entries)} 个目录条目")

    sections = assign_sections(pages, dir_entries)
    print(f"  [chunker] {doc_name}: {len(sections)} 章节")

    for sec in sections:
        sec.pages = [p for p in sec.pages if not is_ad_page(p)]
        for p in sec.pages:
            p.text = clean_page_text(p.text)

    chunks = build_chunks(sections)

    # 统一 doc_name
    for c in chunks:
        c.doc_name = doc_name

    print(f"  [chunker] {doc_name}: {len(chunks)} 片段")
    return sections, chunks


def _chunk_manual(path: Path, doc_name: str) -> tuple[list[Section], list[Chunk]]:
    pages = parse_pdf(path)
    print(f"  [chunker] {doc_name}: {len(pages)} 页")

    content_pages = [p for p in pages
                     if len(p.text.strip()) >= _MANUAL_PAGE_MIN_CHARS
                     and not re.match(r'^[A-Za-z0-9\s\-—]+$', p.text.strip())]
    print(f"  [chunker] {doc_name}: {len(content_pages)} 页有内容")

    sections: list[Section] = []
    for p in content_pages:
        title = _guess_product_title(p.text)
        sections.append(Section(title=title, category="产品手册",
                                start_page=p.num, end_page=p.num, pages=[p]))

    chunks: list[Chunk] = []
    for i, sec in enumerate(sections):
        text = sec.full_text.strip()
        if not text:
            continue
        chunks.append(Chunk(
            id=f"{doc_name.replace('.pdf','')}_chunk_{i + 1:04d}",
            content=text, page_start=sec.start_page, page_end=sec.end_page,
            section_title=sec.title, section_category="产品手册",
            doc_name=doc_name,
        ))

    print(f"  [chunker] {doc_name}: {len(chunks)} 片段")
    return sections, chunks


def chunk_all() -> tuple[list[Section], list[Chunk]]:
    """扫描 source/ 目录，处理所有逐页提取 md 文件"""
    from . import config
    all_sections: list[Section] = []
    all_chunks: list[Chunk] = []
    for md_path in config.discover_sources():
        secs, chunks = chunk_document(md_path)
        all_sections.extend(secs)
        all_chunks.extend(chunks)
    return all_sections, all_chunks


def _guess_product_title(text: str) -> str:
    """从页面文本中猜测产品名称作为 section 标题。
    优先匹配产品名模式，回退到首行有意义文字。
    """
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    # 优先找包含产品关键词的短行
    for line in lines:
        if _MANUAL_TITLE_MIN_LEN <= len(line) <= _MANUAL_TITLE_KEYWORD_MAX:
            if any(kw in line for kw in _MANUAL_PRODUCT_KEYWORDS):
                if line not in _MANUAL_HEADER_STRINGS and not re.match(r'^\d+$', line):
                    return line

    # 取第一行非页眉非纯数字的文字
    for line in lines:
        if line in _MANUAL_HEADER_STRINGS or re.match(r'^\d+$', line):
            continue
        if _MANUAL_TITLE_MIN_LEN <= len(line) < _MANUAL_TITLE_FALLBACK_MAX:
            return line
    return text[:_MANUAL_PAGE_MIN_CHARS].replace('\n', ' ').strip()
