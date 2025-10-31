# ============================================================
# Kalki v2.0 — doc_parser.py v2.3
# ------------------------------------------------------------
# - Automatic filetype/mimetype detection (pathlib+mimetypes)
# - Robust PDF: falls back to 1st page for title if metadata absent, async support
# - OCR text integration (for scanned/image-based PDFs)
# - DOCX, HTML, EPUB, YAML/Markdown front matter parsing
# - Async ingestion: parse_document_async for parallel pipelines
# - Supports & normalizes multiple heading styles, builds hierarchy for markdown
# - Optional keyword/LLM/embedding enrichment
# - Embedding integration for RAG
# - Enriches metadata: word/sentence/paragraph count, dedup, stats logging
# - Parser registry for easy format extension
# ============================================================

import re
from pathlib import Path
import mimetypes
from typing import Optional, Dict, Any, List, Callable, Union
from collections import Counter
from modules.logging_config import get_logger
from modules.utils import safe_execution, compute_sha256
from modules.metadata import extract_text_metadata

import asyncio

logger = get_logger("Kalki.DocParser")

# --- PDF Metadata Extraction (safe, with fallback to first page title) ---
@safe_execution(default={}, logger=logger)
def extract_pdf_metadata(filepath: str) -> Dict[str, Any]:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        logger.warning("PyPDF2 not installed, cannot extract PDF metadata.")
        return {}
    try:
        reader = PdfReader(filepath)
        meta = reader.metadata or {}
        result = {
            "pdf_title": meta.get("/Title") or "",
            "pdf_author": meta.get("/Author") or "",
            "pdf_subject": meta.get("/Subject") or "",
            "pdf_creator": meta.get("/Creator") or "",
            "pdf_producer": meta.get("/Producer") or "",
            "pdf_creation_date": meta.get("/CreationDate") or "",
            "pdf_mod_date": meta.get("/ModDate") or "",
            "pdf_num_pages": len(reader.pages)
        }
        # Fallback: If no title, use first page's first line
        if not result.get("pdf_title") and reader.pages:
            first_page = reader.pages[0].extract_text() or ""
            if first_page.strip():
                result["pdf_title"] = first_page.split("\n")[0].strip()[:100]
        return result
    except Exception as e:
        logger.warning(f"Failed to extract PDF metadata from {filepath}: {e}")
        return {}

# --- DOCX Metadata and Text Extraction ---
@safe_execution(default={}, logger=logger)
def extract_docx_metadata(filepath: str) -> Dict[str, Any]:
    try:
        import docx
    except ImportError:
        logger.warning("python-docx not installed.")
        return {}
    try:
        doc = docx.Document(filepath)
        core = doc.core_properties
        text = "\n".join([p.text for p in doc.paragraphs])
        meta = {
            "docx_title": core.title or "",
            "docx_author": core.author or "",
            "docx_subject": core.subject or "",
            "docx_keywords": core.keywords or "",
            "docx_comments": core.comments or "",
            "docx_category": core.category or "",
            "docx_created": str(core.created) if core.created else "",
            "docx_modified": str(core.modified) if core.modified else "",
            "docx_word_count": len(text.split())
        }
        return meta
    except Exception as e:
        logger.warning(f"Failed to extract DOCX metadata from {filepath}: {e}")
        return {}

# --- EPUB Metadata Extraction ---
@safe_execution(default={}, logger=logger)
def extract_epub_metadata(filepath: str) -> Dict[str, Any]:
    try:
        from ebooklib import epub
    except ImportError:
        logger.warning("ebooklib not installed.")
        return {}
    try:
        book = epub.read_epub(filepath)
        meta = {
            "epub_title": book.get_metadata('DC', 'title')[0][0] if book.get_metadata('DC', 'title') else "",
            "epub_author": book.get_metadata('DC', 'creator')[0][0] if book.get_metadata('DC', 'creator') else "",
            "epub_language": book.get_metadata('DC', 'language')[0][0] if book.get_metadata('DC', 'language') else "",
            "epub_publisher": book.get_metadata('DC', 'publisher')[0][0] if book.get_metadata('DC', 'publisher') else "",
            "epub_description": book.get_metadata('DC', 'description')[0][0] if book.get_metadata('DC', 'description') else "",
        }
        return meta
    except Exception as e:
        logger.warning(f"Failed to extract EPUB metadata from {filepath}: {e}")
        return {}

# --- HTML Metadata and Title Extraction ---
@safe_execution(default={}, logger=logger)
def extract_html_metadata(filepath: str) -> Dict[str, Any]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("bs4 (BeautifulSoup) not installed.")
        return {}
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()
        soup = BeautifulSoup(html, "html.parser")
        meta = {
            "html_title": soup.title.string.strip() if soup.title and soup.title.string else "",
        }
        for m in soup.find_all("meta"):
            if m.get("name") and m.get("content"):
                meta[m["name"].lower()] = m["content"]
        return meta
    except Exception as e:
        logger.warning(f"Failed to extract HTML metadata from {filepath}: {e}")
        return {}

# --- Markdown YAML Front Matter Extraction ---
def extract_md_yaml_front_matter(text: str) -> Dict[str, Any]:
    # Look for YAML front matter at the top of a markdown file
    if text.startswith("---"):
        yaml_lines = []
        in_yaml = False
        for line in text.splitlines():
            if line.strip() == "---":
                if in_yaml:
                    break
                in_yaml = True
                continue
            if in_yaml:
                yaml_lines.append(line)
        try:
            import yaml
            return yaml.safe_load("\n".join(yaml_lines)) or {}
        except Exception:
            return {}
    return {}

# --- Section/Heading Extraction (all styles, normalized, hierarchical for markdown) ---
def extract_headings(text: str, style: str = "all") -> List[Dict[str, Any]]:
    headings = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        # Markdown style, with hierarchy
        if (style in ("all", "markdown")) and re.match(r'^#{1,6} ', line):
            h_level = len(line) - len(line.lstrip('#'))
            headings.append({"heading": line.lstrip("# ").strip().rstrip(":"), "line_no": i+1, "level": h_level})
        # Underline style (====, ---- after heading)
        if (style in ("all", "underline")) and i+1 < len(lines):
            if re.match(r'^[=~\-]+\s*$', lines[i+1]) and line.strip():
                headings.append({"heading": line.strip().rstrip(":"), "line_no": i+1, "level": 1})
        # Numbered style
        if (style in ("all", "numbered")) and re.match(r'^\d+(\.\d+)*\s+[A-Z].+', line):
            n_prefix = re.match(r'^(\d+(\.\d+)*)', line)
            h_level = n_prefix.group(1).count('.') + 1 if n_prefix else 1
            headings.append({"heading": line.strip().rstrip(":"), "line_no": i+1, "level": h_level})
    return headings

def build_heading_hierarchy(headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    root = []
    stack = []
    for h in headings:
        node = {"heading": h["heading"], "line_no": h["line_no"], "level": h.get("level", 1), "children": []}
        while stack and node["level"] <= stack[-1]["level"]:
            stack.pop()
        if stack:
            stack[-1]["children"].append(node)
        else:
            root.append(node)
        stack.append(node)
    return root

# --- Page Marker Extraction for OCR/Text ---
def extract_page_markers(text: str) -> List[int]:
    pages = []
    for idx, line in enumerate(text.splitlines()):
        if re.match(r"^---\s*PAGE\s*\d+\s*---", line, re.IGNORECASE):
            pages.append(idx+1)
    return pages

# --- Optional: Keyword Extraction ---
def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    words = re.findall(r'\b\w{4,}\b', text.lower())
    return [w for w, _ in Counter(words).most_common(top_n)]

# --- Semantic Embeddings (integration for RAG) ---
async def embed_text(text: str, embedder: Optional[Callable[[str], Any]] = None) -> Any:
    if embedder is None:
        logger.warning("No embedder provided (skipping embedding).")
        return None
    if asyncio.iscoroutinefunction(embedder):
        return await embedder(text)
    else:
        # Sync embedder in async context
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, embedder, text)

# --- Parser Registry for Future-Proofing ---
def parse_pdf(filepath: str, text: str) -> Dict[str, Any]:
    meta = extract_pdf_metadata(filepath)
    return meta

def parse_docx(filepath: str, text: str) -> Dict[str, Any]:
    return extract_docx_metadata(filepath)

def parse_epub(filepath: str, text: str) -> Dict[str, Any]:
    return extract_epub_metadata(filepath)

def parse_html(filepath: str, text: str) -> Dict[str, Any]:
    return extract_html_metadata(filepath)

def parse_text(filepath: str, text: str) -> Dict[str, Any]:
    meta = extract_text_metadata(text)
    # Also try YAML front matter for Markdown
    meta.update(extract_md_yaml_front_matter(text))
    return meta

parser_registry: Dict[str, Callable[[str, str], Dict[str, Any]]] = {
    "pdf": parse_pdf,
    "txt": parse_text,
    "md": parse_text,
    "docx": parse_docx,
    "html": parse_html,
    "htm": parse_html,
    "epub": parse_epub,
    # Extend as needed
}

# --- Parse Document Metadata & Structure (now with ocr_text, dedup, async, LLM tags, embedding) ---
def parse_document(
    filepath: str,
    text: str,
    filetype: str = None,
    ocr_text: Optional[str] = None,
    llm_tags: Optional[List[str]] = None,
    hash_cache: Optional[dict] = None,
    compute_embedding: Optional[Callable[[str], Any]] = None
) -> Dict[str, Any]:
    """
    Synchronous parser for backward compatibility.
    """
    # Use the async version but run in event loop
    return asyncio.run(parse_document_async(filepath, text, filetype, ocr_text, llm_tags, hash_cache, compute_embedding))

async def parse_document_async(
    filepath: str,
    text: str,
    filetype: str = None,
    ocr_text: Optional[str] = None,
    llm_tags: Optional[List[str]] = None,
    hash_cache: Optional[dict] = None,
    compute_embedding: Optional[Callable[[str], Any]] = None
) -> Dict[str, Any]:
    path = Path(filepath)
    ext = (filetype or path.suffix.lower().lstrip("."))
    mime_type, _ = mimetypes.guess_type(filepath)
    file_hash = compute_sha256(filepath)
    if hash_cache is not None and file_hash in hash_cache:
        logger.info(f"Skipping {filepath}: already parsed (hash {file_hash})")
        return hash_cache[file_hash]

    used_text = ocr_text if ocr_text else text

    result = {"filepath": str(filepath), "ext": ext, "mime_type": mime_type, "file_hash": file_hash}
    parser = parser_registry.get(ext)
    if parser:
        result.update(parser(filepath, used_text))
    headings = extract_headings(used_text, style="all")
    seen = set()
    deduped = []
    for h in headings:
        k = (h["heading"], h["line_no"])
        if k not in seen:
            deduped.append(h)
            seen.add(k)
    result["headings"] = deduped
    result["heading_tree"] = build_heading_hierarchy(
        [h for h in deduped if "level" in h]
    ) if ext in ("md", "markdown") else []
    result["page_markers"] = extract_page_markers(used_text)
    result["keywords"] = extract_keywords(used_text)
    # Combine LLM tags and keywords
    if llm_tags:
        result["tags"] = sorted(set(result["keywords"] + llm_tags))
    else:
        result["tags"] = result["keywords"]
    # Extra metrics
    result["word_count"] = len(used_text.split())
    result["num_sentences"] = len(re.split(r'(?<=[.!?])\s+', used_text))
    result["num_paragraphs"] = len(used_text.split('\n\n'))
    # Embedding for RAG
    if compute_embedding:
        result["embedding"] = await embed_text(used_text, compute_embedding)
    logger.info(
        f"Parsed {filepath}: {len(deduped)} headings, {len(result['keywords'])} keywords, "
        f"{result.get('pdf_num_pages', 1)} pages, {result['word_count']} words"
    )
    if hash_cache is not None:
        hash_cache[file_hash] = result
    return result

# --- CLI Test/Demo ---
if __name__ == "__main__":
    import sys
    import asyncio
    if not sys.argv[1:]:
        print("Usage: python modules/doc_parser.py <file> [--ocr <ocr_textfile>]")
        sys.exit(1)
    path = sys.argv[1]
    ocr_text = None
    if "--ocr" in sys.argv:
        idx = sys.argv.index("--ocr")
        ocr_path = sys.argv[idx + 1]
        with open(ocr_path, "r", encoding="utf-8", errors="ignore") as f:
            ocr_text = f.read()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    meta = asyncio.run(parse_document_async(path, text, ocr_text=ocr_text))
    print("Metadata & Structure:\n", meta)

# Kalki v2.0 — doc_parser.py v2.3