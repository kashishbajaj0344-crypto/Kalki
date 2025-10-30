"""
KALKI v2.3 — Metadata Module v1.1
------------------------------------------------------------
Robust, extensible metadata extraction for PDF, DOCX, TXT, MD, CSV, and generic files.
- PDF via pdfplumber (title, author, creator, producer, creation/mod dates, pages)
- DOCX via python-docx (title, author, dates, page estimate)
- TXT/MD/CSV/generic (filename, size, dates, type)
- Heuristic text-based enrichment (title, author, date detection)
- Semantic chunk metadata enrichment (language, chunk hash, length)
- Error handling with error info in metadata
- Integration-ready for ingestion pipeline
- Version registration
"""

import re
import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import docx
except ImportError:
    docx = None

try:
    from modules.config import register_module_version
except ImportError:
    def register_module_version(module: str, version: str):
        pass

try:
    from modules.utils import compute_sha256, safe_read
except ImportError:
    def compute_sha256(path): return "error_hash"
    def safe_read(path): return ""

__version__ = "KALKI v2.3 — Metadata Module v1.1"
register_module_version("metadata.py", __version__)

# ------------------------------------------------------------
# PDF helpers
# ------------------------------------------------------------
def _parse_pdf_date(datestr: Optional[str]) -> Optional[str]:
    if not datestr:
        return None
    try:
        datestr = datestr.strip()
        if datestr.startswith('D:'):
            datestr = datestr[2:]
        padded = datestr + "0" * (14 - len(datestr))
        dt = datetime.strptime(padded[:14], "%Y%m%d%H%M%S")
        return dt.isoformat()
    except Exception:
        return None

def _pdf_metadata(path: Path) -> Dict[str, Any]:
    if not pdfplumber:
        return {"error": "pdfplumber not installed"}
    try:
        with pdfplumber.open(path) as pdf:
            meta = pdf.metadata or {}
            return {
                "title": meta.get("Title", path.stem),
                "author": meta.get("Author"),
                "creator": meta.get("Creator"),
                "producer": meta.get("Producer"),
                "creation_date": _parse_pdf_date(meta.get("CreationDate")),
                "mod_date": _parse_pdf_date(meta.get("ModDate")),
                "page_count": len(pdf.pages),
                "file_type": "pdf",
                "file_size_bytes": path.stat().st_size,
                "file_path": str(path),
                "checksum_sha256": compute_sha256(path),
            }
    except Exception as e:
        return {
            "title": path.stem,
            "file_type": "pdf",
            "file_path": str(path),
            "file_size_bytes": path.stat().st_size if path.exists() else None,
            "error": f"Failed to extract PDF metadata: {e}",
            "checksum_sha256": compute_sha256(path),
        }

# ------------------------------------------------------------
# DOCX helpers
# ------------------------------------------------------------
def _docx_metadata(path: Path) -> Dict[str, Any]:
    if not docx:
        return {"error": "python-docx not installed"}
    try:
        doc = docx.Document(str(path))
        props = doc.core_properties
        word_count = sum(len(paragraph.text.split()) for paragraph in doc.paragraphs)
        page_estimate = max(1, word_count // 300) if word_count else None
        return {
            "title": props.title or path.stem,
            "author": props.author,
            "creator": props.created.isoformat() if props.created else None,
            "mod_date": props.modified.isoformat() if props.modified else None,
            "page_count": page_estimate,
            "file_type": "docx",
            "file_size_bytes": path.stat().st_size,
            "file_path": str(path),
            "checksum_sha256": compute_sha256(path),
        }
    except Exception as e:
        return {
            "title": path.stem,
            "file_type": "docx",
            "file_path": str(path),
            "file_size_bytes": path.stat().st_size if path.exists() else None,
            "error": f"Failed to extract DOCX metadata: {e}",
            "checksum_sha256": compute_sha256(path),
        }

# ------------------------------------------------------------
# TXT/MD/CSV helpers
# ------------------------------------------------------------
def _txt_metadata(path: Path) -> Dict[str, Any]:
    try:
        stat = path.stat()
        creation = datetime.fromtimestamp(stat.st_ctime).isoformat() if hasattr(stat, "st_ctime") else None
        mod = datetime.fromtimestamp(stat.st_mtime).isoformat() if hasattr(stat, "st_mtime") else None
        return {
            "title": path.stem,
            "author": None,
            "creator": None,
            "creation_date": creation,
            "mod_date": mod,
            "page_count": None,
            "file_type": path.suffix.lstrip("."),
            "file_size_bytes": stat.st_size,
            "file_path": str(path),
            "checksum_sha256": compute_sha256(path),
        }
    except Exception as e:
        return {
            "title": path.stem,
            "file_type": path.suffix.lstrip("."),
            "file_path": str(path),
            "file_size_bytes": None,
            "error": f"Failed to extract TXT/MD/CSV metadata: {e}",
            "checksum_sha256": compute_sha256(path),
        }

# ------------------------------------------------------------
# Generic fallback
# ------------------------------------------------------------
def _generic_metadata(path: Path) -> Dict[str, Any]:
    try:
        stat = path.stat()
        return {
            "title": path.stem,
            "author": None,
            "file_type": path.suffix.lstrip("."),
            "file_size_bytes": stat.st_size,
            "file_path": str(path),
            "checksum_sha256": compute_sha256(path),
        }
    except Exception as e:
        return {
            "title": path.stem,
            "file_type": path.suffix.lstrip("."),
            "file_path": str(path),
            "file_size_bytes": None,
            "error": f"Failed to extract generic metadata: {e}",
            "checksum_sha256": compute_sha256(path),
        }

# ------------------------------------------------------------
# Heuristic text-based enrichment
# ------------------------------------------------------------
def extract_text_metadata(text: str) -> Dict[str, Optional[str]]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    meta = {"title": None, "author": None, "date": None}
    if lines:
        meta["title"] = lines[0][:200]
    for line in lines[:10]:
        if re.search(r"by\s+([A-Z][a-z]+(\s[A-Z][a-z]+)*)", line, re.I):
            match = re.search(r"by\s+([A-Z][a-z]+(\s[A-Z][a-z]+)*)", line, re.I)
            meta["author"] = match.group(1)
        if re.search(r"\b(20\d{2}|19\d{2})\b", line):
            meta["date"] = re.search(r"\b(20\d{2}|19\d{2})\b", line).group(1)
    return meta

def combine_metadata(file_meta: Dict[str, Any], text_meta: Dict[str, Any]) -> Dict[str, Any]:
    combined = {**file_meta, **{k: v for k, v in text_meta.items() if v}}
    combined["metadata_version"] = __version__
    return combined

# ------------------------------------------------------------
# Semantic chunk-level enrichment
# ------------------------------------------------------------
def enrich_chunk_metadata(base_meta: Dict[str, Any], chunk_id: int, chunk_text: str) -> Dict[str, Any]:
    chunk_meta = base_meta.copy()
    chunk_meta.update({
        "chunk_id": chunk_id,
        "chunk_length": len(chunk_text),
        "chunk_checksum": hashlib.sha1(chunk_text.encode("utf-8")).hexdigest(),
        "language": detect_language(chunk_text),
    })
    return chunk_meta

def detect_language(text: str) -> str:
    if re.search(r"[а-яА-Я]", text):  # Cyrillic
        return "ru"
    elif re.search(r"[一-龯]", text):  # Chinese
        return "zh"
    elif re.search(r"[ぁ-んァ-ン]", text):  # Japanese
        return "ja"
    elif re.search(r"[가-힣]", text):  # Korean
        return "ko"
    else:
        return "en"

# ------------------------------------------------------------
# Main metadata extraction entrypoint
# ------------------------------------------------------------
def extract_metadata(file_path: str | Path) -> Dict[str, Any]:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        meta = _pdf_metadata(path)
    elif suffix == ".docx":
        meta = _docx_metadata(path)
    elif suffix in {".txt", ".md", ".csv"}:
        meta = _txt_metadata(path)
    else:
        meta = _generic_metadata(path)
    # Text-based enrichment (optional)
    try:
        text = safe_read(path)
        text_meta = extract_text_metadata(text)
        meta = combine_metadata(meta, text_meta)
    except Exception:
        pass
    return meta

def save_metadata(meta: Dict[str, Any], out_path: str | Path) -> None:
    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

def get_version() -> str:
    return __version__