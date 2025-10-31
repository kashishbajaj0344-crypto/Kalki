# ============================================================
# Kalki v2.0 — preprocessor.py v2.0
# ------------------------------------------------------------
# - Text normalization, cleaning, splitting, boilerplate removal
# - Ready for PDF/image OCR hooks and custom user filters
# - Pluggable for ingestion, RAG, and ML pipelines
# ============================================================

import re
import unicodedata
from typing import List, Callable, Optional
import asyncio

from modules.logging_config import get_logger

logger = get_logger("Kalki.Preprocessor")

# ----------- Normalization Utilities -----------
def normalize_unicode(text: str) -> str:
    """Normalize unicode characters to NFKC form."""
    return unicodedata.normalize("NFKC", text)

def standardize_whitespace(text: str) -> str:
    """Replace multiple spaces/newlines with single equivalents."""
    text = re.sub(r'\r\n?', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    return text.strip()

def remove_control_chars(text: str) -> str:
    return ''.join(c for c in text if unicodedata.category(c)[0] != "C" or c in "\n\t")

def fix_punctuation_spacing(text: str) -> str:
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    return text

def remove_boilerplate(text: str, boilerplate_patterns: Optional[List[str]] = None) -> str:
    """Remove common boilerplate using regex patterns."""
    if not boilerplate_patterns:
        boilerplate_patterns = [
            r"Page \d+ of \d+",
            r"^\s*Confidential\s*$",
            r"^\s*This is a computer-generated document.*$"
        ]
    clean = text
    for pat in boilerplate_patterns:
        clean = re.sub(pat, '', clean, flags=re.MULTILINE)
    return clean

def clean_text(text: str) -> str:
    """Run all cleaning steps."""
    text = normalize_unicode(text)
    text = remove_control_chars(text)
    text = standardize_whitespace(text)
    text = fix_punctuation_spacing(text)
    text = remove_boilerplate(text)
    return text.strip()

# ----------- Splitting Utilities -----------
def split_by_paragraph(text: str) -> List[str]:
    """Split text by paragraphs (two or more newlines)."""
    return [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

def split_by_sentence(text: str) -> List[str]:
    """Split text into sentences using regex (naive)."""
    sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sents if s.strip()]

def split_by_chunk(text: str, max_tokens: int = 512) -> List[str]:
    """Split text into chunks by token count (naive whitespace tokens)."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunk = ' '.join(words[i:i+max_tokens])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks

# ----------- Main Preprocessing Pipeline -----------
def preprocess(
    text: str,
    cleaning: bool = True,
    split: Optional[str] = None,  # None, 'paragraph', 'sentence', 'chunk'
    chunk_size: int = 512,
    custom_filters: Optional[List[Callable[[str], str]]] = None
) -> List[str]:
    """
    Full preprocessing pipeline: cleaning + splitting + custom filters.
    Returns a list of processed text units (for ingestion, RAG, etc).
    """
    logger.debug(f"Preprocessing text: {len(text)} chars")
    if cleaning:
        text = clean_text(text)
    if custom_filters:
        for f in custom_filters:
            text = f(text)
    if split == "paragraph":
        units = split_by_paragraph(text)
    elif split == "sentence":
        units = split_by_sentence(text)
    elif split == "chunk":
        units = split_by_chunk(text, max_tokens=chunk_size)
    else:
        units = [text.strip()]
    logger.debug(f"Preprocessing produced {len(units)} units")
    return units

# ----------- Async version for pipeline -----------
async def preprocess_text_async(
    text: str,
    cleaning: bool = True,
    split: Optional[str] = None,  # None, 'paragraph', 'sentence', 'chunk'
    chunk_size: int = 512,
    custom_filters: Optional[List[Callable[[str], str]]] = None
) -> List[str]:
    """
    Async version of preprocess function for pipeline usage.
    """
    # Since preprocessing is CPU-bound and fast, we can just call the sync version
    # In a thread pool if needed, but for now this is sufficient
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, preprocess, text, cleaning, split, chunk_size, custom_filters)

# ----------- Optional: PDF/Image OCR Hook -----------
def preprocess_pdf_ocr(text: str) -> str:
    """Stub for OCR postprocessing. (Extend as needed.)"""
    return clean_text(text)

# ----------- CLI test/demo -----------
if __name__ == "__main__":
    import sys
    if not sys.argv[1:]:
        print("Usage: python modules/preprocessor.py <file.txt>")
        sys.exit(1)
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        raw = f.read()
    cleaned = preprocess(raw, split="paragraph")
    for i, unit in enumerate(cleaned):
        print(f"\n--- Unit {i+1} ---\n{unit}")

# Kalki v2.0 — preprocessor.py v2.0