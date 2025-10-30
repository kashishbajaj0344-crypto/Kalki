# modules/chunker.py
"""
KALKI v2.3 — Chunker Module v1.0
------------------------------------------------------------
Adaptive chunking engine for semantic document segmentation.
Integrates paragraph, sentence, and fixed-size chunking modes,
with configurable overlap and async compatibility.

Features:
- Multiple modes: "semantic", "paragraph", "sentence", "fixed"
- Overlap handling for context retention
- Approximate token counting (1 token ≈ 4 chars)
- Optional async wrappers
- Version registration with config
"""

import re
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from modules.config import register_module_version
except ImportError:
    def register_module_version(module: str, version: str):
        pass  # graceful fallback if config not yet available

__version__ = "KALKI v2.3 — Chunker Module v1.0"
register_module_version("chunker.py", __version__)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DEFAULT_MAX_TOKENS = 800
DEFAULT_OVERLAP_TOKENS = 100
CHARS_PER_TOKEN = 4

SENTENCE_SPLITTER = re.compile(r'(?<=[.!?])\s+')
PARAGRAPH_SPLITTER = re.compile(r'\n\s*\n')

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _token_len(text: str) -> int:
    """Estimate token count (approx 1 token ≈ 4 chars)."""
    return max(1, len(text) // CHARS_PER_TOKEN)

def _split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    return [s.strip() for s in SENTENCE_SPLITTER.split(text) if s.strip()]

def _split_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs."""
    return [p.strip() for p in PARAGRAPH_SPLITTER.split(text) if p.strip()]

def _find_piece_start(text: str, piece: str, search_from: int = 0) -> int:
    """Locate where a sub-piece begins (used for char offsets)."""
    if not piece:
        return search_from
    idx = text.find(piece, search_from)
    return idx if idx >= 0 else search_from

# ----------------------------------------------------------------------
# Core Chunking Logic
# ----------------------------------------------------------------------
def chunk_text(
    text: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    mode: str = "semantic"  # "semantic" | "paragraph" | "sentence" | "fixed"
) -> List[Dict[str, Any]]:
    """
    Splits text into semantic or fixed-size chunks with overlap.
    Each chunk dict:
        {
            "chunk_id": int,
            "text": str,
            "start_char": int,
            "end_char": int
        }
    """
    if not text:
        return []

    text_len = len(text)
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN
    max_chars = max_tokens * CHARS_PER_TOKEN
    chunks: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Fixed mode — character window chunking
    # ------------------------------------------------------------------
    if mode == "fixed":
        start = 0
        chunk_id = 0
        while start < text_len:
            end = min(text_len, start + max_chars)
            chunk_txt = text[start:end].strip()
            if chunk_txt:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_txt,
                    "start_char": start,
                    "end_char": end
                })
                chunk_id += 1
            start = max(0, end - overlap_chars)
        return chunks

    # ------------------------------------------------------------------
    # Semantic / paragraph / sentence chunking
    # ------------------------------------------------------------------
    if mode == "paragraph":
        candidates = _split_paragraphs(text)
    elif mode == "sentence":
        candidates = _split_sentences(text)
    else:  # semantic — adaptive
        candidates = _split_paragraphs(text)
        if not candidates or sum(len(c) for c in candidates) < max_chars // 2:
            candidates = _split_sentences(text)

    if not candidates:
        # fallback if no valid segmentation
        return chunk_text(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens, mode="fixed")

    current_buf = ""
    search_pos = 0
    current_start = 0
    chunk_id = 0

    for piece in candidates:
        piece = piece.strip()
        if not piece:
            continue

        piece_start = _find_piece_start(text, piece, search_pos)
        piece_end = piece_start + len(piece)
        search_pos = piece_end

        # if current buffer overflows, flush chunk
        if _token_len(current_buf + " " + piece) > max_tokens and current_buf:
            chunk_text_str = current_buf.strip()
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text_str,
                "start_char": current_start,
                "end_char": current_start + len(chunk_text_str)
            })
            chunk_id += 1

            # retain overlap
            overlap_fragment = chunk_text_str[-overlap_chars:] if overlap_chars else ""
            current_buf = (overlap_fragment + " " + piece).strip()
            current_start = max(current_start, piece_start - len(overlap_fragment))
        else:
            if not current_buf:
                current_start = piece_start
            current_buf += " " + piece if current_buf else piece

    # final flush
    if current_buf.strip():
        chunk_text_str = current_buf.strip()
        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text_str,
            "start_char": current_start,
            "end_char": current_start + len(chunk_text_str)
        })

    return chunks

# ----------------------------------------------------------------------
# Async wrapper
# ----------------------------------------------------------------------
async def chunk_text_async(
    text: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    mode: str = "semantic"
) -> List[Dict[str, Any]]:
    """Async-safe wrapper for chunk_text using asyncio.to_thread."""
    return await asyncio.to_thread(chunk_text, text, max_tokens, overlap_tokens, mode)

# ----------------------------------------------------------------------
# File helper (optional)
# ----------------------------------------------------------------------
def chunk_file(
    file_path: Path,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    mode: str = "semantic"
) -> List[Dict[str, Any]]:
    """Load file content and return chunks."""
    try:
        from modules.utils import safe_read
        text = safe_read(file_path)
    except Exception:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    return chunk_text(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens, mode=mode)

# ----------------------------------------------------------------------
# Module version
# ----------------------------------------------------------------------
def get_version() -> str:
    return __version__
