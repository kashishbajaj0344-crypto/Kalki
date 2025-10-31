"""
KALKI v2.3 — Tagger Module v1.1
------------------------------------------------------------
Full-featured tag generation for document chunks.
- Keyword extraction (stopwords, regex, config)
- Domain detection (academic, financial, legal, technical, medical, general)
- Structured pattern tags (dates, emails, URLs, numbers)
- Chunk-level enrichment (language, checksum)
- Logging, error handling, config, version registration
- Integrates with metadata/chunker, LLM-ready
"""

import re
import hashlib
from typing import List, Dict, Any
from collections import Counter

try:
    from modules.logger import get_logger
except ImportError:
    def get_logger(name="tagger"): import logging; return logging.getLogger(name)
logger = get_logger("tagger")

try:
    from modules.config import register_module_version, CONFIG
except ImportError:
    def register_module_version(module, version): pass
    CONFIG = {}

try:
    from modules.metadata import detect_language
except ImportError:
    def detect_language(text): return "en"

__version__ = "KALKI v2.3 — tagger.py v1.1"
register_module_version("tagger.py", __version__)

# -----------------------------------
# CONFIG
# -----------------------------------
MIN_WORD_LEN = 3
TOP_N = 10
STOPWORDS = set([
    "the", "and", "for", "are", "but", "not", "with", "you", "this", "that",
    "was", "have", "from", "they", "his", "her", "she", "has", "had", "were",
    "which", "their", "will", "would", "there", "what", "when", "where",
    "your", "can", "all", "any", "our", "out", "use", "how", "who", "its",
    "may", "one", "about", "also", "into", "more", "other", "some", "such",
    "only", "than", "then", "now", "over", "new", "these", "could", "them",
    "because", "very", "even", "most", "must", "each", "many", "much", "every"
])

DOMAIN_KEYWORDS = {
    "academic": ["research", "study", "analysis", "results", "paper", "university", "experiment", "findings"],
    "financial": ["invoice", "transaction", "amount", "balance", "payment", "receipt", "bill", "tax"],
    "legal": ["agreement", "contract", "party", "terms", "law", "jurisdiction", "liability", "signature"],
    "technical": ["system", "architecture", "algorithm", "data", "api", "module", "code", "design"],
    "medical": ["patient", "diagnosis", "treatment", "symptom", "disease", "clinical", "hospital", "health"],
    "general": []
}

TAG_PATTERNS = {
    "date": r"\b(20\d{2}|19\d{2})[-/\.]?(0[1-9]|1[0-2])?[-/\.]?(0[1-9]|[12]\d|3[01])?\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "url": r"\bhttps?://[^\s<>\"']+\b",
    "number": r"\b\d+(?:\.\d+)?\b"
}

# -----------------------------------
# Core Tagging Functions
# -----------------------------------

def extract_keywords(text: str, top_n: int = TOP_N) -> List[str]:
    """
    Extracts keywords from text.  
    - Uses regex, stopwords, and word length.
    - Counts frequency, returns top N.
    """
    if not text:
        return []
    try:
        words = re.findall(r'\b\w+\b', text.lower())
        words = [w for w in words if len(w) >= MIN_WORD_LEN and w not in STOPWORDS]
        freq = Counter(words)
        top_keywords = [w for w, _ in freq.most_common(top_n)]
        return top_keywords
    except Exception as e:
        logger.error("Keyword extraction failed: %s", e)
        return []

def detect_domain(text: str) -> str:
    """
    Detects document domain based on presence of domain-specific keywords.
    """
    domain_scores = {k: 0 for k in DOMAIN_KEYWORDS}
    text_lower = text.lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                domain_scores[domain] += 1
    best_domain = max(domain_scores, key=domain_scores.get)
    if domain_scores[best_domain] == 0:
        best_domain = "general"
    return best_domain

def extract_tags(text: str) -> Dict[str, List[str]]:
    """
    Detects structured tags (dates, emails, URLs, numbers) in text.
    """
    tags: Dict[str, List[str]] = {}
    for name, pattern in TAG_PATTERNS.items():
        matches = re.findall(pattern, text)
        if matches:
            # Flatten tuples from regex groups
            if isinstance(matches[0], tuple):
                flat = list(set(["".join(m) for m in matches]))
                tags[name] = flat
            else:
                tags[name] = list(set(matches))
    return tags

def generate_tags(chunk: Dict[str, Any], method: str = "keywords", top_n: int = TOP_N) -> List[str]:
    """
    Generates tags for a given chunk dict using rule-based keyword extraction.
    Accepts:
        chunk = {"text": ..., "chunk_id": ..., ...}
    Returns list of tags.
    """
    text = chunk.get("text", "")
    tags = extract_keywords(text, top_n)
    logger.debug("Generated %d tags for chunk %s", len(tags), chunk.get("chunk_id"))
    return tags

def tag_chunk(chunk_text: str, base_meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enriches a chunk with domain, language, keywords, pattern tags, and checksum.
    """
    domain = detect_domain(chunk_text)
    language = detect_language(chunk_text)
    keywords = extract_keywords(chunk_text, TOP_N)
    structured_tags = extract_tags(chunk_text)
    chunk_tags = {
        "domain": domain,
        "language": language,
        "keywords": keywords,
        "patterns": structured_tags,
        "checksum": hashlib.sha1(chunk_text.encode("utf-8")).hexdigest(),
    }
    enriched = base_meta.copy()
    enriched.update(chunk_tags)
    return enriched

def get_version() -> str:
    return __version__

# -----------------------------------
# Example Usage
# -----------------------------------
# chunk = {"text": "This is a research paper by John Doe in 2020 about algorithms."}
# tags = generate_tags(chunk)
# print(tags)
# meta = tag_chunk(chunk["text"], {})
# print(meta)