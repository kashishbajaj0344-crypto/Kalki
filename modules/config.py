#!/usr/bin/env python3
"""
Kalki v3.0 — config
Provides canonical paths, environment & keyring support, and runtime overrides.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
import keyring

# ---------------- LOAD ENV ----------------
# Load .env file if present (e.g., for OPENAI_API_KEY, POPPLER_PATH, etc.)
load_dotenv()

# ---------------- PATHS ----------------
ROOT_DIR = Path.home() / "Desktop" / "Kalki"
PDF_DIR = ROOT_DIR / "pdfs"
VECTOR_DB_DIR = ROOT_DIR / "vector_db"
RESOURCES_JSON = ROOT_DIR / "kalki_resources.json"
QUERY_LOG = ROOT_DIR / "query_cost.json"
LOG_FILE = ROOT_DIR / "kalki.log"
INGEST_LOCK = ROOT_DIR / "ingest.lock"
SETTINGS_JSON = ROOT_DIR / "kalki_settings.json"

# ---------------- EMBEDDING / LLM ----------------
EMBED_CHUNK_WORDS = int(os.getenv("EMBED_CHUNK_WORDS", "100"))
EMBED_OVERLAP_WORDS = int(os.getenv("EMBED_OVERLAP_WORDS", "20"))
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "30000"))

DEFAULT_EMBED_MODEL = os.getenv("DEFAULT_EMBED_MODEL", "text-embedding-3-large")
DEFAULT_CHAT_MODEL = os.getenv("DEFAULT_CHAT_MODEL", "gpt-4o")
DEFAULT_SUMMARY_MODEL = os.getenv("DEFAULT_SUMMARY_MODEL", DEFAULT_CHAT_MODEL)
DEFAULT_CODE_MODEL = os.getenv("DEFAULT_CODE_MODEL", DEFAULT_CHAT_MODEL)

RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", "2"))
PORTALOCKER_TIMEOUT = int(os.getenv("PORTALOCKER_TIMEOUT", "60"))

# ---------------- OPTIONAL ENV OVERRIDES ----------------
# Poppler path for pdf2image (if needed for OCR/preview)
POPPLER_PATH = os.getenv("POPPLER_PATH")  # e.g., "/usr/local/bin"

# ---------------- KEYRING ----------------
# Keyring/service name for secure API key storage
KEYRING_SERVICE = os.getenv("KALKI_KEYRING_SERVICE", "kalki_openai")

def get_openai_api_key() -> str:
    """Get OpenAI API key, preferring keyring, falling back to .env, and storing if found."""
    # First try keyring
    key = keyring.get_password(KEYRING_SERVICE, "OPENAI_API_KEY")
    if key:
        return key

    # Fallback: check .env
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        # Save into keyring for persistence
        try:
            keyring.set_password(KEYRING_SERVICE, "OPENAI_API_KEY", env_key)
        except Exception:
            pass
        return env_key

    return ""  # No key found

def set_openai_api_key(key_value: str) -> None:
    """Manually set and store OpenAI API key securely in keyring."""
    keyring.set_password(KEYRING_SERVICE, "OPENAI_API_KEY", key_value)

# ---------------- RUNTIME SETTINGS ----------------
def load_runtime_settings() -> dict:
    """Load runtime settings (JSON overrides for GUI/CLI)."""
    try:
        if SETTINGS_JSON.exists():
            return json.loads(SETTINGS_JSON.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def save_runtime_settings(d: dict):
    """Save runtime settings (JSON overrides)."""
    try:
        SETTINGS_JSON.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

# ---------------- ENSURE DIRECTORIES ----------------
for p in [ROOT_DIR, PDF_DIR, VECTOR_DB_DIR]:
    p.mkdir(parents=True, exist_ok=True)
