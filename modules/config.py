"""
KALKI v2.3 — Config Module v1.5
------------------------------------------------------------
Central configuration manager for all Kalki modules.
- Loads .env with override support (dotenv)
- Ensures all core directories exist, dynamic path helpers
- Exposes runtime CONFIG dict for all modules
- Registers module versions globally for traceability
- Save/load config to disk, runtime env reload, validation
- Phase 2–ready for ingestion, agent, and LLM pipelines
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Dotenv support
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(path, override=False): pass

__version__ = "KALKI v2.3 — config.py v1.5"
CONFIG_SCHEMA_VERSION = "1.0.0"

# Config signature for integrity checking
import hashlib
CONFIG_SIGNATURE = hashlib.md5(str(__version__ + CONFIG_SCHEMA_VERSION).encode()).hexdigest()[:8]

# -------------------------------
# Project root and directories
# -------------------------------
ROOT = Path(__file__).parent.parent.resolve()
ENV_PATH = ROOT / ".env"
load_dotenv(ENV_PATH, override=True)

DATA_DIR = ROOT / "data"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
LOG_DIR = ROOT / "logs"
PDF_DIR = ROOT / "pdfs"
INGEST_DIR = DATA_DIR / "ingested"
SESSION_FILE = DATA_DIR / "session.json"

for p in (DATA_DIR, LOG_DIR, VECTOR_DB_DIR, PDF_DIR, INGEST_DIR):
    p.mkdir(parents=True, exist_ok=True)

# Directory mapping for easy access
DIRS = {
    "root": str(ROOT),
    "data": str(DATA_DIR),
    "vector_db": str(VECTOR_DB_DIR),
    "logs": str(LOG_DIR),
    "pdfs": str(PDF_DIR),
    "ingested": str(INGEST_DIR),
    "session": str(SESSION_FILE)
}

# -------------------------------
# Dynamic log file naming (Phase 2)
# -------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_NAME = os.getenv("LOG_NAME", f"kalki_{timestamp}.log")
LOG_PATH = LOG_DIR / LOG_NAME

# -------------------------------
# Centralized config object
# -------------------------------
def _get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(key, default)

CONFIG: Dict[str, Any] = {
    "kalki_version": "v2.3",
    "config_schema_version": CONFIG_SCHEMA_VERSION,
    "root": str(ROOT),
    "env_path": str(ENV_PATH),
    "data_dir": str(DATA_DIR),
    "vector_db_dir": str(VECTOR_DB_DIR),
    "log_dir": str(LOG_DIR),
    "pdf_dir": str(PDF_DIR),
    "log_name": LOG_NAME,
    "log_path": str(LOG_PATH),
    "ingest_dir": str(INGEST_DIR),
    "session_file": str(SESSION_FILE),
    "LOG_LEVEL": _get_env("LOG_LEVEL", "INFO"),
    "KALKI_ENV": _get_env("KALKI_ENV", "development"),
}

# -------------------------------
# Version Registry (audit/trace)
# -------------------------------
MODULE_VERSIONS: Dict[str, str] = {
    "config.py": __version__,
}

def register_module_version(module_name: str, version: str):
    MODULE_VERSIONS[module_name] = version
    print(f"[Kalki.Config] Registered {module_name}: {version}")

def get_module_versions() -> Dict[str, str]:
    return MODULE_VERSIONS.copy()

# -------------------------------
# Runtime Helpers
# -------------------------------
def get_path(name: str) -> Optional[str]:
    """Retrieve standard project paths dynamically."""
    return CONFIG.get(name.lower())


def get_config(section: str, key: str, default: Optional[Any] = None) -> Any:
        """Compatibility helper used across modules.

        Usage examples in the codebase expect a function with signature
            get_config("llm", "openai_model", "gpt-3.5-turbo")

        This helper will look up values in the CONFIG dictionary with
        the following precedence:
            1. Exact key: "{section}_{key}" (e.g. "llm_openai_model")
            2. Fallback key: "{key}" (e.g. "openai_model")
            3. Finally, return the provided default

        This keeps the function simple and tolerant of missing values.
        """
        # Normalize inputs
        section = (section or "").strip()
        key = (key or "").strip()
        composed = f"{section}_{key}" if section else key

        # 1) Try section_key
        if composed and composed in CONFIG:
                return CONFIG[composed]

        # 2) Try key alone
        if key in CONFIG:
                return CONFIG[key]

        # 3) Not found -> return default
        return default

def reload_env():
    """Reload environment vars at runtime."""
    load_dotenv(ENV_PATH, override=True)
    print("[Kalki.Config] Environment reloaded.")

def save_config_to_file(path: Path = ROOT / "config.json") -> None:
    """Save the CONFIG dictionary to disk (for debugging or audit)."""
    path.write_text(json.dumps(CONFIG, indent=2))
    print(f"[Kalki.Config] CONFIG saved to {path}")

def validate_config() -> None:
    """Check all critical paths and env variables. Raises ValueError if misconfigured."""
    required = [
        ("data_dir", DATA_DIR),
        ("log_dir", LOG_DIR),
        ("vector_db_dir", VECTOR_DB_DIR),
        ("pdf_dir", PDF_DIR),
        ("ingest_dir", INGEST_DIR),
    ]
    for key, val in required:
        if not val or (isinstance(val, Path) and not val.exists()):
            raise ValueError(f"CONFIG ERROR: {key} is missing or invalid ({val})")

# Optionally lock config after startup (advanced, optional)
CONFIG_LOCKED = False
def lock_config():
    global CONFIG_LOCKED
    CONFIG_LOCKED = True

# -------------------------------
# Debug
# -------------------------------
if __name__ == "__main__":
    print("[Kalki.Config] Module loaded successfully.")
    print("Root Dir:", ROOT)
    print("Vector DB:", VECTOR_DB_DIR)
    print("Log Dir:", LOG_DIR)
    print("PDF Dir:", PDF_DIR)
    print("Environment Path:", ENV_PATH)
    print("Registered Versions:", get_module_versions())
    validate_config()
    save_config_to_file()

# Kalki v2.3 — config.py v1.5