#!/usr/bin/env python3
"""
Kalki v3.0 — utility helpers: JSON, logging, hashing, safe wrappers.
"""
from datetime import datetime
from pathlib import Path
import json
import logging
import hashlib
from typing import Any, Optional

logger = logging.getLogger("kalki.utils")

def now_ts() -> str:
    """Return current timestamp as ISO string."""
    return datetime.now().isoformat()

def load_json(path: Path | str, default: Optional[Any] = None):
    """Load JSON from a file, return default on failure."""
    try:
        path = Path(path)
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.debug(f"load_json error {path}: {e}")
        return default

def save_json(path: Path | str, data: Any):
    """Save JSON to a file, log warning on failure."""
    try:
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"save_json error {path}: {e}")

def safe_log(msg: str):
    """Log message and print to console."""
    logger.info(msg)
    print(msg)

def sha256_file(path: Path | str) -> str:
    """Return SHA256 hash of a file."""
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

if __name__ == "__main__":
    # quick test of sha256 and json save/load
    from pathlib import Path
    p = Path("test_dummy.txt")
    p.write_text("hello")
    h = sha256_file(p)
    safe_log(f"SHA256(test_dummy.txt) = {h}")
    save_json(p.with_suffix(".json"), {"hello": "world"})
    print("utils tests done")
