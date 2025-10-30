"""
KALKI v2.3 — Utils Module v0.8
------------------------------------------------------------
Unified utilities for safe execution, file I/O, hashing, logging,
retrying, async operations, and general helpers.
Designed for ingestion, chunking, tagging, and vector DB pipelines.
"""

import functools
import hashlib
import logging
import asyncio
from pathlib import Path
from typing import Any, Callable, Optional, List

__version__ = "KALKI v2.3 — utils.py v0.8"

# =============================================================
# Logger setup
# =============================================================
def get_logger(name: str = "kalki_utils", level: int = logging.INFO) -> logging.Logger:
    """Return a standard logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger

logger = get_logger()

# =============================================================
# Safe execution decorator
# =============================================================
def safe_execution(default: Any = None):
    """Decorator to safely execute a function and log exceptions."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception("Error in %s: %s", func.__name__, e)
                return default
        return wrapper
    return decorator

# =============================================================
# File I/O helpers
# =============================================================
@safe_execution(default="")
def safe_read(file_path: Path, encoding: str = "utf-8") -> str:
    """Read text file safely, returns empty string if fails."""
    file_path = Path(file_path)
    if not file_path.exists():
        logger.warning("File does not exist: %s", file_path)
        return ""
    with file_path.open("r", encoding=encoding, errors="ignore") as f:
        return f.read()

@safe_execution(default=False)
def safe_write(file_path: Path, content: str, overwrite: bool = True) -> bool:
    """Write text safely; create directories if needed."""
    file_path = Path(file_path)
    if not overwrite and file_path.exists():
        logger.warning("File exists and overwrite=False: %s", file_path)
        return False
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        f.write(content)
    return True

@safe_execution(default=[])
def read_lines(file_path: Path) -> List[str]:
    """Return non-empty lines from a file."""
    content = safe_read(file_path)
    return [line.strip() for line in content.splitlines() if line.strip()]

# =============================================================
# Async-safe I/O
# =============================================================
async def async_safe_read(file_path: Path, encoding: str = "utf-8") -> str:
    """Async-safe text reader using asyncio.to_thread."""
    return await asyncio.to_thread(safe_read, file_path, encoding)

async def async_safe_write(file_path: Path, content: str, overwrite: bool = True):
    """Async-safe text writer."""
    return await asyncio.to_thread(safe_write, file_path, content, overwrite)

async def async_read_lines(file_path: Path) -> List[str]:
    """Async-safe line reader."""
    text = await async_safe_read(file_path)
    return [line.strip() for line in text.splitlines() if line.strip()]

# =============================================================
# Hashing helpers
# =============================================================
@safe_execution(default="error_hash")
def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file (memory-efficient)."""
    file_path = Path(file_path)
    if not file_path.exists():
        logger.warning("File not found for hashing: %s", file_path)
        return "error_hash"
    sha256_hash = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

@safe_execution(default="error_hash")
def compute_sha256_str(data: str) -> str:
    """Compute SHA256 hash of a string."""
    try:
        return hashlib.sha256(data.encode("utf-8")).hexdigest()
    except Exception as e:
        logger.error("Failed to hash string: %s", e)
        return "error_hash"

# =============================================================
# Retry decorators
# =============================================================
def retry(times: int = 3, delay: float = 1.0, exceptions=(Exception,)):
    """Retry decorator for synchronous functions."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    logger.warning("Attempt %d failed for %s: %s", attempt, func.__name__, e)
                    if attempt < times:
                        import time
                        time.sleep(delay * attempt)
            return None
        return wrapper
    return decorator

def async_retry(times: int = 3, delay: float = 1.0, exceptions=(Exception,)):
    """Retry decorator for asynchronous functions."""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(1, times + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    logger.warning("Async attempt %d failed for %s: %s", attempt, func.__name__, e)
                    if attempt < times:
                        await asyncio.sleep(delay * attempt)
            return None
        return wrapper
    return decorator

# =============================================================
# Path helpers
# =============================================================
def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists (create if missing)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

# Backwards-compatible alias: some modules import `ensure_dirs` (plural)
def ensure_dirs(path: Path) -> Path:
    """Compatibility wrapper for older code that calls `ensure_dirs`.

    Delegates to `ensure_dir` to maintain a single implementation.
    """
    return ensure_dir(path)

def list_files_recursively(directory: Path, extensions: Optional[List[str]] = None) -> List[Path]:
    """List all files in a directory recursively matching extensions."""
    directory = Path(directory)
    extensions = [ext.lower() for ext in (extensions or [])]
    return [f for f in directory.rglob("*") if f.is_file() and (not extensions or f.suffix.lower() in extensions)]

# =============================================================
# Module version
# =============================================================
def get_version() -> str:
    return __version__