"""
KALKI v2.3 — FileHash Module v1.4
------------------------------------------------------------
Enterprise-grade file hashing and deduplication utilities.
- SHA256, SHA1, MD5 support; streaming for large files
- Async-safe file/string hashing for pipeline scaling
- Robust error handling, logging, version registration
- Used for deduplication, versioning, checksum audit
"""

import hashlib
from pathlib import Path
from typing import Optional, Set
import asyncio

from modules.logger import get_logger
from modules.config import register_module_version

__version__ = "KALKI v2.3 — filehash.py v1.4"
register_module_version("filehash.py", __version__)

logger = get_logger("FileHash")

def compute_sha256(file_path: Path, chunk_size: int = 65536) -> Optional[str]:
    """Compute SHA256 hash for a file (memory-efficient, streamed)."""
    if not file_path.exists() or not file_path.is_file():
        logger.error("File does not exist or is not a file: %s", file_path)
        return None
    sha256 = hashlib.sha256()
    try:
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                sha256.update(chunk)
        digest = sha256.hexdigest()
        logger.debug("Computed SHA256 for %s: %s", file_path.name, digest)
        return digest
    except Exception as e:
        logger.error("Error computing SHA256 for %s: %s", file_path, e)
        return None

def compute_sha1(file_path: Path, chunk_size: int = 65536) -> Optional[str]:
    """Compute SHA1 hash for a file (streamed)."""
    if not file_path.exists() or not file_path.is_file():
        logger.error("File does not exist or is not a file: %s", file_path)
        return None
    sha1 = hashlib.sha1()
    try:
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                sha1.update(chunk)
        digest = sha1.hexdigest()
        logger.debug("Computed SHA1 for %s: %s", file_path.name, digest)
        return digest
    except Exception as e:
        logger.error("Error computing SHA1 for %s: %s", file_path, e)
        return None

def compute_md5(file_path: Path, chunk_size: int = 65536) -> Optional[str]:
    """Compute MD5 hash for a file (streamed)."""
    if not file_path.exists() or not file_path.is_file():
        logger.error("File does not exist or is not a file: %s", file_path)
        return None
    md5 = hashlib.md5()
    try:
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                md5.update(chunk)
        digest = md5.hexdigest()
        logger.debug("Computed MD5 for %s: %s", file_path.name, digest)
        return digest
    except Exception as e:
        logger.error("Error computing MD5 for %s: %s", file_path, e)
        return None

def hash_string(text: str, algorithm: str = "sha256") -> str:
    """Compute hash of a string (default: SHA256)."""
    try:
        if algorithm == "sha256":
            h = hashlib.sha256()
        elif algorithm == "sha1":
            h = hashlib.sha1()
        elif algorithm == "md5":
            h = hashlib.md5()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        h.update(text.encode("utf-8"))
        digest = h.hexdigest()
        logger.debug("Computed %s for string: %s", algorithm.upper(), digest)
        return digest
    except Exception as e:
        logger.error("Error hashing string: %s", e)
        return "error_hash"

def is_duplicate(file_path: Path, known_hashes: Set[str]) -> bool:
    """
    Checks if the given file is a duplicate by comparing its SHA256 hash
    with a provided set of known hashes.
    Returns True if duplicate, False otherwise (or on error).
    """
    digest = compute_sha256(file_path)
    if digest is None:
        logger.warning("Unable to hash file for duplicate check: %s", file_path)
        return False  # Treat as not duplicate to avoid data loss
    is_dup = digest in known_hashes
    logger.info("File %s is %sduplicate.", file_path, "" if is_dup else "not ")
    return is_dup

# ------------------------------------------------------------
# Async wrappers for pipeline scaling
# ------------------------------------------------------------
async def async_compute_sha256(file_path: Path, chunk_size: int = 65536) -> Optional[str]:
    return await asyncio.to_thread(compute_sha256, file_path, chunk_size)

async def async_hash_string(text: str, algorithm: str = "sha256") -> str:
    return await asyncio.to_thread(hash_string, text, algorithm)

def get_version() -> str:
    return __version__

# ------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python filehash.py <file>")
    else:
        fp = Path(sys.argv[1])
        print("SHA256:", compute_sha256(fp))
        print("SHA1:  ", compute_sha1(fp))
        print("MD5:   ", compute_md5(fp))