"""
modules/boot.py
KALKI v2.3 — Boot manager: validates environment, required binaries, and directories.
Performs environment checks, directory creation, binary verification, and version registration.
"""

import shutil
from typing import List
from pathlib import Path
from modules.config import CONFIG, register_module_version
from modules.logger import get_logger
from modules.utils import ensure_dirs

__version__ = "Kalki v2.3 - modules/boot.py - v0.3"
register_module_version("boot.py", __version__)

logger = get_logger("boot")

REQUIRED_DIRS = [
    Path(CONFIG["data_dir"]),
    Path(CONFIG["log_dir"]),
    Path(CONFIG["vector_db_dir"]),
    Path(CONFIG["ingest_dir"]),
]

REQUIRED_BINS = ["python3"]  # Extend as needed (e.g., "tesseract", "git", ...)

class BootManager:
    """
    Handles environment and system checks for Kalki bootstrapping.
    """
    def __init__(self) -> None:
        self.required_bins: List[str] = REQUIRED_BINS
        self.dirs: List[Path] = REQUIRED_DIRS

    def ensure_env(self) -> None:
        """
        Ensures all required directories exist and required binaries are available in PATH.
        """
        logger.info("Ensuring environment for Kalki v2.3 (%s).", CONFIG.get("KALKI_ENV"))
        ensure_dirs(*self.dirs)
        self._check_binaries()
        logger.info("Environment OK — directories and binaries validated.")

    def _check_binaries(self) -> None:
        """
        Checks for required system binaries. Logs warning if any are missing.
        """
        missing = []
        for b in self.required_bins:
            if shutil.which(b) is None:
                missing.append(b)
        if missing:
            logger.warning("Missing system binaries: %s", missing)
        else:
            logger.debug("All required binaries present: %s", self.required_bins)

    def health_check(self) -> bool:
        """
        Performs a minimal health check (write access to data dir).
        Returns True if check passes, False otherwise.
        """
        try:
            testfile = Path(CONFIG["data_dir"]) / ".kalki_healthcheck"
            testfile.write_text("OK", encoding="utf-8")
            testfile.unlink()
            logger.debug("Health check passed for data_dir.")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False