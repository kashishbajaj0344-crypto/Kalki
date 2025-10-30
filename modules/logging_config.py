# ============================================================
# Kalki v2.0 — logging_config.py v2.2 | 2025-10-22 01:10:11 UTC
# ------------------------------------------------------------
# - Unified, colored logging across all modules
# - Rotating file log support (env or argument)
# - Robust timestamped log file naming (prevents duplicates)
# - Per-module log level control
# - Thread-safe, production-grade configuration
# ============================================================

import logging
import os
import sys
import threading
import datetime
import re
from logging.handlers import RotatingFileHandler

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Environment-level overrides
LOG_LEVEL = os.environ.get("KALKI_LOG_LEVEL", "INFO").upper()
LOG_FILE_ENV = os.environ.get("KALKI_LOG_FILE")  # If set, always used

# Per-module log levels
module_levels = {
    "Kalki.Ingest": logging.DEBUG,
    "Kalki.RetryWorker": logging.INFO,
    # Add more module levels as needed
}

# Optional: coloredlogs for prettier output (if installed)
try:
    import coloredlogs
    HAVE_COLOREDLOGS = True
except ImportError:
    HAVE_COLOREDLOGS = False

_log_lock = threading.Lock()

def add_timestamp_to_logfile(log_file: str) -> str:
    """
    Returns logfile path with a trailing _YYYYMMDD_HHMMSS if not already present.
    """
    base, ext = os.path.splitext(log_file)
    # Regex for a trailing _YYYYMMDD_HHMMSS before the extension
    if re.search(r"_[0-9]{8}_[0-9]{6}$", base):
        # Already has a timestamp, don't add another
        return log_file
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{ts}{ext or '.log'}"

def get_logger(name: str, level: str = None, log_file: str = None) -> logging.Logger:
    """
    Returns a configured logger with rotating file and/or env file handler.
    Usage:
        logger = get_logger("Kalki.Ingest")
    """
    logger = logging.getLogger(name)
    with _log_lock:
        if getattr(logger, "_kalki_configured", False):
            return logger  # Don't double-configure

        # Per-module or default level
        eff_level = module_levels.get(name, level or LOG_LEVEL)
        logger.setLevel(eff_level)

        # Console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
        logger.addHandler(handler)

        # Timestamped log file if requested (argument or env)
        log_file_to_use = log_file or LOG_FILE_ENV
        if log_file_to_use:
            log_file_to_use = add_timestamp_to_logfile(log_file_to_use)
            file_handler = RotatingFileHandler(
                log_file_to_use,
                maxBytes=10_000_000,  # 10 MB
                backupCount=5,
                encoding="utf-8"
            )
            file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
            logger.addHandler(file_handler)

        if HAVE_COLOREDLOGS:
            coloredlogs.install(
                level=eff_level,
                logger=logger,
                fmt=LOG_FORMAT,
                datefmt=LOG_DATEFMT
            )

        logger._kalki_configured = True
    return logger

# Example usage
if __name__ == "__main__":
    logger = get_logger("Kalki.LogDemo", log_file="kalki_demo.log")
    logger.debug("This is a DEBUG message.")
    logger.info("This is an INFO message.")
    logger.warning("This is a WARNING message.")
    logger.error("This is an ERROR message.")
    logger.critical("This is a CRITICAL message.")

# Kalki v2.0 — logging_config.py v2.2 | 2025-10-22 01:10:11 UTC