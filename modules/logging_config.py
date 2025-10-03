#!/usr/bin/env python3
"""
Central logging configuration for Kalki.
Configures console + rotating file handlers and a common formatter.
Call init_logger() at program start (main.py).
"""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from .config import LOG_FILE

def init_logger(level: int = logging.INFO):
    """
    Initialize root logger with console + rotating file handlers.
    """
    root = logging.getLogger()
    if root.handlers:
        # Already initialized
        return
    root.setLevel(level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ch.setFormatter(ch_formatter)
    root.addHandler(ch)

    # Ensure log folder exists
    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)

    # Rotating file handler
    fh = RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
    fh.setLevel(level)
    fh_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh.setFormatter(fh_formatter)
    root.addHandler(fh)

# ------------------ TEST ------------------
if __name__ == "__main__":
    init_logger()
    logger = logging.getLogger("kalki.logging_config")
    logger.info("Logging configuration test successful.")
