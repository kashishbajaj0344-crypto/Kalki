"""
KALKI v2.3 — Logger Module v1.3
------------------------------------------------------------
Centralized, robust logging for all Kalki pipeline modules.
- Colored logs (level-based, coloredlogs optional)
- Rotating file handler (daily or size-based, configurable)
- Dynamic log file naming per run/module (Phase 2-ready)
- Thread-safe global logger registry
- Module-level contextual loggers
- Configurable log level via ENV/CONFIG
- Silent fallback with warning if CONFIG missing
- trace() helper for detailed tracing
- Version registration confirmation (stdout)
"""

import os
import sys
import logging
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler
from pathlib import Path
from typing import Optional
from datetime import datetime

# --- CONFIG import with silent fallback ---
try:
    from modules.config import CONFIG, register_module_version
except ImportError:
    CONFIG = {}
    def register_module_version(module, version): pass

if not CONFIG:
    print("[Kalki.Logger] CONFIG not loaded — using default settings.")

__version__ = "KALKI v2.3 — logger.py v1.3"
register_module_version("logger.py", __version__)
print(f"[Kalki.Logger] {__version__} registered successfully.")

# --- Logging Directory & Dynamic File Naming ---
LOG_DIR = Path(CONFIG.get("log_dir", Path(__file__).resolve().parent.parent / "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_NAME = CONFIG.get("log_name", f"kalki_{timestamp}.log")
LOG_PATH = LOG_DIR / LOG_NAME

# --- Coloredlogs support (optional) ---
try:
    import coloredlogs
except ImportError:
    coloredlogs = None

class KalkiLogger:
    _logger_registry = {}

    @staticmethod
    def get_logger(
        name: str = "Kalki",
        level: Optional[int] = None,
        log_file: Optional[Path] = None,
        rotation: str = "daily",  # "daily" or "size"
        max_bytes: int = 5_000_000,
        backup_count: int = 7,
        propagate: bool = False
    ) -> logging.Logger:
        """
        Return a context-aware logger. Creates one if not exists.
        - Colored logs if available
        - Rotating file handler (daily or size)
        - Console handler always present
        - Dynamic file naming per run/module
        """
        if name in KalkiLogger._logger_registry:
            return KalkiLogger._logger_registry[name]

        logger = logging.getLogger(name)
        logger.propagate = propagate

        # Determine log level from env/config or override
        env_level = CONFIG.get("LOG_LEVEL", "INFO").upper()
        log_level = level if level is not None else getattr(logging, env_level, logging.INFO)
        logger.setLevel(log_level)

        fmt_str = "[%(asctime)s] [%(name)s] [%(levelname)s]: %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt_str, datefmt=datefmt)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Rotating file handler (dynamic file naming)
        file_path = log_file or LOG_PATH
        if rotation == "daily":
            file_handler = TimedRotatingFileHandler(file_path, when="midnight", backupCount=backup_count, encoding="utf-8")
        else:
            file_handler = RotatingFileHandler(file_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Colored logs (optional)
        if coloredlogs:
            coloredlogs.install(
                level=log_level,
                logger=logger,
                fmt=fmt_str,
                datefmt=datefmt
            )

        KalkiLogger._logger_registry[name] = logger
        logger.debug(f"Logger initialized for '{name}' at level {logging.getLevelName(log_level)}")
        return logger

# Global logger shortcut
logger = KalkiLogger.get_logger("Kalki")

def get_logger(module_name: str):
    """Create or retrieve a module-specific logger."""
    return KalkiLogger.get_logger(f"Kalki.{module_name}")

def trace(logger, message: str):
    """Shorthand for ultra-detailed debug tracing."""
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"[TRACE] {message}")

def get_version() -> str:
    return __version__

# ------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------
if __name__ == "__main__":
    log = get_logger("test")
    log.info("Logger initialized successfully.")
    log.debug("Debug mode operational.")
    trace(log, "This is a trace-level debug message.")
    log.warning("Sample warning test.")
    log.error("Error logging verified.")
    log.critical("Critical log verified.")