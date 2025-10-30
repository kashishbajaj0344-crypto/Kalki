# ============================================================
# Kalki 3.0 — Main Entrypoint
# ------------------------------------------------------------
# Initializes core systems (logging, config) and prints a welcome.
# Robust to errors, debug-friendly, and future-proof for CLI/GUI/agents.
# ============================================================

import os
from modules.logging_config import setup_logging, get_logger
from modules.config import __version__, CONFIG_SIGNATURE, DIRS
from modules.utils import safe_execution

# Optional: ASCII splash logo for terminal with separated letters
SPLASH_LOGO = r"""
  _  __     _      _       _  __  _ 
 | |/ /    / \    | |     | |/ / | |
 | ' /    / _ \   | |     | ' /  | |
 | . \   / ___ \  | |___  |  .\  | |
 |_|\_\ /_/   \_\ |_____| |_|\_\ |_|

 K   A   L   K   I   3 . 0 — The Ultimate Personal AI
"""

debug_mode = os.getenv("KALKI_DEBUG", "false").lower() == "true"

@safe_execution(default_return=None)
def main():
    # Set up logging: DEBUG if flag set, else INFO
    setup_logging(log_level="DEBUG" if debug_mode else "INFO")
    logger = get_logger("Kalki.Main")

    print(SPLASH_LOGO)
    logger.info("🧠 Kalki 3.0: The Ultimate Personal AI")
    logger.info("Version: %s", __version__)
    logger.info("Config Signature: %s", CONFIG_SIGNATURE)

    # Optional startup summary
    if debug_mode:
        logger.debug("=== Kalki Directory Layout ===")
        for name, path in DIRS.items():
            logger.debug("%s: %s", name, path)

    # Placeholders for future system initialization
    def init_cli():
        logger.info("[TODO] CLI subsystem not yet implemented.")

    def init_gui():
        logger.info("[TODO] GUI subsystem not yet implemented.")

    def init_eventbus():
        logger.info("[TODO] Eventbus/agent system not yet implemented.")

    # Call placeholders (for clear logs)
    init_cli()
    init_gui()
    init_eventbus()

    logger.info("Welcome! Your AI assistant is starting up...")

if __name__ == "__main__":
    main()

# Kalki Main Entrypoint v0.2 — 2025-10-21