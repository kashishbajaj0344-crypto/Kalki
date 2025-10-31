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

@safe_execution(default=None)
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

    # Initialize core subsystems
    def init_cli():
        from modules.cli import cli_ingest, cli_chunk
        logger.info("CLI subsystem initialized successfully.")
        return True

    def init_gui():
        from modules.gui import KalkiGUI
        logger.info("GUI subsystem initialized successfully.")
        return True

    def init_eventbus():
        from modules.eventbus import EventBus
        eventbus = EventBus()
        logger.info("EventBus/agent system initialized successfully.")
        return eventbus

    # Initialize subsystems
    cli_ok = init_cli()
    gui_ok = init_gui()
    eventbus = init_eventbus()

    logger.info("Welcome! Your AI assistant is starting up...")
    logger.info(f"Subsystems initialized - CLI: {cli_ok}, GUI: {gui_ok}, EventBus: {eventbus is not None}")

if __name__ == "__main__":
    main()

# Kalki Main Entrypoint v0.2 — 2025-10-21