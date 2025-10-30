#!/usr/bin/env python3
"""
kalki.py — root launcher for Kalki v2.3 (Phase 1 bootstrap)
Starts config, logger, eventbus, session, and runs basic health checks.
"""
from modules.boot import BootManager
from modules.logger import get_logger
from modules.eventbus import EventBus
from modules.session import Session
from modules.config import CONFIG

logger = get_logger("kalki")

def main() -> None:
    logger.info("Kalki v2.3 boot sequence start.")
    boot = BootManager()
    boot.ensure_env()
    eventbus = EventBus()
    session = Session.load_or_create()
    # publish a startup event
    eventbus.publish_sync("kalki.startup", {"session_id": session.session_id})
    logger.info("Kalki v2.3 boot sequence complete. Ready.")
    # keep process alive for interactive dev (replace with real loop later)
    try:
        while True:
            cmd = input("kalki> ").strip().lower()
            if cmd in ("quit", "exit"):
                logger.info("Shutdown requested by user.")
                break
            elif cmd == "status":
                logger.info(f"Session: {session.session_id} | files: {len(session.metadata)}")
            elif cmd == "help" or cmd == "?":
                print("Commands: status | help | exit")
            else:
                print("Unknown command. Type 'help'.")
    except (KeyboardInterrupt, EOFError):
        logger.info("Interrupted — exiting.")
    finally:
        session.save()
        logger.info("Kalki shutdown complete.")

if __name__ == "__main__":
    main()

# Kalki v2.3 — kalki.py — v0.1