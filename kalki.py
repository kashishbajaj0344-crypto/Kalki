#!/usr/bin/env python3
"""
kalki.py — root launcher for Kalki v2.4 (Phase 22 foundation)
Starts config, logger, eventbus, session, robustness monitoring, and runs basic health checks.
Enhanced with comprehensive system robustness and error recovery.
"""
from modules.boot import BootManager
from modules.logger import get_logger
from modules.eventbus import EventBus
from modules.session import Session
from modules.config import CONFIG
from modules.robustness import start_robustness_monitoring, stop_robustness_monitoring, get_robustness_manager

logger = get_logger("kalki")

def main() -> None:
    logger.info("Kalki v2.4 boot sequence start.")
    boot = BootManager()
    boot.ensure_env()
    eventbus = EventBus()
    session = Session.load_or_create()

    # Start robustness monitoring
    robustness_manager = start_robustness_monitoring(eventbus)
    logger.info("System robustness monitoring started")

    # publish a startup event
    eventbus.publish_sync("kalki.startup", {"session_id": session.session_id})

    # Register emergency restart handler
    def emergency_restart_handler(event_data):
        reason = event_data.get("reason", "Unknown emergency")
        logger.critical(f"Emergency restart requested: {reason}")
        robustness_manager.trigger_emergency_restart(reason)

    eventbus.subscribe("system.emergency_restart", emergency_restart_handler)

    logger.info("Kalki v2.4 boot sequence complete. Ready.")
    # keep process alive for interactive dev (replace with real loop later)
    try:
        while True:
            cmd = input("kalki> ").strip().lower()
            if cmd in ("quit", "exit"):
                logger.info("Shutdown requested by user.")
                break
            elif cmd == "status":
                health = robustness_manager.get_system_health()
                logger.info(f"Session: {session.session_id} | files: {len(session.metadata)} | health: {health['overall_status']}")
                print(f"System Health: {health['overall_status']}")
                for check_name, check_data in health['checks'].items():
                    print(f"  {check_name}: {check_data['status']}")
            elif cmd == "health":
                health = robustness_manager.get_system_health()
                print("=== System Health Report ===")
                print(f"Overall Status: {health['overall_status']}")
                print(f"CPU: {health['resources']['cpu_percent']:.1f}%")
                print(f"Memory: {health['resources']['memory_percent']:.1f}% ({health['resources']['memory_used_mb']}MB used)")
                print(f"Disk: {health['resources']['disk_usage_percent']:.1f}%")
                print("\nHealth Checks:")
                for check_name, check_data in health['checks'].items():
                    status_icon = "✅" if check_data['status'] == 'healthy' else "⚠️" if check_data['status'] == 'degraded' else "❌"
                    print(f"  {status_icon} {check_name}: {check_data['status']}")
                    if check_data.get('last_error'):
                        print(f"    Error: {check_data['last_error']}")
            elif cmd == "help" or cmd == "?":
                print("Commands: status | health | help | exit")
            else:
                print("Unknown command. Type 'help'.")
    except (KeyboardInterrupt, EOFError):
        logger.info("Interrupted — exiting.")
    except Exception as e:
        logger.exception(f"Unexpected error in main loop: {e}")
        robustness_manager.trigger_emergency_restart(f"Main loop error: {e}")
    finally:
        # Stop robustness monitoring
        stop_robustness_monitoring()
        session.save()
        logger.info("Kalki shutdown complete.")

if __name__ == "__main__":
    main()

# Kalki v2.4 — kalki.py — Enhanced with system robustness and error recovery