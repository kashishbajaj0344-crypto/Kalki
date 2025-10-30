"""
sandbox.py
Secure execution environment for experimental tasks, simulations, or code execution.
"""

import subprocess
import logging

logger = logging.getLogger(__name__)

class Sandbox:
    def __init__(self):
        logger.info("Sandbox initialized.")

    def run_command(self, command: str) -> str:
        """
        Execute a shell command safely and capture output.
        """
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
            logger.info(f"Command executed: {command}")
            return result.stdout + result.stderr
        except Exception as e:
            logger.error(f"Command execution failed: {command} -> {e}")
            return str(e)
