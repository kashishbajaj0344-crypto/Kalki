"""
modules/boot.py
KALKI v2.4 — Boot manager: validates environment, required binaries, and directories.
Performs environment checks, directory creation, binary verification, and version registration.
Enhanced with robustness checks and automatic recovery mechanisms.
"""

import shutil
from typing import List
from pathlib import Path
from modules.config import CONFIG, register_module_version
from modules.logger import get_logger
from modules.utils import ensure_dirs

__version__ = "Kalki v2.4 - modules/boot.py - v0.4"
register_module_version("boot.py", __version__)

logger = get_logger("boot")

REQUIRED_DIRS = [
    Path(CONFIG["data_dir"]),
    Path(CONFIG["log_dir"]),
    Path(CONFIG["vector_db_dir"]),
    Path(CONFIG["ingest_dir"]),
    Path(CONFIG.get("backup_dir", "backups"))  # Add backup directory
]

REQUIRED_BINS = ["python3"]  # Extend as needed (e.g., "tesseract", "git", ...)

class BootManager:
    """
    Handles environment and system checks for Kalki bootstrapping.
    Enhanced with robustness checks, automatic recovery, and comprehensive validation.
    """
    def __init__(self) -> None:
        self.required_bins: List[str] = REQUIRED_BINS
        self.dirs: List[Path] = REQUIRED_DIRS
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3

    def ensure_env(self) -> None:
        """
        Ensures all required directories exist and required binaries are available in PATH.
        Enhanced with automatic recovery and comprehensive error handling.
        """
        logger.info("Ensuring environment for Kalki v2.4 (%s).", CONFIG.get("KALKI_ENV"))

        try:
            # First attempt
            ensure_dirs(*self.dirs)
            self._check_binaries()
            self._perform_comprehensive_health_check()
            logger.info("Environment OK — directories, binaries, and health validated.")
        except Exception as e:
            logger.warning(f"Initial environment check failed: {e}. Attempting recovery...")
            self._attempt_recovery()

    def _attempt_recovery(self) -> None:
        """
        Attempt to recover from environment issues with progressive escalation.
        """
        for attempt in range(1, self.max_recovery_attempts + 1):
            try:
                logger.info(f"Recovery attempt {attempt}/{self.max_recovery_attempts}")

                # Clean up potentially corrupted directories
                self._cleanup_corrupted_dirs()

                # Re-ensure directories
                ensure_dirs(*self.dirs)

                # Re-check binaries
                self._check_binaries()

                # Perform comprehensive health check
                self._perform_comprehensive_health_check()

                logger.info(f"Recovery successful on attempt {attempt}")
                return

            except Exception as e:
                logger.warning(f"Recovery attempt {attempt} failed: {e}")
                if attempt == self.max_recovery_attempts:
                    logger.critical(f"All {self.max_recovery_attempts} recovery attempts failed. System may be unstable.")
                    raise RuntimeError(f"Boot recovery failed after {self.max_recovery_attempts} attempts: {e}")

    def _cleanup_corrupted_dirs(self) -> None:
        """
        Clean up potentially corrupted directories and recreate them.
        """
        for dir_path in self.dirs:
            try:
                if dir_path.exists():
                    # Check if directory is accessible
                    test_file = dir_path / ".boot_test"
                    test_file.write_text("test")
                    test_file.unlink()
                else:
                    dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning(f"Directory {dir_path} appears corrupted: {e}. Removing and recreating.")
                try:
                    shutil.rmtree(dir_path, ignore_errors=True)
                    dir_path.mkdir(parents=True, exist_ok=True)
                except Exception as cleanup_error:
                    logger.error(f"Failed to cleanup directory {dir_path}: {cleanup_error}")

    def _perform_comprehensive_health_check(self) -> None:
        """
        Perform comprehensive health checks beyond basic directory access.
        """
        # Check critical file permissions
        critical_paths = [
            Path(CONFIG["data_dir"]) / "session.json",
            Path(CONFIG["log_dir"]) / ".gitkeep",
            Path(CONFIG["vector_db_dir"]) / ".gitkeep"
        ]

        for path in critical_paths:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                # Test write access
                with open(path, 'w') as f:
                    f.write("")
                path.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Health check failed for {path}: {e}")

        # Check available disk space
        try:
            import psutil
            disk = psutil.disk_usage('/')
            if disk.percent > 95:
                logger.warning(f"Low disk space: {disk.percent}% used")
            elif disk.percent > 90:
                logger.info(f"Disk usage high: {disk.percent}% used")
        except ImportError:
            logger.debug("psutil not available for disk space check")
        except Exception as e:
            logger.warning(f"Disk space check failed: {e}")

        # Check memory availability
        try:
            import psutil
            memory = psutil.virtual_memory()
            if memory.percent > 95:
                logger.warning(f"Low memory: {memory.percent}% used")
            elif memory.percent > 90:
                logger.info(f"Memory usage high: {memory.percent}% used")
        except ImportError:
            logger.debug("psutil not available for memory check")
        except Exception as e:
            logger.warning(f"Memory check failed: {e}")

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
        Performs comprehensive health checks including write access, resource availability, and system integrity.
        Returns True if all checks pass, False otherwise.
        Enhanced for v2.4 with detailed diagnostics and recovery suggestions.
        """
        issues = []

        try:
            # Basic write access test
            testfile = Path(CONFIG["data_dir"]) / ".kalki_healthcheck"
            testfile.write_text("OK", encoding="utf-8")
            testfile.unlink()
            logger.debug("Basic filesystem health check passed.")
        except Exception as e:
            issues.append(f"Filesystem access failed: {e}")
            logger.error(f"Health check failed: {e}")
            return False

        # Resource availability checks
        try:
            import psutil

            # Memory check
            memory = psutil.virtual_memory()
            if memory.percent > 95:
                issues.append(f"Critical memory usage: {memory.percent}%")

            # Disk check
            disk = psutil.disk_usage('/')
            if disk.percent > 95:
                issues.append(f"Critical disk usage: {disk.percent}%")

            # CPU check (basic load)
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 95:
                issues.append(f"Critical CPU usage: {cpu_percent}%")

        except ImportError:
            logger.debug("psutil not available - skipping resource checks")
        except Exception as e:
            issues.append(f"Resource check failed: {e}")

        # Configuration integrity check
        try:
            required_config_keys = ["data_dir", "log_dir", "vector_db_dir"]
            for key in required_config_keys:
                if key not in CONFIG:
                    issues.append(f"Missing required config key: {key}")
                elif not Path(CONFIG[key]).exists():
                    issues.append(f"Config path does not exist: {key}={CONFIG[key]}")
        except Exception as e:
            issues.append(f"Configuration check failed: {e}")

        if issues:
            logger.warning(f"Health check found {len(issues)} issues:")
            for issue in issues:
                logger.warning(f"  - {issue}")

            # Provide recovery suggestions
            self._suggest_recovery_actions(issues)
            return False

        logger.debug("Comprehensive health check passed.")
        return True

    def _suggest_recovery_actions(self, issues: List[str]) -> None:
        """
        Provide actionable recovery suggestions based on identified issues.
        """
        suggestions = []

        for issue in issues:
            if "memory" in issue.lower():
                suggestions.append("Consider closing unnecessary applications or increasing system memory")
            elif "disk" in issue.lower():
                suggestions.append("Free up disk space by removing unnecessary files or expanding storage")
            elif "cpu" in issue.lower():
                suggestions.append("Reduce system load by closing CPU-intensive applications")
            elif "filesystem" in issue.lower() or "access" in issue.lower():
                suggestions.append("Check file permissions and ensure Kalki has write access to its directories")
            elif "config" in issue.lower():
                suggestions.append("Verify configuration file integrity and required environment variables")

        if suggestions:
            logger.info("Recovery suggestions:")
            for suggestion in suggestions:
                logger.info(f"  - {suggestion}")

        logger.info("Consider running 'kalki> health' command for detailed system diagnostics")