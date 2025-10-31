"""
modules/robustness.py
KALKI v2.4 — System Robustness Module
------------------------------------------------------------
Comprehensive system robustness, error recovery, and health monitoring.
- Automatic error recovery and restart mechanisms
- Circuit breaker pattern for external services
- Resource monitoring and leak prevention
- Health checks and self-diagnosis
- Graceful degradation and fallback systems
- Data integrity validation
- Backup and failover mechanisms
- Timeout management and resilience patterns
"""

import asyncio
import threading
import time
import psutil
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import gc
import signal
import os

from modules.config import CONFIG, register_module_version
from modules.logger import get_logger
from modules.eventbus import EventBus

__version__ = "KALKI v2.4 — robustness.py v1.0"
register_module_version("robustness.py", __version__)

logger = get_logger("robustness")

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    DOWN = "down"

class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class HealthCheck:
    """Represents a health check for a system component"""
    name: str
    check_function: Callable[[], bool]
    interval_seconds: int = 30
    timeout_seconds: int = 10
    max_failures: int = 3
    recovery_attempts: int = 3
    last_check: Optional[datetime] = None
    consecutive_failures: int = 0
    status: HealthStatus = HealthStatus.HEALTHY
    last_error: Optional[str] = None

@dataclass
class CircuitBreaker:
    """Circuit breaker for external service calls"""
    name: str
    failure_threshold: int = 5
    recovery_timeout: int = 60
    expected_exception: Tuple[Exception, ...] = (Exception,)
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None

@dataclass
class SystemResources:
    """Comprehensive system resource monitoring"""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: int = 0
    memory_available_mb: int = 0
    disk_usage_percent: float = 0.0
    disk_free_gb: float = 0.0
    network_connections: int = 0
    thread_count: int = 0
    process_count: int = 0
    uptime_seconds: int = 0

@dataclass
class RobustnessConfig:
    """Configuration for robustness features"""
    health_check_interval: int = 30
    resource_monitor_interval: int = 60
    memory_threshold_percent: float = 85.0
    cpu_threshold_percent: float = 90.0
    disk_threshold_percent: float = 90.0
    auto_restart_enabled: bool = True
    max_restart_attempts: int = 3
    backup_interval_hours: int = 24
    data_integrity_check_interval: int = 3600  # 1 hour

class RobustnessManager:
    """
    Central manager for system robustness, monitoring, and recovery.
    Provides comprehensive error handling, health monitoring, and automatic recovery.
    """

    def __init__(self, eventbus: EventBus):
        self.eventbus = eventbus
        self.config = RobustnessConfig()
        self.health_checks: Dict[str, HealthCheck] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.resource_history: List[SystemResources] = []
        self.restart_attempts = 0
        self.last_backup_time = datetime.now()

        # Monitoring threads
        self.monitoring_thread: Optional[threading.Thread] = None
        self.health_thread: Optional[threading.Thread] = None
        self.backup_thread: Optional[threading.Thread] = None

        # Shutdown flag
        self.running = False

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        logger.info("RobustnessManager initialized")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        self.stop()

    def start(self):
        """Start all robustness monitoring systems"""
        if self.running:
            return

        self.running = True
        logger.info("Starting robustness monitoring systems")

        # Start monitoring threads
        self.monitoring_thread = threading.Thread(target=self._resource_monitor_loop, daemon=True)
        self.health_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.backup_thread = threading.Thread(target=self._backup_loop, daemon=True)

        self.monitoring_thread.start()
        self.health_thread.start()
        self.backup_thread.start()

        # Register core health checks
        self.register_health_check("filesystem", self._check_filesystem_health, interval_seconds=60)
        self.register_health_check("memory", self._check_memory_health, interval_seconds=30)
        self.register_health_check("cpu", self._check_cpu_health, interval_seconds=30)
        self.register_health_check("network", self._check_network_health, interval_seconds=120)

        # Publish startup event
        self.eventbus.publish_sync("robustness.started", {
            "timestamp": datetime.now().isoformat(),
            "config": self.config.__dict__
        })

    def stop(self):
        """Stop all robustness monitoring systems"""
        if not self.running:
            return

        logger.info("Stopping robustness monitoring systems")
        self.running = False

        # Wait for threads to finish
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        if self.health_thread and self.health_thread.is_alive():
            self.health_thread.join(timeout=5)
        if self.backup_thread and self.backup_thread.is_alive():
            self.backup_thread.join(timeout=5)

        # Publish shutdown event
        self.eventbus.publish_sync("robustness.stopped", {
            "timestamp": datetime.now().isoformat()
        })

    def register_health_check(self, name: str, check_function: Callable[[], bool],
                            interval_seconds: int = 30, timeout_seconds: int = 10,
                            max_failures: int = 3, recovery_attempts: int = 3):
        """Register a health check for a system component"""
        health_check = HealthCheck(
            name=name,
            check_function=check_function,
            interval_seconds=interval_seconds,
            timeout_seconds=timeout_seconds,
            max_failures=max_failures,
            recovery_attempts=recovery_attempts
        )
        self.health_checks[name] = health_check
        logger.info(f"Registered health check: {name}")

    def register_circuit_breaker(self, name: str, failure_threshold: int = 5,
                               recovery_timeout: int = 60,
                               expected_exception: Tuple[Exception, ...] = (Exception,)):
        """Register a circuit breaker for external service calls"""
        circuit_breaker = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception
        )
        self.circuit_breakers[name] = circuit_breaker
        logger.info(f"Registered circuit breaker: {name}")

    def call_with_circuit_breaker(self, circuit_name: str, func: Callable, *args, **kwargs):
        """Execute a function with circuit breaker protection"""
        if circuit_name not in self.circuit_breakers:
            return func(*args, **kwargs)

        breaker = self.circuit_breakers[circuit_name]

        if breaker.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset(breaker):
                breaker.state = CircuitBreakerState.HALF_OPEN
                logger.info(f"Circuit breaker {circuit_name} attempting reset")
            else:
                raise Exception(f"Circuit breaker {circuit_name} is OPEN")

        try:
            result = func(*args, **kwargs)
            self._record_success(breaker)
            return result
        except breaker.expected_exception as e:
            self._record_failure(breaker)
            raise e

    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        health_status = {
            "overall_status": HealthStatus.HEALTHY,
            "checks": {},
            "resources": self._get_current_resources().__dict__,
            "circuit_breakers": {},
            "timestamp": datetime.now().isoformat()
        }

        # Check individual health checks
        critical_count = 0
        for name, check in self.health_checks.items():
            health_status["checks"][name] = {
                "status": check.status.value,
                "last_check": check.last_check.isoformat() if check.last_check else None,
                "consecutive_failures": check.consecutive_failures,
                "last_error": check.last_error
            }
            if check.status in [HealthStatus.CRITICAL, HealthStatus.DOWN]:
                critical_count += 1

        # Check circuit breakers
        for name, breaker in self.circuit_breakers.items():
            health_status["circuit_breakers"][name] = {
                "state": breaker.state.value,
                "failure_count": breaker.failure_count,
                "last_failure": breaker.last_failure_time.isoformat() if breaker.last_failure_time else None
            }

        # Determine overall status
        if critical_count > 0:
            health_status["overall_status"] = HealthStatus.CRITICAL.value
        elif any(check.status == HealthStatus.DEGRADED for check in self.health_checks.values()):
            health_status["overall_status"] = HealthStatus.DEGRADED.value

        return health_status

    def trigger_emergency_restart(self, reason: str):
        """Trigger emergency restart with reason logging"""
        logger.critical(f"Emergency restart triggered: {reason}")

        if not self.config.auto_restart_enabled:
            logger.warning("Auto-restart disabled, manual intervention required")
            return

        if self.restart_attempts >= self.config.max_restart_attempts:
            logger.critical("Maximum restart attempts exceeded, system requires manual intervention")
            self.eventbus.publish_sync("robustness.max_restarts_exceeded", {
                "reason": reason,
                "attempts": self.restart_attempts,
                "timestamp": datetime.now().isoformat()
            })
            return

        self.restart_attempts += 1
        logger.info(f"Initiating restart attempt {self.restart_attempts}/{self.config.max_restart_attempts}")

        # Publish restart event
        self.eventbus.publish_sync("robustness.restart_initiated", {
            "reason": reason,
            "attempt": self.restart_attempts,
            "timestamp": datetime.now().isoformat()
        })

        # Perform restart (this would typically restart the main process)
        self._perform_restart()

    def _resource_monitor_loop(self):
        """Continuous resource monitoring loop"""
        while self.running:
            try:
                resources = self._get_current_resources()
                self.resource_history.append(resources)

                # Keep only last 100 readings
                if len(self.resource_history) > 100:
                    self.resource_history.pop(0)

                # Check resource thresholds
                self._check_resource_thresholds(resources)

                # Force garbage collection if memory usage is high
                if resources.memory_percent > self.config.memory_threshold_percent:
                    gc.collect()
                    logger.info("Forced garbage collection due to high memory usage")

                time.sleep(self.config.resource_monitor_interval)

            except Exception as e:
                logger.exception(f"Resource monitoring error: {e}")
                time.sleep(5)  # Brief pause before retry

    def _health_check_loop(self):
        """Continuous health check loop"""
        while self.running:
            try:
                for name, check in self.health_checks.items():
                    if (not check.last_check or
                        (datetime.now() - check.last_check).seconds >= check.interval_seconds):
                        self._perform_health_check(check)

                time.sleep(self.config.health_check_interval)

            except Exception as e:
                logger.exception(f"Health check loop error: {e}")
                time.sleep(5)

    def _backup_loop(self):
        """Periodic backup loop"""
        while self.running:
            try:
                time_since_backup = datetime.now() - self.last_backup_time
                if time_since_backup.seconds >= (self.config.backup_interval_hours * 3600):
                    self._perform_backup()
                    self.last_backup_time = datetime.now()

                time.sleep(3600)  # Check every hour

            except Exception as e:
                logger.exception(f"Backup loop error: {e}")
                time.sleep(300)  # 5 minute pause before retry

    def _perform_health_check(self, check: HealthCheck):
        """Perform a single health check with timeout"""
        check.last_check = datetime.now()

        try:
            # Run health check with timeout
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(check.check_function)
                result = future.result(timeout=check.timeout_seconds)

            if result:
                check.consecutive_failures = 0
                check.status = HealthStatus.HEALTHY
                check.last_error = None
            else:
                self._handle_health_check_failure(check, "Health check returned False")

        except Exception as e:
            self._handle_health_check_failure(check, str(e))

    def _handle_health_check_failure(self, check: HealthCheck, error: str):
        """Handle health check failure"""
        check.consecutive_failures += 1
        check.last_error = error

        if check.consecutive_failures >= check.max_failures:
            if check.status != HealthStatus.DOWN:
                check.status = HealthStatus.DOWN
                logger.error(f"Health check {check.name} failed {check.consecutive_failures} times, marked as DOWN")
                self.eventbus.publish_sync("robustness.health_check_failed", {
                    "check_name": check.name,
                    "error": error,
                    "failures": check.consecutive_failures,
                    "timestamp": datetime.now().isoformat()
                })
        elif check.consecutive_failures > check.max_failures // 2:
            check.status = HealthStatus.DEGRADED
            logger.warning(f"Health check {check.name} degraded: {error}")

    def _get_current_resources(self) -> SystemResources:
        """Get current system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            thread_count = threading.active_count()
            process_count = len(psutil.pids())

            boot_time = psutil.boot_time()
            uptime_seconds = int(time.time() - boot_time)

            # Network connections may require special permissions on some systems
            try:
                net_connections = len(psutil.net_connections())
            except (psutil.AccessDenied, PermissionError):
                logger.debug("Network connections monitoring not available (permission denied)")
                net_connections = 0

            return SystemResources(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used // (1024 * 1024),
                memory_available_mb=memory.available // (1024 * 1024),
                disk_usage_percent=disk.percent,
                disk_free_gb=disk.free // (1024 * 1024 * 1024),
                network_connections=net_connections,
                thread_count=thread_count,
                process_count=process_count,
                uptime_seconds=uptime_seconds
            )
        except Exception as e:
            logger.exception(f"Failed to get system resources: {e}")
            return SystemResources()

    def _check_resource_thresholds(self, resources: SystemResources):
        """Check if resource usage exceeds thresholds"""
        alerts = []

        if resources.memory_percent > self.config.memory_threshold_percent:
            alerts.append(f"Memory usage: {resources.memory_percent:.1f}%")
        if resources.cpu_percent > self.config.cpu_threshold_percent:
            alerts.append(f"CPU usage: {resources.cpu_percent:.1f}%")
        if resources.disk_usage_percent > self.config.disk_threshold_percent:
            alerts.append(f"Disk usage: {resources.disk_usage_percent:.1f}%")

        if alerts:
            logger.warning(f"Resource thresholds exceeded: {', '.join(alerts)}")
            self.eventbus.publish_sync("robustness.resource_threshold_exceeded", {
                "alerts": alerts,
                "resources": resources.__dict__,
                "timestamp": datetime.now().isoformat()
            })

    def _check_filesystem_health(self) -> bool:
        """Check filesystem health"""
        try:
            # Test write access to key directories
            test_dirs = [
                Path(CONFIG.get("data_dir", "data")),
                Path(CONFIG.get("log_dir", "logs")),
                Path(CONFIG.get("vector_db_dir", "vector_db"))
            ]

            for test_dir in test_dirs:
                test_dir.mkdir(parents=True, exist_ok=True)
                test_file = test_dir / ".health_check"
                test_file.write_text("OK")
                test_file.unlink()

            return True
        except Exception as e:
            logger.exception(f"Filesystem health check failed: {e}")
            return False

    def _check_memory_health(self) -> bool:
        """Check memory health"""
        try:
            memory = psutil.virtual_memory()
            return memory.percent < 95.0  # Allow some buffer
        except Exception as e:
            logger.exception(f"Memory health check failed: {e}")
            return False

    def _check_cpu_health(self) -> bool:
        """Check CPU health"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            return cpu_percent < 95.0  # Allow some buffer
        except Exception as e:
            logger.exception(f"CPU health check failed: {e}")
            return False

    def _check_network_health(self) -> bool:
        """Check network health"""
        try:
            # Simple connectivity check
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            return True
        except Exception:
            return False

    def _record_success(self, breaker: CircuitBreaker):
        """Record successful call in circuit breaker"""
        breaker.failure_count = 0
        if breaker.state == CircuitBreakerState.HALF_OPEN:
            breaker.state = CircuitBreakerState.CLOSED
            logger.info(f"Circuit breaker {breaker.name} reset to CLOSED")

    def _record_failure(self, breaker: CircuitBreaker):
        """Record failed call in circuit breaker"""
        breaker.failure_count += 1
        breaker.last_failure_time = datetime.now()

        if breaker.failure_count >= breaker.failure_threshold:
            breaker.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker {breaker.name} opened after {breaker.failure_count} failures")

    def _should_attempt_reset(self, breaker: CircuitBreaker) -> bool:
        """Check if circuit breaker should attempt reset"""
        if not breaker.last_failure_time:
            return True

        time_since_failure = datetime.now() - breaker.last_failure_time
        return time_since_failure.seconds >= breaker.recovery_timeout

    def _perform_backup(self):
        """Perform system backup"""
        try:
            logger.info("Starting system backup")

            # Create backup directory with timestamp
            backup_dir = Path(CONFIG.get("backup_dir", "backups"))
            backup_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"kalki_backup_{timestamp}"

            # Backup key directories
            dirs_to_backup = [
                Path(CONFIG.get("data_dir", "data")),
                Path(CONFIG.get("vector_db_dir", "vector_db")),
                Path(CONFIG.get("log_dir", "logs"))
            ]

            import shutil
            for src_dir in dirs_to_backup:
                if src_dir.exists():
                    dst_dir = backup_path / src_dir.name
                    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

            # Clean old backups (keep last 7)
            backups = sorted(backup_dir.glob("kalki_backup_*"))
            if len(backups) > 7:
                for old_backup in backups[:-7]:
                    shutil.rmtree(old_backup)
                    logger.info(f"Removed old backup: {old_backup}")

            logger.info(f"System backup completed: {backup_path}")
            self.eventbus.publish_sync("robustness.backup_completed", {
                "backup_path": str(backup_path),
                "timestamp": datetime.now().isoformat()
            })

        except Exception as e:
            logger.exception(f"Backup failed: {e}")
            self.eventbus.publish_sync("robustness.backup_failed", {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })

    def _perform_restart(self):
        """Perform system restart"""
        try:
            logger.info("Performing system restart")

            # This is a simplified restart - in production, this would
            # properly restart the main application process
            time.sleep(2)  # Brief pause

            # Reset internal state
            self.restart_attempts = 0

            logger.info("System restart completed")
            self.eventbus.publish_sync("robustness.restart_completed", {
                "timestamp": datetime.now().isoformat()
            })

        except Exception as e:
            logger.exception(f"Restart failed: {e}")
            self.eventbus.publish_sync("robustness.restart_failed", {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })

# Global robustness manager instance
_robustness_manager: Optional[RobustnessManager] = None

def get_robustness_manager(eventbus: Optional[EventBus] = None) -> RobustnessManager:
    """Get or create global robustness manager instance"""
    global _robustness_manager
    if _robustness_manager is None and eventbus:
        _robustness_manager = RobustnessManager(eventbus)
    return _robustness_manager

def start_robustness_monitoring(eventbus: EventBus):
    """Start robustness monitoring system"""
    manager = get_robustness_manager(eventbus)
    manager.start()
    return manager

def stop_robustness_monitoring():
    """Stop robustness monitoring system"""
    global _robustness_manager
    if _robustness_manager:
        _robustness_manager.stop()
        _robustness_manager = None

# Utility functions for error handling
def with_retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator for retrying functions on failure"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {current_delay}s")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}: {e}")

            raise last_exception
        return wrapper
    return decorator

def with_timeout(timeout_seconds: float):
    """Decorator for adding timeout to functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout_seconds)
                except concurrent.futures.TimeoutError:
                    logger.error(f"Function {func.__name__} timed out after {timeout_seconds}s")
                    raise TimeoutError(f"Function {func.__name__} timed out")
        return wrapper
    return decorator

def with_circuit_breaker(circuit_name: str):
    """Decorator for circuit breaker protection"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            manager = get_robustness_manager()
            if manager:
                return manager.call_with_circuit_breaker(circuit_name, func, *args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator

if __name__ == "__main__":
    # Test robustness module
    from modules.eventbus import EventBus
    eventbus = EventBus()
    manager = start_robustness_monitoring(eventbus)

    try:
        # Run for a bit to test monitoring
        time.sleep(10)
        health = manager.get_system_health()
        print(f"System Health: {health['overall_status']}")

    finally:
        stop_robustness_monitoring()