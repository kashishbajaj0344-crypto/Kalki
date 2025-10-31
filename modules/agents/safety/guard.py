"""
Kalki Safety Module - Phase 18 Enhanced
Safety constraints and enforcement mechanisms with advanced features.
"""

import asyncio
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Pattern, Union
from enum import Enum
import yaml

# Memory integration imports (with fallbacks)
try:
    from modules.memory.episodic_memory import EpisodicMemory
    from modules.memory.semantic_memory import SemanticMemory
except ImportError:
    class EpisodicMemory:
        def add_event(self, event): pass

    class SemanticMemory:
        def store(self, key, value): pass
        def retrieve(self, key): return None

logger = logging.getLogger("Kalki.Safety")


class ConstraintType(Enum):
    """Type of safety constraint."""
    RATE_LIMIT = "rate_limit"
    FORBIDDEN_OPERATION = "forbidden_operation"
    CONTENT_FILTER = "content_filter"
    RESOURCE_LIMIT = "resource_limit"


@dataclass
class ConstraintViolation:
    """Represents a constraint violation with severity levels."""
    constraint_type: ConstraintType
    message: str
    severity: str = "hard"  # "soft" or "hard"
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SafetyCheckResult:
    """Result of a safety check."""
    allowed: bool
    violations: List[ConstraintViolation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RateLimiter:
    """Enforces rate limiting constraints with adaptive tuning."""

    def __init__(self, max_calls: int, time_window: int, adaptive: bool = True):
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum number of calls allowed
            time_window: Time window in seconds
            adaptive: Whether to enable adaptive rate limiting
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.adaptive = adaptive
        self._call_history: Dict[str, List[datetime]] = {}
        self._lock = threading.Lock()

    def check(self, identifier: str) -> bool:
        """
        Check if identifier is within rate limit.

        Args:
            identifier: Unique identifier (e.g., agent_id, user_id)

        Returns:
            True if within limit, False otherwise
        """
        with self._lock:
            now = datetime.now()
            cutoff = now - timedelta(seconds=self.time_window)

            # Clean old entries
            if identifier in self._call_history:
                self._call_history[identifier] = [
                    t for t in self._call_history[identifier]
                    if t > cutoff
                ]
            else:
                self._call_history[identifier] = []

            # Check limit
            if len(self._call_history[identifier]) >= self.max_calls:
                return False

            # Record this call
            self._call_history[identifier].append(now)
            return True

    def get_remaining(self, identifier: str) -> int:
        """Get remaining calls in current window."""
        with self._lock:
            now = datetime.now()
            cutoff = now - timedelta(seconds=self.time_window)

            if identifier not in self._call_history:
                return self.max_calls

            recent_calls = [
                t for t in self._call_history[identifier]
                if t > cutoff
            ]

            return max(0, self.max_calls - len(recent_calls))

    def adjust_rate_limit(self, name: str, agent_id: str, success_rate: float) -> None:
        """Dynamically tune rate limits based on recent agent performance."""
        with self._lock:
            # Example: reward reliable agents
            if success_rate > 0.95:
                self.max_calls = min(self.max_calls + 1, 100)
                logger.info(f"Rate limit for {name} increased to {self.max_calls} (agent {agent_id} success rate: {success_rate:.2f})")
            elif success_rate < 0.5:
                self.max_calls = max(1, self.max_calls - 1)
                logger.info(f"Rate limit for {name} decreased to {self.max_calls} (agent {agent_id} success rate: {success_rate:.2f})")


class ContentFilter:
    """Filters content based on regex patterns with thread safety."""

    def __init__(self):
        """Initialize content filter."""
        self.forbidden_patterns: List[Pattern] = []
        self._lock = threading.Lock()

    def add_pattern(self, pattern: str, flags: int = re.IGNORECASE) -> None:
        """
        Add a forbidden pattern.

        Args:
            pattern: Regex pattern
            flags: Regex flags
        """
        with self._lock:
            self.forbidden_patterns.append(re.compile(pattern, flags))

    def check(self, content: str) -> tuple[bool, List[str]]:
        """
        Check if content contains forbidden patterns.

        Args:
            content: Content to check

        Returns:
            Tuple of (is_safe, list_of_matched_patterns)
        """
        with self._lock:
            matched_patterns = []

            for pattern in self.forbidden_patterns:
                if pattern.search(content):
                    matched_patterns.append(pattern.pattern)

            return len(matched_patterns) == 0, matched_patterns


class PolicyManager:
    """Policy abstraction layer for loading constraint rules from files."""

    def __init__(self, safety_guard: 'SafetyGuard'):
        self.safety_guard = safety_guard

    def load_from_file(self, path: Path) -> bool:
        """
        Load policies from JSON or YAML file.

        Args:
            path: Path to policy file

        Returns:
            True if loaded successfully
        """
        try:
            with open(path, 'r') as f:
                if path.suffix.lower() == '.yaml' or path.suffix.lower() == '.yml':
                    policies = yaml.safe_load(f)
                else:
                    policies = json.load(f)

            # Load forbidden operations
            for operation in policies.get('forbidden_operations', []):
                self.safety_guard.add_forbidden_operation(operation)

            # Load content patterns
            for pattern in policies.get('content_patterns', []):
                self.safety_guard.add_content_filter_pattern(pattern)

            # Load rate limits
            for name, config in policies.get('rate_limits', {}).items():
                self.safety_guard.add_rate_limit(
                    name=name,
                    max_calls=config['max_calls'],
                    time_window=config['time_window']
                )

            logger.info(f"Loaded safety policies from {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load policies from {path}: {e}")
            return False


class SafetyGuard:
    """Enhanced main safety guard that enforces all constraints with advanced features."""

    def __init__(self, memory_store=None):
        """
        Initialize safety guard.

        Args:
            memory_store: Optional memory store for logging violations
        """
        self.memory_store = memory_store
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()

        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.forbidden_operations: set[str] = set()
        self.content_filter = ContentFilter()
        self.pre_execution_checks: List[Callable] = []
        self.post_execution_validators: List[Callable] = []

        # Thread safety
        self._lock = threading.Lock()

        # Policy management
        self.policy_manager = PolicyManager(self)

        # Global enforcement state
        self._global_mode = "normal"  # "normal", "safe_lock", "maintenance"

        # Violation severity weights for scoring
        self.violation_weights = {
            ConstraintType.RATE_LIMIT: 1,
            ConstraintType.CONTENT_FILTER: 3,
            ConstraintType.FORBIDDEN_OPERATION: 5,
            ConstraintType.RESOURCE_LIMIT: 4,
        }

        logger.info("Enhanced SafetyGuard initialized with advanced features")

    def add_rate_limit(self, name: str, max_calls: int, time_window: int, adaptive: bool = True) -> None:
        """
        Add a rate limit constraint.

        Args:
            name: Name of the rate limit
            max_calls: Maximum calls allowed
            time_window: Time window in seconds
            adaptive: Whether to enable adaptive rate limiting
        """
        with self._lock:
            self.rate_limiters[name] = RateLimiter(max_calls, time_window, adaptive)

    def add_forbidden_operation(self, operation: str) -> None:
        """
        Add a forbidden operation.

        Args:
            operation: Operation identifier to forbid
        """
        with self._lock:
            self.forbidden_operations.add(operation.lower())

    def add_content_filter_pattern(self, pattern: str) -> None:
        """
        Add a content filter pattern.

        Args:
            pattern: Regex pattern to filter
        """
        self.content_filter.add_pattern(pattern)

    def set_global_mode(self, mode: str) -> None:
        """
        Set global enforcement mode.

        Args:
            mode: "normal", "safe_lock", or "maintenance"
        """
        with self._lock:
            self._global_mode = mode
            logger.warning(f"Global safety mode changed to: {mode}")

    def check_global(self, operation: str) -> bool:
        """
        Check global constraints that apply across all agents.

        Args:
            operation: Operation being performed

        Returns:
            True if globally allowed
        """
        with self._lock:
            if self._global_mode == "safe_lock":
                logger.warning(f"Operation '{operation}' blocked due to safe_lock mode")
                return False
            elif self._global_mode == "maintenance":
                # Allow only read operations during maintenance
                if operation.lower() not in ["read", "query", "get"]:
                    logger.warning(f"Operation '{operation}' blocked during maintenance mode")
                    return False
            return True

    def pre_execution_check(self, agent_id: str, operation: str,
                           context: Dict[str, Any]) -> SafetyCheckResult:
        """
        Perform pre-execution safety check with all enhancements.

        Args:
            agent_id: Agent performing the action
            operation: Operation being performed
            context: Execution context

        Returns:
            SafetyCheckResult indicating if action should be allowed
        """
        violations = []

        # Global check first
        if not self.check_global(operation):
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.FORBIDDEN_OPERATION,
                message=f"Operation blocked by global safety mode: {self._global_mode}",
                severity="hard",
                metadata={"global_mode": self._global_mode}
            ))

        # Check rate limits
        for limit_name, limiter in self.rate_limiters.items():
            if not limiter.check(agent_id):
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.RATE_LIMIT,
                    message=f"Rate limit '{limit_name}' exceeded for {agent_id}",
                    severity="hard",
                    metadata={"limit_name": limit_name, "agent_id": agent_id}
                ))

        # Check forbidden operations
        if operation.lower() in self.forbidden_operations:
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.FORBIDDEN_OPERATION,
                message=f"Operation '{operation}' is forbidden",
                severity="hard",
                metadata={"operation": operation}
            ))

        # Check content filters
        content = str(context.get("content", ""))
        is_safe, matched_patterns = self.content_filter.check(content)

        if not is_safe:
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.CONTENT_FILTER,
                message=f"Content contains forbidden patterns: {', '.join(matched_patterns)}",
                severity="hard",
                metadata={"patterns": matched_patterns}
            ))

        # Run custom pre-execution checks
        for check in self.pre_execution_checks:
            try:
                check_result = check(agent_id, operation, context)
                if not check_result:
                    violations.append(ConstraintViolation(
                        constraint_type=ConstraintType.RESOURCE_LIMIT,
                        message="Custom pre-execution check failed",
                        severity="hard",
                        metadata={"check": check.__name__}
                    ))
            except Exception as e:
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.RESOURCE_LIMIT,
                    message=f"Pre-execution check error: {str(e)}",
                    severity="soft",
                    metadata={"error": str(e)}
                ))

        # Separate soft and hard violations
        hard_violations = [v for v in violations if v.severity == "hard"]
        soft_violations = [v for v in violations if v.severity == "soft"]

        allowed = len(hard_violations) == 0

        result = SafetyCheckResult(
            allowed=allowed,
            violations=violations,
            metadata={
                "hard_violations": len(hard_violations),
                "soft_violations": len(soft_violations),
                "violation_score": self.compute_violation_score(violations)
            }
        )

        # Log violations if any
        if violations:
            self._log_violations(agent_id, operation, violations)

        # Log blocked actions
        if not allowed:
            self._log_blocked_action(agent_id, operation, hard_violations)

        return result

    async def pre_execution_check_async(self, agent_id: str, operation: str,
                                       context: Dict[str, Any]) -> SafetyCheckResult:
        """
        Async version of pre_execution_check for concurrent operations.

        Args:
            agent_id: Agent performing the action
            operation: Operation being performed
            context: Execution context

        Returns:
            SafetyCheckResult indicating if action should be allowed
        """
        # For now, just wrap the sync version - can be optimized later
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.pre_execution_check, agent_id, operation, context)

    def post_execution_validate(self, agent_id: str, operation: str,
                               result: Any) -> SafetyCheckResult:
        """
        Validate result after execution.

        Args:
            agent_id: Agent that performed the action
            operation: Operation that was performed
            result: Result of the operation

        Returns:
            SafetyCheckResult indicating if result is valid
        """
        violations = []

        # Run custom post-execution validators
        for validator in self.post_execution_validators:
            try:
                is_valid = validator(agent_id, operation, result)
                if not is_valid:
                    violations.append(ConstraintViolation(
                        constraint_type=ConstraintType.RESOURCE_LIMIT,
                        message="Post-execution validation failed",
                        severity="soft",
                        metadata={"validator": validator.__name__}
                    ))
            except Exception as e:
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.RESOURCE_LIMIT,
                    message=f"Post-execution validator error: {str(e)}",
                    severity="soft",
                    metadata={"error": str(e)}
                ))

        return SafetyCheckResult(
            allowed=len([v for v in violations if v.severity == "hard"]) == 0,
            violations=violations
        )

    def appeal(self, agent_id: str, violation: ConstraintViolation,
              reason: str) -> Dict[str, Any]:
        """
        Submit an appeal for a blocked action with enhanced logging.

        Args:
            agent_id: Agent submitting appeal
            violation: The violation being appealed
            reason: Reason for the appeal

        Returns:
            Appeal record
        """
        appeal_data = {
            "agent_id": agent_id,
            "violation_type": violation.constraint_type.value,
            "violation_message": violation.message,
            "violation_severity": violation.severity,
            "appeal_reason": reason,
            "timestamp": datetime.now().isoformat(),
            "status": "pending",
            "violation_score": self.violation_weights.get(violation.constraint_type, 1)
        }

        # Log appeal to episodic memory
        self.episodic_memory.add_event({
            'type': 'safety_appeal',
            'agent_id': agent_id,
            'violation': violation.constraint_type.value,
            'reason': reason,
            'timestamp': time.time()
        })

        # Log appeal to memory store
        if self.memory_store:
            self.memory_store.put(
                f"appeal_{agent_id}_{time.time()}",
                appeal_data,
                metadata={'type': 'safety_appeal'}
            )

        logger.info(f"[SafetyGuard] Appeal submitted by {agent_id}: {reason}")
        return appeal_data

    def review_appeals(self, evaluator: Callable[[Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
        """
        Review pending appeals using an evaluator function.

        Args:
            evaluator: Function that takes appeal data and returns approval decision

        Returns:
            List of processed appeals
        """
        processed_appeals = []

        if not self.memory_store:
            return processed_appeals

        # Query pending appeals
        appeals = self.memory_store.query(metadata={'type': 'safety_appeal', 'status': 'pending'})

        for key, appeal in appeals.items():
            try:
                approved = evaluator(appeal)
                appeal['status'] = 'approved' if approved else 'rejected'
                appeal['reviewed_at'] = datetime.now().isoformat()

                # Update in memory store
                self.memory_store.put(key, appeal, metadata={'type': 'safety_appeal'})

                # Log decision
                self.episodic_memory.add_event({
                    'type': 'appeal_reviewed',
                    'agent_id': appeal['agent_id'],
                    'approved': approved,
                    'violation': appeal['violation_type'],
                    'timestamp': time.time()
                })

                processed_appeals.append(appeal)

                logger.info(f"[SafetyGuard] Appeal {key} {'approved' if approved else 'rejected'}")

            except Exception as e:
                logger.error(f"Failed to review appeal {key}: {e}")

        return processed_appeals

    def compute_violation_score(self, violations: List[ConstraintViolation]) -> int:
        """
        Compute a severity score for violations.

        Args:
            violations: List of constraint violations

        Returns:
            Total severity score
        """
        return sum(self.violation_weights.get(v.constraint_type, 1) for v in violations)

    def get_violation_stats(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get violation statistics.

        Args:
            agent_id: Optional agent ID to filter by

        Returns:
            Statistics dictionary
        """
        if not self.memory_store:
            return {"error": "No memory store available"}

        # Query violations
        query_metadata = {'type': 'blocked_action'}
        if agent_id:
            query_metadata['agent_id'] = agent_id

        violations = self.memory_store.query(metadata=query_metadata)

        stats = {
            "total_violations": len(violations),
            "by_type": {},
            "by_agent": {},
            "recent_violations": []
        }

        for key, violation in violations.items():
            # Count by type
            v_type = violation.get('operation', 'unknown')
            stats["by_type"][v_type] = stats["by_type"].get(v_type, 0) + 1

            # Count by agent
            agent = violation.get('agent_id', 'unknown')
            stats["by_agent"][agent] = stats["by_agent"].get(agent, 0) + 1

            # Recent violations (last 10)
            if len(stats["recent_violations"]) < 10:
                stats["recent_violations"].append({
                    "agent_id": agent,
                    "operation": v_type,
                    "timestamp": violation.get('timestamp'),
                    "violations": violation.get('violations', [])
                })

        return stats

    def _log_violations(self, agent_id: str, operation: str,
                       violations: List[ConstraintViolation]) -> None:
        """Log violations to episodic memory."""
        # Log to episodic memory for pattern analysis
        self.episodic_memory.add_event({
            'type': 'safety_violation',
            'agent_id': agent_id,
            'operation': operation,
            'violations': [{
                'type': v.constraint_type.value,
                'message': v.message,
                'severity': v.severity
            } for v in violations],
            'violation_score': self.compute_violation_score(violations),
            'timestamp': time.time()
        })

    def _log_blocked_action(self, agent_id: str, operation: str,
                           violations: List[ConstraintViolation]) -> None:
        """Log blocked action to memory store and logger."""
        # Standardized logging
        violation_messages = [v.message for v in violations]
        logger.warning(
            f"[SafetyGuard] {agent_id} blocked: {', '.join(violation_messages)}"
        )

        if not self.memory_store:
            return

        log_data = {
            "agent_id": agent_id,
            "operation": operation,
            "violations": [
                {
                    "type": v.constraint_type.value,
                    "message": v.message,
                    "severity": v.severity,
                    "metadata": v.metadata
                }
                for v in violations
            ],
            "violation_score": self.compute_violation_score(violations),
            "timestamp": datetime.now().isoformat()
        }

        self.memory_store.put(
            f"blocked_{agent_id}_{time.time()}",
            log_data,
            metadata={'type': 'blocked_action', 'agent_id': agent_id}
        )

# [Kalki v2.3 — agents/safety/guard.py v1.0]