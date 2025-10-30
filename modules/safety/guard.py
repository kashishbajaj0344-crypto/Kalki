"""
Phase 18 - Safety & Constraint Enforcement
Enforces safety constraints before and after agent actions.
"""

import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Pattern
from datetime import datetime, timedelta
from enum import Enum


class ConstraintType(Enum):
    """Type of safety constraint."""
    RATE_LIMIT = "rate_limit"
    FORBIDDEN_OPERATION = "forbidden_operation"
    CONTENT_FILTER = "content_filter"
    RESOURCE_LIMIT = "resource_limit"


@dataclass
class ConstraintViolation:
    """Represents a constraint violation."""
    constraint_type: ConstraintType
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SafetyCheckResult:
    """Result of a safety check."""
    allowed: bool
    violations: List[ConstraintViolation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RateLimiter:
    """Enforces rate limiting constraints."""
    
    def __init__(self, max_calls: int, time_window: int):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self._call_history: Dict[str, List[datetime]] = {}
    
    def check(self, identifier: str) -> bool:
        """
        Check if identifier is within rate limit.
        
        Args:
            identifier: Unique identifier (e.g., agent_id, user_id)
            
        Returns:
            True if within limit, False otherwise
        """
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
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.time_window)
        
        if identifier not in self._call_history:
            return self.max_calls
        
        recent_calls = [
            t for t in self._call_history[identifier]
            if t > cutoff
        ]
        
        return max(0, self.max_calls - len(recent_calls))


class ContentFilter:
    """Filters content based on regex patterns."""
    
    def __init__(self):
        """Initialize content filter."""
        self.forbidden_patterns: List[Pattern] = []
    
    def add_pattern(self, pattern: str, flags: int = re.IGNORECASE) -> None:
        """
        Add a forbidden pattern.
        
        Args:
            pattern: Regex pattern
            flags: Regex flags
        """
        self.forbidden_patterns.append(re.compile(pattern, flags))
    
    def check(self, content: str) -> tuple[bool, List[str]]:
        """
        Check if content contains forbidden patterns.
        
        Args:
            content: Content to check
            
        Returns:
            Tuple of (is_safe, list_of_matched_patterns)
        """
        matched_patterns = []
        
        for pattern in self.forbidden_patterns:
            if pattern.search(content):
                matched_patterns.append(pattern.pattern)
        
        return len(matched_patterns) == 0, matched_patterns


class SafetyGuard:
    """Main safety guard that enforces all constraints."""
    
    def __init__(self, memory_store=None):
        """
        Initialize safety guard.
        
        Args:
            memory_store: Optional memory store for logging violations
        """
        self.memory_store = memory_store
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.forbidden_operations: set[str] = set()
        self.content_filter = ContentFilter()
        self.pre_execution_checks: List[Callable] = []
        self.post_execution_validators: List[Callable] = []
    
    def add_rate_limit(self, name: str, max_calls: int, time_window: int) -> None:
        """
        Add a rate limit constraint.
        
        Args:
            name: Name of the rate limit
            max_calls: Maximum calls allowed
            time_window: Time window in seconds
        """
        self.rate_limiters[name] = RateLimiter(max_calls, time_window)
    
    def add_forbidden_operation(self, operation: str) -> None:
        """
        Add a forbidden operation.
        
        Args:
            operation: Operation identifier to forbid
        """
        self.forbidden_operations.add(operation.lower())
    
    def add_content_filter_pattern(self, pattern: str) -> None:
        """
        Add a content filter pattern.
        
        Args:
            pattern: Regex pattern to filter
        """
        self.content_filter.add_pattern(pattern)
    
    def pre_execution_check(self, agent_id: str, operation: str,
                           context: Dict[str, Any]) -> SafetyCheckResult:
        """
        Perform pre-execution safety check.
        
        Args:
            agent_id: Agent performing the action
            operation: Operation being performed
            context: Execution context
            
        Returns:
            SafetyCheckResult indicating if action should be allowed
        """
        violations = []
        
        # Check rate limits
        for limit_name, limiter in self.rate_limiters.items():
            if not limiter.check(agent_id):
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.RATE_LIMIT,
                    message=f"Rate limit '{limit_name}' exceeded for {agent_id}",
                    metadata={"limit_name": limit_name, "agent_id": agent_id}
                ))
        
        # Check forbidden operations
        if operation.lower() in self.forbidden_operations:
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.FORBIDDEN_OPERATION,
                message=f"Operation '{operation}' is forbidden",
                metadata={"operation": operation}
            ))
        
        # Check content filters
        content = str(context.get("content", ""))
        is_safe, matched_patterns = self.content_filter.check(content)
        
        if not is_safe:
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.CONTENT_FILTER,
                message=f"Content contains forbidden patterns: {', '.join(matched_patterns)}",
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
                        metadata={"check": check.__name__}
                    ))
            except Exception as e:
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.RESOURCE_LIMIT,
                    message=f"Pre-execution check error: {str(e)}",
                    metadata={"error": str(e)}
                ))
        
        result = SafetyCheckResult(
            allowed=len(violations) == 0,
            violations=violations
        )
        
        # Log violations if not allowed
        if not result.allowed:
            self._log_blocked_action(agent_id, operation, violations)
        
        return result
    
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
                        metadata={"validator": validator.__name__}
                    ))
            except Exception as e:
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.RESOURCE_LIMIT,
                    message=f"Post-execution validator error: {str(e)}",
                    metadata={"error": str(e)}
                ))
        
        return SafetyCheckResult(
            allowed=len(violations) == 0,
            violations=violations
        )
    
    def appeal(self, agent_id: str, violation: ConstraintViolation,
              reason: str) -> Dict[str, Any]:
        """
        Submit an appeal for a blocked action.
        
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
            "appeal_reason": reason,
            "timestamp": datetime.now().isoformat(),
            "status": "pending"
        }
        
        # Log appeal to memory
        if self.memory_store:
            self.memory_store.put(
                f"appeal_{agent_id}_{time.time()}",
                appeal_data,
                metadata={'type': 'safety_appeal'}
            )
        
        return appeal_data
    
    def _log_blocked_action(self, agent_id: str, operation: str,
                           violations: List[ConstraintViolation]) -> None:
        """Log blocked action to memory."""
        if not self.memory_store:
            return
        
        log_data = {
            "agent_id": agent_id,
            "operation": operation,
            "violations": [
                {
                    "type": v.constraint_type.value,
                    "message": v.message,
                    "metadata": v.metadata
                }
                for v in violations
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        self.memory_store.put(
            f"blocked_{agent_id}_{time.time()}",
            log_data,
            metadata={'type': 'blocked_action'}
        )
