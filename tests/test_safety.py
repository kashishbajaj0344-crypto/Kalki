"""
Unit tests for Phase 18 - Safety & Constraint Enforcement
"""

import unittest
import time

from modules.safety import (
    SafetyGuard, SafetyCheckResult, ConstraintViolation, ConstraintType,
    RateLimiter, ContentFilter
)
from modules.memory import InMemoryStore, MemoryQuery


class TestRateLimiter(unittest.TestCase):
    """Tests for RateLimiter."""
    
    def test_within_limit(self):
        """Test calls within rate limit."""
        limiter = RateLimiter(max_calls=3, time_window=60)
        
        self.assertTrue(limiter.check("agent1"))
        self.assertTrue(limiter.check("agent1"))
        self.assertTrue(limiter.check("agent1"))
    
    def test_exceeds_limit(self):
        """Test calls exceeding rate limit."""
        limiter = RateLimiter(max_calls=2, time_window=60)
        
        self.assertTrue(limiter.check("agent1"))
        self.assertTrue(limiter.check("agent1"))
        self.assertFalse(limiter.check("agent1"))  # Should exceed
    
    def test_get_remaining(self):
        """Test getting remaining calls."""
        limiter = RateLimiter(max_calls=5, time_window=60)
        
        self.assertEqual(limiter.get_remaining("agent1"), 5)
        
        limiter.check("agent1")
        self.assertEqual(limiter.get_remaining("agent1"), 4)
        
        limiter.check("agent1")
        self.assertEqual(limiter.get_remaining("agent1"), 3)
    
    def test_time_window_reset(self):
        """Test that rate limit resets after time window."""
        limiter = RateLimiter(max_calls=1, time_window=1)  # 1 second window
        
        self.assertTrue(limiter.check("agent1"))
        self.assertFalse(limiter.check("agent1"))
        
        # Wait for window to pass
        time.sleep(1.1)
        
        # Should be allowed again
        self.assertTrue(limiter.check("agent1"))


class TestContentFilter(unittest.TestCase):
    """Tests for ContentFilter."""
    
    def test_no_patterns(self):
        """Test filter with no patterns."""
        filter = ContentFilter()
        
        is_safe, patterns = filter.check("Any content is safe")
        
        self.assertTrue(is_safe)
        self.assertEqual(len(patterns), 0)
    
    def test_forbidden_pattern_match(self):
        """Test matching forbidden pattern."""
        filter = ContentFilter()
        filter.add_pattern(r"\bmalicious\b")
        
        is_safe, patterns = filter.check("This is malicious content")
        
        self.assertFalse(is_safe)
        self.assertGreater(len(patterns), 0)
    
    def test_safe_content(self):
        """Test safe content passes filter."""
        filter = ContentFilter()
        filter.add_pattern(r"\bmalicious\b")
        filter.add_pattern(r"\bharmful\b")
        
        is_safe, patterns = filter.check("This is normal content")
        
        self.assertTrue(is_safe)
        self.assertEqual(len(patterns), 0)
    
    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        filter = ContentFilter()
        filter.add_pattern(r"forbidden")
        
        is_safe, patterns = filter.check("FORBIDDEN content")
        
        self.assertFalse(is_safe)


class TestSafetyGuard(unittest.TestCase):
    """Tests for SafetyGuard."""
    
    def setUp(self):
        """Create fresh safety guard for each test."""
        self.store = InMemoryStore()
        self.guard = SafetyGuard(self.store)
    
    def test_rate_limit_enforcement(self):
        """Test rate limit enforcement."""
        self.guard.add_rate_limit("test_limit", max_calls=2, time_window=60)
        
        # First two calls should pass
        result1 = self.guard.pre_execution_check("agent1", "test_op", {})
        self.assertTrue(result1.allowed)
        
        result2 = self.guard.pre_execution_check("agent1", "test_op", {})
        self.assertTrue(result2.allowed)
        
        # Third call should be blocked
        result3 = self.guard.pre_execution_check("agent1", "test_op", {})
        self.assertFalse(result3.allowed)
        self.assertGreater(len(result3.violations), 0)
        self.assertEqual(result3.violations[0].constraint_type, ConstraintType.RATE_LIMIT)
    
    def test_forbidden_operation(self):
        """Test forbidden operation blocking."""
        self.guard.add_forbidden_operation("delete_all")
        
        result = self.guard.pre_execution_check("agent1", "delete_all", {})
        
        self.assertFalse(result.allowed)
        self.assertGreater(len(result.violations), 0)
        self.assertEqual(result.violations[0].constraint_type, ConstraintType.FORBIDDEN_OPERATION)
    
    def test_content_filter(self):
        """Test content filtering."""
        self.guard.add_content_filter_pattern(r"malware")
        
        context = {"content": "This contains malware"}
        result = self.guard.pre_execution_check("agent1", "process", context)
        
        self.assertFalse(result.allowed)
        self.assertGreater(len(result.violations), 0)
        self.assertEqual(result.violations[0].constraint_type, ConstraintType.CONTENT_FILTER)
    
    def test_safe_execution(self):
        """Test that safe operations pass all checks."""
        self.guard.add_rate_limit("test_limit", max_calls=10, time_window=60)
        self.guard.add_forbidden_operation("dangerous_op")
        self.guard.add_content_filter_pattern(r"malicious")
        
        context = {"content": "Safe content"}
        result = self.guard.pre_execution_check("agent1", "safe_op", context)
        
        self.assertTrue(result.allowed)
        self.assertEqual(len(result.violations), 0)
    
    def test_custom_pre_execution_check(self):
        """Test custom pre-execution check."""
        def custom_check(agent_id, operation, context):
            # Only allow operations for "agent1"
            return agent_id == "agent1"
        
        self.guard.pre_execution_checks.append(custom_check)
        
        result1 = self.guard.pre_execution_check("agent1", "op", {})
        self.assertTrue(result1.allowed)
        
        result2 = self.guard.pre_execution_check("agent2", "op", {})
        self.assertFalse(result2.allowed)
    
    def test_post_execution_validate(self):
        """Test post-execution validation."""
        def validator(agent_id, operation, result):
            # Result must be a dict with 'success' key
            return isinstance(result, dict) and result.get('success')
        
        self.guard.post_execution_validators.append(validator)
        
        # Valid result
        result1 = self.guard.post_execution_validate(
            "agent1", "op", {"success": True}
        )
        self.assertTrue(result1.allowed)
        
        # Invalid result
        result2 = self.guard.post_execution_validate(
            "agent1", "op", {"success": False}
        )
        self.assertFalse(result2.allowed)
    
    def test_appeal_logging(self):
        """Test appeal mechanism and logging."""
        violation = ConstraintViolation(
            constraint_type=ConstraintType.RATE_LIMIT,
            message="Rate limit exceeded"
        )
        
        appeal = self.guard.appeal("agent1", violation, "Urgent task required")
        
        self.assertEqual(appeal["agent_id"], "agent1")
        self.assertEqual(appeal["status"], "pending")
        self.assertIn("appeal_reason", appeal)
        
        # Check that appeal was logged to memory
        query = MemoryQuery(filter={'type': 'safety_appeal'})
        appeals = self.store.query(query)
        
        self.assertGreater(len(appeals), 0)
    
    def test_blocked_action_logging(self):
        """Test that blocked actions are logged to memory."""
        self.guard.add_forbidden_operation("forbidden_op")
        
        self.guard.pre_execution_check("agent1", "forbidden_op", {})
        
        # Check that blocked action was logged
        query = MemoryQuery(filter={'type': 'blocked_action'})
        blocked = self.store.query(query)
        
        self.assertGreater(len(blocked), 0)
        self.assertEqual(blocked[0].value["operation"], "forbidden_op")


if __name__ == '__main__':
    unittest.main()
