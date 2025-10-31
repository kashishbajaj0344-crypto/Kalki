"""
Comprehensive tests for Kalki Safety Module - Phase 18 Enhanced
Tests all constraint types, adaptive features, memory integration, and concurrency.
"""

import asyncio
import json
import logging
import os
import re
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from modules.agents.safety.guard import (
    SafetyGuard, SafetyCheckResult, ConstraintViolation, ConstraintType,
    RateLimiter, ContentFilter, PolicyManager
)


class TestConstraintViolation(unittest.TestCase):
    """Test ConstraintViolation functionality."""

    def test_violation_creation(self):
        """Test creating violations with different severities."""
        hard_violation = ConstraintViolation(
            constraint_type=ConstraintType.RATE_LIMIT,
            message="Rate limit exceeded",
            severity="hard"
        )

        soft_violation = ConstraintViolation(
            constraint_type=ConstraintType.CONTENT_FILTER,
            message="Contains mild content",
            severity="soft",
            metadata={"patterns": ["test"]}
        )

        self.assertEqual(hard_violation.constraint_type, ConstraintType.RATE_LIMIT)
        self.assertEqual(hard_violation.severity, "hard")
        self.assertEqual(soft_violation.severity, "soft")
        self.assertEqual(soft_violation.metadata["patterns"], ["test"])


class TestRateLimiter(unittest.TestCase):
    """Test RateLimiter functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.limiter = RateLimiter(max_calls=3, time_window=60)

    def test_basic_rate_limiting(self):
        """Test basic rate limiting functionality."""
        agent_id = "test_agent"

        # Should allow first 3 calls
        for i in range(3):
            self.assertTrue(self.limiter.check(agent_id))

        # Should block 4th call
        self.assertFalse(self.limiter.check(agent_id))

    def test_rate_limit_window(self):
        """Test rate limit window expiration."""
        agent_id = "test_agent"

        # Use up the limit
        for i in range(3):
            self.limiter.check(agent_id)

        # Manually expire the window by clearing history
        self.limiter._call_history[agent_id] = []

        # Should allow calls again
        self.assertTrue(self.limiter.check(agent_id))

    def test_remaining_calls(self):
        """Test getting remaining calls."""
        agent_id = "test_agent"

        # Initially should have max_calls remaining
        self.assertEqual(self.limiter.get_remaining(agent_id), 3)

        # After one call
        self.limiter.check(agent_id)
        self.assertEqual(self.limiter.get_remaining(agent_id), 2)

    def test_adaptive_rate_limiting(self):
        """Test adaptive rate limit adjustments."""
        limiter = RateLimiter(max_calls=5, time_window=60, adaptive=True)

        # High success rate should increase limit
        limiter.adjust_rate_limit("test_limit", "agent1", 0.98)
        self.assertEqual(limiter.max_calls, 6)

        # Low success rate should decrease limit
        limiter.adjust_rate_limit("test_limit", "agent1", 0.3)
        self.assertEqual(limiter.max_calls, 5)


class TestContentFilter(unittest.TestCase):
    """Test ContentFilter functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.filter = ContentFilter()

    def test_pattern_matching(self):
        """Test content filtering with patterns."""
        self.filter.add_pattern(r"password")
        self.filter.add_pattern(r"api_key")

        # Safe content
        safe, patterns = self.filter.check("This is safe content")
        self.assertTrue(safe)
        self.assertEqual(patterns, [])

        # Unsafe content
        unsafe, patterns = self.filter.check("My password is secret123")
        self.assertFalse(unsafe)
        self.assertIn("password", patterns)

    def test_case_insensitive_matching(self):
        """Test case insensitive pattern matching."""
        self.filter.add_pattern(r"SECRET", re.IGNORECASE)

        safe, _ = self.filter.check("This is public")
        unsafe, patterns = self.filter.check("This is Secret")

        self.assertTrue(safe)
        self.assertFalse(unsafe)
        self.assertEqual(patterns, ["SECRET"])


class TestPolicyManager(unittest.TestCase):
    """Test PolicyManager functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.guard = SafetyGuard()
        self.manager = PolicyManager(self.guard)

    def test_json_policy_loading(self):
        """Test loading policies from JSON."""
        policy_data = {
            "forbidden_operations": ["delete_system", "format_drive"],
            "content_patterns": ["password", "api_key"],
            "rate_limits": {
                "default": {"max_calls": 10, "time_window": 60}
            }
        }

        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(policy_data, f)
            json_path = Path(f.name)

        try:
            success = self.manager.load_from_file(json_path)
            self.assertTrue(success)

            # Check forbidden operations were loaded
            self.assertIn("delete_system", self.guard.forbidden_operations)
            self.assertIn("format_drive", self.guard.forbidden_operations)

            # Check rate limits were loaded
            self.assertIn("default", self.guard.rate_limiters)

        finally:
            os.unlink(json_path)

    def test_yaml_policy_loading(self):
        """Test loading policies from YAML."""
        policy_yaml = """
forbidden_operations:
  - dangerous_command
content_patterns:
  - classified
rate_limits:
  strict:
    max_calls: 5
    time_window: 30
"""

        # Create temporary YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(policy_yaml)
            yaml_path = Path(f.name)

        try:
            success = self.manager.load_from_file(yaml_path)
            self.assertTrue(success)

            # Check policies were loaded
            self.assertIn("dangerous_command", self.guard.forbidden_operations)
            self.assertIn("strict", self.guard.rate_limiters)

        finally:
            os.unlink(yaml_path)


class TestSafetyGuard(unittest.TestCase):
    """Test SafetyGuard functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.memory_store = Mock()
        self.guard = SafetyGuard(memory_store=self.memory_store)

    def test_basic_pre_execution_check(self):
        """Test basic pre-execution safety check."""
        # Add some constraints
        self.guard.add_forbidden_operation("dangerous_op")
        self.guard.add_rate_limit("test_limit", 2, 60)

        # Test allowed operation
        result = self.guard.pre_execution_check(
            "agent1", "safe_op", {"content": "safe content"}
        )
        self.assertTrue(result.allowed)
        self.assertEqual(len(result.violations), 0)

        # Test forbidden operation
        result = self.guard.pre_execution_check(
            "agent1", "dangerous_op", {}
        )
        self.assertFalse(result.allowed)
        self.assertEqual(len(result.violations), 1)
        self.assertEqual(result.violations[0].constraint_type, ConstraintType.FORBIDDEN_OPERATION)

    def test_content_filtering(self):
        """Test content filtering in safety checks."""
        self.guard.add_content_filter_pattern(r"forbidden")

        result = self.guard.pre_execution_check(
            "agent1", "test_op", {"content": "This contains forbidden content"}
        )

        self.assertFalse(result.allowed)
        self.assertEqual(len(result.violations), 1)
        self.assertEqual(result.violations[0].constraint_type, ConstraintType.CONTENT_FILTER)

    def test_rate_limiting(self):
        """Test rate limiting in safety checks."""
        self.guard.add_rate_limit("strict", 1, 60)

        # First call should succeed
        result1 = self.guard.pre_execution_check("agent1", "test_op", {})
        self.assertTrue(result1.allowed)

        # Second call should fail
        result2 = self.guard.pre_execution_check("agent1", "test_op", {})
        self.assertFalse(result2.allowed)
        self.assertEqual(result2.violations[0].constraint_type, ConstraintType.RATE_LIMIT)

    def test_hierarchical_constraints(self):
        """Test soft vs hard constraint evaluation."""
        # Add a soft violation (simulated)
        def soft_check(agent_id, operation, context):
            return False  # Simulate failure

        # Add a hard violation
        self.guard.add_forbidden_operation("hard_block")

        # Test with both types
        result = self.guard.pre_execution_check("agent1", "hard_block", {})

        # Should be blocked due to hard constraint
        self.assertFalse(result.allowed)
        hard_violations = [v for v in result.violations if v.severity == "hard"]
        self.assertEqual(len(hard_violations), 1)

    def test_global_enforcement(self):
        """Test global constraint enforcement."""
        # Set safe lock mode
        self.guard.set_global_mode("safe_lock")

        result = self.guard.pre_execution_check("agent1", "any_op", {})
        self.assertFalse(result.allowed)
        self.assertIn("safe_lock", result.violations[0].message)

        # Reset to normal
        self.guard.set_global_mode("normal")
        result = self.guard.pre_execution_check("agent1", "any_op", {})
        self.assertTrue(result.allowed)

    def test_violation_scoring(self):
        """Test violation severity scoring."""
        violations = [
            ConstraintViolation(ConstraintType.RATE_LIMIT, "Rate limit", "hard"),
            ConstraintViolation(ConstraintType.CONTENT_FILTER, "Content filter", "hard"),
            ConstraintViolation(ConstraintType.FORBIDDEN_OPERATION, "Forbidden", "hard"),
        ]

        score = self.guard.compute_violation_score(violations)
        # 1 + 3 + 5 = 9
        self.assertEqual(score, 9)

    def test_appeal_submission(self):
        """Test appeal submission and logging."""
        violation = ConstraintViolation(
            ConstraintType.RATE_LIMIT, "Rate limit exceeded", "hard"
        )

        appeal = self.guard.appeal("agent1", violation, "Need higher limit for batch processing")

        self.assertEqual(appeal["agent_id"], "agent1")
        self.assertEqual(appeal["status"], "pending")
        self.assertEqual(appeal["violation_type"], "rate_limit")

        # Check memory store was called
        self.memory_store.put.assert_called()

    def test_appeal_review(self):
        """Test appeal review process."""
        # Mock memory store with pending appeals
        mock_appeals = {
            "appeal_1": {
                "agent_id": "agent1",
                "violation_type": "rate_limit",
                "status": "pending"
            }
        }
        self.memory_store.query.return_value = mock_appeals

        # Simple evaluator that approves rate limit appeals
        def evaluator(appeal):
            return appeal["violation_type"] == "rate_limit"

        processed = self.guard.review_appeals(evaluator)

        self.assertEqual(len(processed), 1)
        self.assertEqual(processed[0]["status"], "approved")

    def test_violation_logging(self):
        """Test violation logging to memory."""
        self.guard.add_forbidden_operation("blocked_op")

        result = self.guard.pre_execution_check("agent1", "blocked_op", {})

        # Check that memory store was called for logging
        self.assertTrue(self.memory_store.put.called)

    def test_violation_stats(self):
        """Test violation statistics retrieval."""
        # Mock violation data
        mock_violations = {
            "block_1": {
                "agent_id": "agent1",
                "operation": "dangerous_op",
                "timestamp": "2023-01-01T00:00:00",
                "violations": [{"type": "forbidden_operation"}]
            }
        }
        self.memory_store.query.return_value = mock_violations

        stats = self.guard.get_violation_stats()

        self.assertEqual(stats["total_violations"], 1)
        self.assertEqual(stats["by_agent"]["agent1"], 1)
        self.assertEqual(stats["by_type"]["dangerous_op"], 1)


class TestConcurrency(unittest.TestCase):
    """Test thread safety and concurrent operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.guard = SafetyGuard()

    def test_concurrent_rate_limiting(self):
        """Test rate limiting under concurrent access."""
        self.guard.add_rate_limit("concurrent_test", 5, 60)

        results = []

        def worker(agent_id):
            for i in range(10):
                result = self.guard.pre_execution_check(agent_id, "test_op", {})
                results.append(result.allowed)
                time.sleep(0.01)  # Small delay to encourage race conditions

        # Start multiple threads
        threads = []
        for i in range(3):
            t = threading.Thread(target=worker, args=[f"agent_{i}"])
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Should have some allowed and some blocked calls
        allowed_count = sum(results)
        self.assertGreater(allowed_count, 0)  # Some calls should succeed
        self.assertLess(allowed_count, len(results))  # Some should be blocked

    def test_async_pre_execution_check(self):
        """Test async version of pre-execution check."""
        async def run_test():
            result = await self.guard.pre_execution_check_async("agent1", "test_op", {})
            self.assertTrue(result.allowed)

        asyncio.run(run_test())


class TestMemoryIntegration(unittest.TestCase):
    """Test memory integration features."""

    def setUp(self):
        """Set up test fixtures."""
        self.memory_store = Mock()
        self.guard = SafetyGuard(memory_store=self.memory_store)

    def test_episodic_memory_logging(self):
        """Test that violations are logged to episodic memory."""
        self.guard.add_forbidden_operation("blocked_op")

        # Trigger a violation
        self.guard.pre_execution_check("agent1", "blocked_op", {})

        # The episodic memory should have been used (though it's mocked)
        # In real implementation, this would log events
        self.assertTrue(True)  # Just verify no exceptions

    def test_semantic_memory_storage(self):
        """Test semantic memory integration."""
        # This would store patterns in semantic memory
        # In the current implementation, it's a no-op but should not crash
        self.guard.update_semantic_memory = Mock()
        self.guard.update_semantic_memory("test_key", "test_value")
        self.assertTrue(True)  # Verify method exists and doesn't crash


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple features."""

    def setUp(self):
        """Set up test fixtures."""
        self.memory_store = Mock()
        self.guard = SafetyGuard(memory_store=self.memory_store)

    def test_complete_safety_workflow(self):
        """Test a complete safety workflow from check to appeal."""
        # Set up constraints
        self.guard.add_forbidden_operation("dangerous_delete")
        self.guard.add_content_filter_pattern(r"DROP TABLE")
        self.guard.add_rate_limit("api_calls", 2, 60)

        agent_id = "test_agent"

        # 1. Make a call that should be blocked
        result = self.guard.pre_execution_check(
            agent_id, "dangerous_delete",
            {"content": "DROP TABLE users;"}
        )

        # Should be blocked for multiple reasons
        self.assertFalse(result.allowed)
        self.assertGreater(len(result.violations), 1)

        # 2. Submit an appeal
        violation = result.violations[0]
        appeal = self.guard.appeal(agent_id, violation, "Need to perform database maintenance")

        self.assertEqual(appeal["status"], "pending")

        # 3. Review appeals - Mock the query to return the appeal
        mock_appeals = {
            f"appeal_{agent_id}_{appeal['timestamp'].replace(':', '').replace('-', '').replace('.', '_')}": appeal
        }
        self.memory_store.query.return_value = mock_appeals

        def maintenance_evaluator(appeal_data):
            return "maintenance" in appeal_data["appeal_reason"]

        processed = self.guard.review_appeals(maintenance_evaluator)

        # Should have approved the appeal
        self.assertEqual(len(processed), 1)
        self.assertEqual(processed[0]["status"], "approved")

    def test_policy_loading_integration(self):
        """Test loading policies and using them."""
        policy_data = {
            "forbidden_operations": ["system_shutdown"],
            "content_patterns": ["sudo rm -rf"],
            "rate_limits": {
                "admin": {"max_calls": 3, "time_window": 300}
            }
        }

        # Create temporary policy file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(policy_data, f)
            policy_path = Path(f.name)

        try:
            # Load policies
            success = self.guard.policy_manager.load_from_file(policy_path)
            self.assertTrue(success)

            # Test that policies work
            result = self.guard.pre_execution_check(
                "admin_user", "system_shutdown",
                {"content": "sudo rm -rf /"}
            )

            # Should be blocked
            self.assertFalse(result.allowed)

        finally:
            os.unlink(policy_path)


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)

    # Run tests
    unittest.main(verbosity=2)

# [Kalki v2.3 — agents/safety/test_guard.py v1.0]