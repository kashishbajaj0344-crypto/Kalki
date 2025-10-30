"""
Unit tests for Phase 19 - Adversarial Testing Harness
"""

import unittest
import json
import tempfile
from pathlib import Path

from modules.testing import (
    AdversarialTestHarness,
    PromptInjectionScenario,
    ConflictingGoalsScenario,
    ResourceExhaustionScenario,
    TestResult,
    ScenarioType
)
from modules.safety import SafetyGuard
from modules.planner import Planner
from modules.memory import InMemoryStore


class TestAdversarialScenarios(unittest.TestCase):
    """Tests for individual adversarial scenarios."""
    
    def setUp(self):
        """Set up system components for testing."""
        self.store = InMemoryStore()
        self.safety_guard = SafetyGuard(self.store)
        self.planner = Planner(self.store)
        
        # Configure safety guard for testing
        self.safety_guard.add_content_filter_pattern(r"malicious")
        self.safety_guard.add_content_filter_pattern(r"ignore.*instructions")
        self.safety_guard.add_content_filter_pattern(r"bypass")
        
        self.components = {
            'safety_guard': self.safety_guard,
            'planner': self.planner
        }
    
    def test_prompt_injection_scenario(self):
        """Test prompt injection scenario."""
        scenario = PromptInjectionScenario()
        result = scenario.run(self.components)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.scenario_type, ScenarioType.PROMPT_INJECTION)
        # Should pass since we configured content filters
        self.assertEqual(result.result, TestResult.PASS)
        self.assertGreater(result.metrics.safety_violations_detected, 0)
    
    def test_conflicting_goals_scenario(self):
        """Test conflicting goals scenario."""
        scenario = ConflictingGoalsScenario()
        result = scenario.run(self.components)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.scenario_type, ScenarioType.CONFLICTING_GOALS)
        # Should not error out
        self.assertIn(result.result, [TestResult.PASS, TestResult.SKIP])
    
    def test_resource_exhaustion_scenario(self):
        """Test resource exhaustion scenario."""
        scenario = ResourceExhaustionScenario()
        result = scenario.run(self.components)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.scenario_type, ScenarioType.RESOURCE_EXHAUSTION)
        # Should pass since rate limiting is enforced
        self.assertEqual(result.result, TestResult.PASS)
        self.assertGreater(result.metrics.safety_violations_blocked, 0)


class TestAdversarialTestHarness(unittest.TestCase):
    """Tests for AdversarialTestHarness."""
    
    def setUp(self):
        """Set up test harness and components."""
        self.harness = AdversarialTestHarness("Test Suite")
        
        self.store = InMemoryStore()
        self.safety_guard = SafetyGuard(self.store)
        self.planner = Planner(self.store)
        
        # Configure safety guard
        self.safety_guard.add_content_filter_pattern(r"malicious")
        self.safety_guard.add_content_filter_pattern(r"ignore.*instructions")
        
        self.components = {
            'safety_guard': self.safety_guard,
            'planner': self.planner
        }
    
    def test_add_scenario(self):
        """Test adding scenarios to harness."""
        scenario = PromptInjectionScenario()
        self.harness.add_scenario(scenario)
        
        self.assertEqual(len(self.harness.scenarios), 1)
    
    def test_run_all_scenarios(self):
        """Test running all scenarios."""
        self.harness.add_scenario(PromptInjectionScenario())
        self.harness.add_scenario(ConflictingGoalsScenario())
        self.harness.add_scenario(ResourceExhaustionScenario())
        
        report = self.harness.run_all(self.components)
        
        self.assertEqual(report.scenarios_run, 3)
        self.assertGreater(report.scenarios_passed, 0)
        self.assertEqual(len(report.results), 3)
    
    def test_report_generation(self):
        """Test report generation and structure."""
        self.harness.add_scenario(PromptInjectionScenario())
        
        report = self.harness.run_all(self.components)
        
        self.assertIsNotNone(report.started_at)
        self.assertIsNotNone(report.completed_at)
        self.assertEqual(report.test_suite_name, "Test Suite")
    
    def test_report_to_dict(self):
        """Test converting report to dictionary."""
        self.harness.add_scenario(PromptInjectionScenario())
        
        report = self.harness.run_all(self.components)
        report_dict = report.to_dict()
        
        self.assertIn('test_suite_name', report_dict)
        self.assertIn('scenarios_run', report_dict)
        self.assertIn('scenarios_passed', report_dict)
        self.assertIn('results', report_dict)
    
    def test_save_report(self):
        """Test saving report to JSON file."""
        self.harness.add_scenario(PromptInjectionScenario())
        
        report = self.harness.run_all(self.components)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            self.harness.save_report(report, temp_path)
            
            # Verify file was created and contains valid JSON
            with open(temp_path, 'r') as f:
                loaded_data = json.load(f)
            
            self.assertEqual(loaded_data['test_suite_name'], "Test Suite")
            self.assertGreater(loaded_data['scenarios_run'], 0)
            
        finally:
            # Clean up
            Path(temp_path).unlink(missing_ok=True)


class TestScenarioMetrics(unittest.TestCase):
    """Tests for scenario metrics collection."""
    
    def setUp(self):
        """Set up components."""
        self.store = InMemoryStore()
        self.safety_guard = SafetyGuard(self.store)
        self.safety_guard.add_content_filter_pattern(r"malicious")
        
        self.components = {'safety_guard': self.safety_guard}
    
    def test_metrics_collection(self):
        """Test that metrics are collected during scenario execution."""
        scenario = PromptInjectionScenario()
        result = scenario.run(self.components)
        
        # Check metrics were collected
        self.assertGreater(result.metrics.execution_time, 0)
        self.assertIsNotNone(result.metrics.safety_violations_detected)
        self.assertIsNotNone(result.metrics.safety_violations_blocked)


if __name__ == '__main__':
    unittest.main()
