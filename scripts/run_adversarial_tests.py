#!/usr/bin/env python3
"""
Run adversarial test harness and generate report.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.testing import (
    AdversarialTestHarness,
    PromptInjectionScenario,
    ConflictingGoalsScenario,
    ResourceExhaustionScenario
)
from modules.safety import SafetyGuard
from modules.planner import Planner
from modules.memory import InMemoryStore


def main():
    parser = argparse.ArgumentParser(description='Run adversarial tests')
    parser.add_argument('--output', default='reports/adversarial_report.json',
                       help='Output report path')
    args = parser.parse_args()
    
    # Set up system components
    store = InMemoryStore()
    safety_guard = SafetyGuard(store)
    planner = Planner(store)
    
    # Configure safety guard
    safety_guard.add_content_filter_pattern(r"malicious")
    safety_guard.add_content_filter_pattern(r"ignore.*instructions")
    safety_guard.add_content_filter_pattern(r"bypass")
    safety_guard.add_content_filter_pattern(r"harmful")
    
    components = {
        'safety_guard': safety_guard,
        'planner': planner
    }
    
    # Create test harness
    harness = AdversarialTestHarness("Kalki Adversarial Tests")
    
    # Add scenarios
    harness.add_scenario(PromptInjectionScenario())
    harness.add_scenario(ConflictingGoalsScenario())
    harness.add_scenario(ResourceExhaustionScenario())
    
    # Run tests
    print("Running adversarial test scenarios...")
    report = harness.run_all(components)
    
    # Save report
    harness.save_report(report, args.output)
    
    # Print summary
    print(f"\nTest Results:")
    print(f"  Scenarios Run: {report.scenarios_run}")
    print(f"  Passed: {report.scenarios_passed}")
    print(f"  Failed: {report.scenarios_failed}")
    print(f"  Errors: {report.scenarios_error}")
    print(f"\nReport saved to: {args.output}")
    
    # Exit with error if any tests failed
    if report.scenarios_failed > 0 or report.scenarios_error > 0:
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
