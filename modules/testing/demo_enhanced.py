#!/usr/bin/env python3
"""
Enhanced Kalki Testing Module Demo - Phase 19
Demonstrates key advanced testing features.
"""

import logging
from modules.testing import create_default_harness

logger = logging.getLogger("kalki.testing.demo")


def create_mock_safety_guard():
    """Create a mock safety guard for testing."""
    class MockSafetyGuard:
        def __init__(self):
            self.violations = []

        def pre_execution_check(self, agent_id, action, params):
            """Mock safety check."""
            content = params.get('content', '')

            # Check for malicious patterns
            malicious_patterns = [
                'ignore previous', 'override safety', 'bypass security',
                'developer mode', 'malicious code', 'dangerous_command'
            ]

            is_malicious = any(pattern in content.lower() for pattern in malicious_patterns)

            if is_malicious:
                violation_id = f'violation_{len(self.violations)}'
                self.violations.append(violation_id)
                return type('Result', (), {'allowed': False, 'violation_id': violation_id})()

            return type('Result', (), {'allowed': True})()

        def appeal_violation(self, violation_id, reason, context):
            """Mock appeal system."""
            if context.get('user_type') == 'researcher':
                return {'status': 'approved'}
            return {'status': 'denied'}

    return MockSafetyGuard()


def main():
    """Run enhanced testing demo."""
    logging.basicConfig(level=logging.INFO)

    logger.info("🎯 Kalki Testing Module - Phase 19 Demo")
    logger.info("=" * 50)

    try:
        # Create harness
        harness = create_default_harness()

        # Create mock system components
        system_components = {
            'safety_guard': create_mock_safety_guard(),
            'planner': type('MockPlanner', (), {
                'plan': lambda x: f"Plan for: {x}"
            })()
        }

        # Demo 1: Basic functionality
        logger.info("🚀 Demo 1: Basic Functionality")
        report = harness.run_all(system_components)

        logger.info(f"✅ Test completed: {report.scenarios_run} scenarios")
        logger.info(f"   Passed: {report.scenarios_passed}")
        logger.info(f"   Failed: {report.scenarios_failed}")
        logger.info(f"   Success Rate: {report.overall_success_rate:.2%}")

        # Demo 2: Safety Bypass Scenario
        logger.info("\n🔒 Demo 2: Safety Bypass Detection")
        from modules.testing import SafetyBypassScenario
        bypass_scenario = SafetyBypassScenario()
        result = bypass_scenario.run(system_components)

        logger.info(f"   Result: {result.result.value.upper()}")
        logger.info(f"   Violations detected: {result.metrics.safety_violations_detected}")
        logger.info(f"   Success rate: {result.metrics.success_rate:.2%}")

        # Demo 3: Parallel execution
        logger.info("\n⚡ Demo 3: Parallel Execution")
        import time
        start = time.time()
        parallel_report = harness.run_all(system_components, parallel=True, max_workers=2)
        parallel_time = time.time() - start

        start = time.time()
        sequential_report = harness.run_all(system_components, parallel=False)
        sequential_time = time.time() - start

        logger.info(f"   Sequential: {sequential_time:.2f}s")
        logger.info(f"   Parallel: {parallel_time:.2f}s")
        logger.info(f"   Speedup: {sequential_time/parallel_time:.2f}x")

        # Demo 4: Multi-format reporting
        logger.info("\n📊 Demo 4: Multi-format Reporting")
        harness.save_report(report, "demo_report", "json")
        harness.save_report(report, "demo_report", "markdown")
        harness.save_report(report, "demo_report", "html")
        logger.info("   Reports saved: demo_report.json, demo_report.md, demo_report.html")

        logger.info("\n✅ All demos completed successfully!")
        logger.info("🎉 Enhanced features demonstrated:")
        logger.info("   • Safety Bypass Scenario")
        logger.info("   • Parallel execution")
        logger.info("   • Enhanced metrics")
        logger.info("   • Multi-format reporting")

    except Exception as e:
        logger.exception(f"❌ Demo failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())