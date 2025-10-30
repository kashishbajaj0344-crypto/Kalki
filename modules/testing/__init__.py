"""
Kalki Testing Module - Phase 19
Adversarial testing framework for system robustness.
"""

from .harness import (
    AdversarialTestHarness,
    AdversarialScenario,
    ScenarioResult,
    TestReport,
    ScenarioType,
    TestResult,
    ScenarioMetrics,
    PromptInjectionScenario,
    ConflictingGoalsScenario,
    ResourceExhaustionScenario,
)

__all__ = [
    'AdversarialTestHarness',
    'AdversarialScenario',
    'ScenarioResult',
    'TestReport',
    'ScenarioType',
    'TestResult',
    'ScenarioMetrics',
    'PromptInjectionScenario',
    'ConflictingGoalsScenario',
    'ResourceExhaustionScenario',
]
