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
    SafetyBypassScenario,
    create_default_harness,
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
    'SafetyBypassScenario',
    'create_default_harness',
]

__version__ = "19.0.0"