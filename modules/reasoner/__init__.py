"""
Kalki Reasoner Module - Phase 16
Iterative reasoning with checkpointing and rule-based inference.
Enhanced with memory integration, meta-reasoning, and LLM support.
"""

from .reasoner import (
    Reasoner, ReasoningTrace, Step, StepType,
    InferenceEngine, Rule
)

__all__ = [
    'Reasoner',
    'ReasoningTrace',
    'Step',
    'StepType',
    'InferenceEngine',
    'Rule',
]

# [Kalki v2.3 — reasoner/__init__.py v1.0]