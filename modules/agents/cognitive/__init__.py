"""
Cognitive agents (Phase 6, 10, 11) - Meta-reasoning, creativity, and self-improvement

Lightweight re-exports: implementations moved into per-agent modules.
"""

from .meta_hypothesis import MetaHypothesisAgent
from .creative import CreativeAgent
from .feedback import FeedbackAgent
from .optimization import OptimizationAgent

__all__ = [
    "MetaHypothesisAgent",
    "CreativeAgent",
    "FeedbackAgent",
    "OptimizationAgent",
]