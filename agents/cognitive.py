"""Compatibility shim: re-export cognitive agents from `modules.agents.cognitive`."""
from modules.agents.cognitive import (
    MetaHypothesisAgent,
    CreativeAgent,
    FeedbackAgent,
    OptimizationAgent,
)

__all__ = [
    "MetaHypothesisAgent",
    "CreativeAgent",
    "FeedbackAgent",
    "OptimizationAgent",
]
