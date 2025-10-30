"""Compatibility shim: re-export core agents from `modules.agents.core`."""
from modules.agents.core import (
    DocumentIngestAgent,
    SearchAgent,
    PlannerAgent,
    ReasoningAgent,
    MemoryAgent,
)

__all__ = [
    "DocumentIngestAgent",
    "SearchAgent",
    "PlannerAgent",
    "ReasoningAgent",
    "MemoryAgent",
]
