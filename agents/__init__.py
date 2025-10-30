"""Top-level compatibility shims that re-export the repo-native `modules.*` package.

This package exists to support older import paths that expect `agents.*` at
top-level. It re-exports the core runtime types from `modules.agents`.
"""
from modules.agents import AgentManager, BaseAgent, AgentCapability
from modules.eventbus import EventBus

__all__ = ["AgentManager", "BaseAgent", "AgentCapability", "EventBus"]
