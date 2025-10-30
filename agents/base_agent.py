"""Compatibility shim to re-export base agent types from `modules.agents.base_agent`."""
from modules.agents.base_agent import BaseAgent, AgentCapability, AgentStatus

__all__ = ["BaseAgent", "AgentCapability", "AgentStatus"]
