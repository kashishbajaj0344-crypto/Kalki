"""
Kalki Agents Module - Phase 17
Multi-agent coordination with messaging and execution.
Enhanced with memory integration, cooperative chaining, and self-monitoring.
"""

from .base import (
    Agent, AgentContext, AgentResult, AgentStatus,
    Message, MessageBus,
    AgentRegistry, AgentRunner,
    AgentTask, AgentMonitor
)
from .sample_agents import (
    SearchAgent, ExecutorAgent, SafetyAgent, ReasoningAgent
)

__all__ = [
    'Agent',
    'AgentContext',
    'AgentResult',
    'AgentStatus',
    'Message',
    'MessageBus',
    'AgentRegistry',
    'AgentRunner',
    'AgentTask',
    'AgentMonitor',
    'SearchAgent',
    'ExecutorAgent',
    'SafetyAgent',
    'ReasoningAgent',
]

# [Kalki v2.3 — agents/__init__.py v1.0]
