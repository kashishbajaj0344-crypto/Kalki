"""
Kalki v2.3 Agent Framework
Multi-phase intelligent agent system

This __init__ is intentionally lightweight: it exposes a few core types and
provides a place to add package-level exports for the individual agents.
Avoid importing all agent modules here to prevent slow imports and circular
dependencies. If you want to expose agent classes at package level, add them
to the __all__ list or use lazy imports (see module __getattr__ example below).
"""

from .base_agent import BaseAgent, AgentCapability
from .agent_manager import AgentManager
from modules.eventbus import EventBus

# If/when you add agent modules you can export them here.
# Example (replace with real module/class names):
# from .planner_agent import PlannerAgent
# from .executor_agent import ExecutorAgent
# from .conversational_agent import ConversationalAgent

__all__ = [
    'BaseAgent',
    'AgentCapability',
    'AgentManager',
    'EventBus',
    # add exported agent names here, e.g. 'PlannerAgent', 'ExecutorAgent'
]

# Optional: lazy attribute access to avoid importing every agent module at import time.
# Requires Python 3.7+. Uncomment and adapt when ready.
#
# def __getattr__(name: str):
#     if name == "PlannerAgent":
#         from .planner_agent import PlannerAgent
#         return PlannerAgent
#     if name == "ExecutorAgent":
#         from .executor_agent import ExecutorAgent
#         return ExecutorAgent
#     raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
"""
Kalki v2.3 Agent Framework
Multi-phase intelligent agent system

This __init__ is intentionally lightweight: it exposes a few core types and
provides a place to add package-level exports for the individual agents.
Avoid importing all agent modules here to prevent slow imports and circular
dependencies. If you want to expose agent classes at package level, add them
to the __all__ list or use lazy imports (see module __getattr__ example below).
"""

from .base_agent import BaseAgent, AgentCapability
from .agent_manager import AgentManager
from modules.eventbus import EventBus

# If/when you add agent modules you can export them here.
# Example (replace with real module/class names):
# from .planner_agent import PlannerAgent
# from .executor_agent import ExecutorAgent
# from .conversational_agent import ConversationalAgent

__all__ = [
    'BaseAgent',
    'AgentCapability',
    'AgentManager',
    'EventBus',
    # add exported agent names here, e.g. 'PlannerAgent', 'ExecutorAgent'
]

# Optional: lazy attribute access to avoid importing every agent module at import time.
# Requires Python 3.7+. Uncomment and adapt when ready.
#
# def __getattr__(name: str):
#     if name == "PlannerAgent":
#         from .planner_agent import PlannerAgent
#         return PlannerAgent
#     if name == "ExecutorAgent":
#         from .executor_agent import ExecutorAgent
#         return ExecutorAgent
#     raise AttributeError(f"module {__name__!r} has no attribute {name!r}")