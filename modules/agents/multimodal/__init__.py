"""
Multi-modal agents (Phase 13, 17) - Vision, Audio, Sensor Fusion, AR/VR

Includes:
- VisionAgent
- AudioAgent
- SensorFusionAgent (depends on VisionAgent, AudioAgent)
- ARInsightAgent (depends on VisionAgent, SensorFusionAgent)

Also includes AgentOrchestrator: a dependency-aware orchestrator with:
- dependency resolution with cycle detection
- sequential and optional parallel initialization
- avoidance of redundant initialization
- contextual logging
- task validation and execution timing
"""

from .vision import VisionAgent
from .audio import AudioAgent