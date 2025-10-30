"""Compatibility shim: re-export multimodal agents from `modules.agents.multimodal`."""
from modules.agents.multimodal import (
    VisionAgent,
    AudioAgent,
    SensorFusionAgent,
    ARInsightAgent,
)

__all__ = ["VisionAgent", "AudioAgent", "SensorFusionAgent", "ARInsightAgent"]
