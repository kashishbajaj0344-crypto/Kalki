#!/usr/bin/env python3
"""
Phase 10: Creative Cognition & Synthetic Intuition

Separate agent modules for creative capabilities:
- CreativeAgent: Core creative idea generation with dream mode
- DreamModeAgent: Specialized dream session management
- IdeaFusionAgent: Cross-domain idea synthesis
- PatternRecognitionAgent: Advanced pattern discovery and analysis
"""

from .creative_agent import CreativeAgent
from .dream_mode_agent import DreamModeAgent
from .idea_fusion_agent import IdeaFusionAgent
from .pattern_recognition_agent import PatternRecognitionAgent

__all__ = [
    "CreativeAgent",
    "DreamModeAgent",
    "IdeaFusionAgent",
    "PatternRecognitionAgent"
]
