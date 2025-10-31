#!/usr/bin/env python3
"""
Phase 6: Adaptive Cognition & Meta-Reasoning
- MetaHypothesisAgent: Hypothesis generation and testing
- FeedbackAgent: Continuous learning from outcomes
- PerformanceMonitorAgent: Metrics tracking
- ConflictDetectionAgent: Knowledge conflict detection
"""
import logging
from typing import Dict, Any, Optional, List

from .meta_hypothesis_agent import MetaHypothesisAgent
from .feedback_agent import FeedbackAgent
from .performance_monitor_agent import PerformanceMonitorAgent
from .conflict_detection_agent import ConflictDetectionAgent

__all__ = [
    'MetaHypothesisAgent',
    'FeedbackAgent',
    'PerformanceMonitorAgent',
    'ConflictDetectionAgent'
]
