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
from datetime import datetime
from .base_agent import BaseAgent
from ..utils import now_ts

logger = logging.getLogger("kalki.agents.phase6")


class MetaHypothesisAgent(BaseAgent):
    """
    Generates and tests hypotheses for problem-solving
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="MetaHypothesisAgent", config=config)
        self.hypotheses = []
    
    def generate_hypothesis(self, problem: str, observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a hypothesis based on problem and observations
        """
        try:
            hypothesis_id = f"hyp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # Simple hypothesis generation (can be enhanced with LLM)
            hypothesis = {
                "hypothesis_id": hypothesis_id,
                "problem": problem,
                "observations": observations,
                "statement": self._formulate_hypothesis(problem, observations),
                "confidence": 0.5,
                "status": "untested",
                "created_at": now_ts()
            }
            
            self.hypotheses.append(hypothesis)
            self.logger.info(f"Generated hypothesis {hypothesis_id}")
            return hypothesis
        except Exception as e:
            self.logger.exception(f"Failed to generate hypothesis: {e}")
            raise
    
    def _formulate_hypothesis(self, problem: str, observations: List[Dict[str, Any]]) -> str:
        """Formulate hypothesis statement"""
        # Simple formulation (can be enhanced with LLM)
        return f"Based on {len(observations)} observations, {problem} may be explained by pattern analysis."
    
    def test_hypothesis(self, hypothesis_id: str, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test a hypothesis with experimental results
        """
        try:
            hypothesis = next((h for h in self.hypotheses if h["hypothesis_id"] == hypothesis_id), None)
            if not hypothesis:
                raise ValueError(f"Hypothesis {hypothesis_id} not found")
            
            # Update hypothesis based on test results
            success = test_results.get("success", False)
            hypothesis["status"] = "confirmed" if success else "rejected"
            hypothesis["test_results"] = test_results
            hypothesis["tested_at"] = now_ts()
            
            if success:
                hypothesis["confidence"] = min(0.95, hypothesis["confidence"] + 0.2)
            else:
                hypothesis["confidence"] = max(0.05, hypothesis["confidence"] - 0.3)
            
            self.logger.info(f"Tested hypothesis {hypothesis_id}: {hypothesis['status']}")
            return hypothesis
        except Exception as e:
            self.logger.exception(f"Failed to test hypothesis: {e}")
            raise
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute hypothesis tasks"""
        action = task.get("action")
        
        if action == "generate":
            hypothesis = self.generate_hypothesis(task["problem"], task["observations"])
            return {"status": "success", "hypothesis": hypothesis}
        elif action == "test":
            hypothesis = self.test_hypothesis(task["hypothesis_id"], task["test_results"])
            return {"status": "success", "hypothesis": hypothesis}
        elif action == "list":
            return {"status": "success", "hypotheses": self.hypotheses}
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}


class FeedbackAgent(BaseAgent):
    """
    Learns from outcomes and adjusts strategies
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="FeedbackAgent", config=config)
        self.feedback_history = []
        self.learning_rate = config.get("learning_rate", 0.1) if config else 0.1
    
    def record_feedback(self, task_id: str, outcome: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, Any]:
        """
        Record feedback from task outcome
        """
        try:
            feedback = {
                "task_id": task_id,
                "outcome": outcome,
                "expected": expected,
                "error": self._calculate_error(outcome, expected),
                "timestamp": now_ts()
            }
            
            self.feedback_history.append(feedback)
            self.logger.debug(f"Recorded feedback for task {task_id}")
            return feedback
        except Exception as e:
            self.logger.exception(f"Failed to record feedback: {e}")
            raise
    
    def _calculate_error(self, outcome: Dict[str, Any], expected: Dict[str, Any]) -> float:
        """Calculate error between outcome and expected"""
        # Simplified error calculation
        if outcome.get("status") == expected.get("status"):
            return 0.0
        return 1.0
    
    def get_adjustments(self) -> Dict[str, Any]:
        """
        Get recommended adjustments based on feedback
        """
        try:
            if not self.feedback_history:
                return {"adjustments": []}
            
            # Analyze recent feedback
            recent_feedback = self.feedback_history[-10:]
            avg_error = sum(f["error"] for f in recent_feedback) / len(recent_feedback)
            
            adjustments = {
                "avg_error": avg_error,
                "recommendation": "increase_resources" if avg_error > 0.5 else "maintain",
                "timestamp": now_ts()
            }
            
            return adjustments
        except Exception as e:
            self.logger.exception(f"Failed to get adjustments: {e}")
            return {"adjustments": []}
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute feedback tasks"""
        action = task.get("action")
        
        if action == "record":
            feedback = self.record_feedback(task["task_id"], task["outcome"], task["expected"])
            return {"status": "success", "feedback": feedback}
        elif action == "adjustments":
            adjustments = self.get_adjustments()
            return {"status": "success", "adjustments": adjustments}
        elif action == "history":
            return {"status": "success", "history": self.feedback_history}
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}


class PerformanceMonitorAgent(BaseAgent):
    """
    Tracks and reports performance metrics
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="PerformanceMonitorAgent", config=config)
        self.metrics = {}
    
    def record_metric(self, metric_name: str, value: float, metadata: Optional[Dict[str, Any]] = None):
        """Record a performance metric"""
        try:
            if metric_name not in self.metrics:
                self.metrics[metric_name] = []
            
            entry = {
                "value": value,
                "timestamp": now_ts(),
                "metadata": metadata or {}
            }
            
            self.metrics[metric_name].append(entry)
            self.logger.debug(f"Recorded metric {metric_name}: {value}")
        except Exception as e:
            self.logger.exception(f"Failed to record metric: {e}")
    
    def get_metric_stats(self, metric_name: str) -> Dict[str, Any]:
        """Get statistics for a metric"""
        try:
            if metric_name not in self.metrics:
                return {}
            
            values = [m["value"] for m in self.metrics[metric_name]]
            
            stats = {
                "metric_name": metric_name,
                "count": len(values),
                "min": min(values) if values else 0,
                "max": max(values) if values else 0,
                "avg": sum(values) / len(values) if values else 0,
                "latest": values[-1] if values else 0
            }
            
            return stats
        except Exception as e:
            self.logger.exception(f"Failed to get metric stats: {e}")
            return {}
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute performance monitoring tasks"""
        action = task.get("action")
        
        if action == "record":
            self.record_metric(task["metric_name"], task["value"], task.get("metadata"))
            return {"status": "success"}
        elif action == "stats":
            stats = self.get_metric_stats(task["metric_name"])
            return {"status": "success", "stats": stats}
        elif action == "list_metrics":
            return {"status": "success", "metrics": list(self.metrics.keys())}
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}


class ConflictDetectionAgent(BaseAgent):
    """
    Detects conflicts in knowledge base
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="ConflictDetectionAgent", config=config)
        self.conflicts = []
    
    def detect_conflicts(self, knowledge_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect conflicts between knowledge items
        """
        try:
            conflicts = []
            
            # Simple conflict detection: check for contradictory statements
            for i, item1 in enumerate(knowledge_items):
                for item2 in knowledge_items[i+1:]:
                    if self._are_conflicting(item1, item2):
                        conflict = {
                            "conflict_id": f"conflict_{len(conflicts)}",
                            "item1": item1,
                            "item2": item2,
                            "severity": "medium",
                            "detected_at": now_ts()
                        }
                        conflicts.append(conflict)
            
            self.conflicts.extend(conflicts)
            self.logger.info(f"Detected {len(conflicts)} conflicts")
            return conflicts
        except Exception as e:
            self.logger.exception(f"Failed to detect conflicts: {e}")
            return []
    
    def _are_conflicting(self, item1: Dict[str, Any], item2: Dict[str, Any]) -> bool:
        """Check if two knowledge items conflict"""
        # Simplified conflict detection
        # Can be enhanced with semantic similarity and contradiction detection
        
        text1 = item1.get("text", "").lower()
        text2 = item2.get("text", "").lower()
        
        # Check for simple contradictions (not/no)
        if "not" in text1 and text2.replace("not", "").strip() == text1.replace("not", "").strip():
            return True
        
        return False
    
    def resolve_conflict(self, conflict_id: str, resolution: str, chosen_item: int):
        """
        Resolve a detected conflict
        """
        try:
            conflict = next((c for c in self.conflicts if c["conflict_id"] == conflict_id), None)
            if not conflict:
                raise ValueError(f"Conflict {conflict_id} not found")
            
            conflict["resolution"] = resolution
            conflict["chosen_item"] = chosen_item
            conflict["resolved_at"] = now_ts()
            
            self.logger.info(f"Resolved conflict {conflict_id}")
        except Exception as e:
            self.logger.exception(f"Failed to resolve conflict: {e}")
            raise
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute conflict detection tasks"""
        action = task.get("action")
        
        if action == "detect":
            conflicts = self.detect_conflicts(task["knowledge_items"])
            return {"status": "success", "conflicts": conflicts}
        elif action == "resolve":
            self.resolve_conflict(task["conflict_id"], task["resolution"], task["chosen_item"])
            return {"status": "success"}
        elif action == "list":
            return {"status": "success", "conflicts": self.conflicts}
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}
