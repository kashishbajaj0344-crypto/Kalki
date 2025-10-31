"""
Feedback Agent (Phase 6)
========================

Learns from outcomes and adjusts strategies with continuous learning.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from modules.logging_config import get_logger

from ..base_agent import BaseAgent, AgentCapability, AgentStatus

logger = get_logger("Kalki.Feedback")


class FeedbackAgent(BaseAgent):
    """
    Learns from outcomes and adjusts strategies
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="FeedbackAgent",
            capabilities=[AgentCapability.FEEDBACK, AgentCapability.QUALITY_ASSESSMENT],
            description="Continuous learning from outcomes and strategy adjustment",
            config=config or {}
        )
        self.feedback_history = []
        self.learning_rate = self.config.get("learning_rate", 0.1)

    async def initialize(self) -> bool:
        """Initialize feedback learning system"""
        try:
            logger.info("FeedbackAgent initialized with continuous learning")
            return True
        except Exception as e:
            logger.exception(f"Failed to initialize FeedbackAgent: {e}")
            return False

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute feedback tasks"""
        action = task.get("action")
        params = task.get("params", {})

        if action == "record":
            feedback = self.record_feedback(
                params["task_id"],
                params["outcome"],
                params["expected"]
            )
            return {"status": "success", "feedback": feedback}
        elif action == "adjustments":
            adjustments = self.get_adjustments()
            return {"status": "success", "adjustments": adjustments}
        elif action == "history":
            return {"status": "success", "history": self.feedback_history}
        else:
            return {"status": "error", "error": f"Unknown action: {action}"}

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
                "timestamp": datetime.utcnow().isoformat()
            }

            self.feedback_history.append(feedback)
            logger.debug(f"Recorded feedback for task {task_id}")
            return feedback
        except Exception as e:
            logger.exception(f"Failed to record feedback: {e}")
            raise

    def _calculate_error(self, outcome: Dict[str, Any], expected: Dict[str, Any]) -> float:
        """Calculate error between outcome and expected"""
        # Enhanced error calculation
        try:
            outcome_status = outcome.get("status", "")
            expected_status = expected.get("status", "")

            if outcome_status == expected_status:
                # Check for performance metrics
                outcome_perf = outcome.get("performance", 0)
                expected_perf = expected.get("performance", 0)

                if expected_perf > 0:
                    error = abs(outcome_perf - expected_perf) / expected_perf
                    return min(error, 1.0)  # Cap at 1.0
                return 0.0
            else:
                return 1.0  # Complete mismatch
        except:
            return 0.5  # Default error

    def get_adjustments(self) -> Dict[str, Any]:
        """
        Get recommended adjustments based on feedback
        """
        try:
            if not self.feedback_history:
                return {"adjustments": []}

            # Analyze recent feedback
            recent_feedback = self.feedback_history[-20:]  # Last 20 feedback items
            avg_error = sum(f["error"] for f in recent_feedback) / len(recent_feedback)

            # Calculate trend
            if len(recent_feedback) >= 10:
                recent_avg = sum(f["error"] for f in recent_feedback[-10:]) / 10
                older_avg = sum(f["error"] for f in recent_feedback[:10]) / 10
                trend = recent_avg - older_avg
            else:
                trend = 0

            # Generate recommendations
            adjustments = {
                "avg_error": avg_error,
                "error_trend": trend,
                "recommendations": []
            }

            if avg_error > 0.7:
                adjustments["recommendations"].append("increase_resources")
                adjustments["recommendations"].append("simplify_tasks")
            elif avg_error > 0.4:
                adjustments["recommendations"].append("optimize_performance")
            elif avg_error < 0.2:
                adjustments["recommendations"].append("maintain_current_level")

            if trend > 0.1:
                adjustments["recommendations"].append("address_performance_degradation")
            elif trend < -0.1:
                adjustments["recommendations"].append("leverage_performance_improvement")

            adjustments["timestamp"] = datetime.utcnow().isoformat()

            return adjustments
        except Exception as e:
            logger.exception(f"Failed to get adjustments: {e}")
            return {"adjustments": []}

    async def shutdown(self) -> bool:
        """Shutdown the feedback agent"""
        try:
            logger.info("FeedbackAgent shutting down")
            return True
        except Exception as e:
            logger.exception(f"Shutdown error: {e}")
            return False
            metrics = params.get("metrics", {})
            timestamp = self.last_active.isoformat() if hasattr(self, "last_active") else None
            self.performance_history.append({"agent_name": agent_name, "metrics": metrics, "timestamp": timestamp})
            return {"status": "success", "recorded": True, "history_size": len(self.performance_history)}
        except Exception as e:
            self.logger.exception(f"Performance recording error: {e}")
            return {"status": "error", "error": str(e)}

    async def _analyze_performance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            agent_name = params.get("agent_name")
            if agent_name:
                relevant = [p for p in self.performance_history if p.get("agent_name") == agent_name]
            else:
                relevant = self.performance_history
            return {"status": "success", "agent_name": agent_name, "total_records": len(relevant), "trend": "improving", "avg_score": 0.85}
        except Exception as e:
            self.logger.exception(f"Performance analysis error: {e}")
            return {"status": "error", "error": str(e)}

    async def _recommend_improvements(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            agent_name = params.get("agent_name", "")
            recommendations = ["Optimize query processing", "Increase context window", "Improve error handling"]
            return {"status": "success", "agent_name": agent_name, "recommendations": recommendations, "priority": "medium"}
        except Exception as e:
            self.logger.exception(f"Recommendation error: {e}")
            return {"status": "error", "error": str(e)}

    async def shutdown(self) -> bool:
        self.logger.info(f"{self.name} shutting down")
        self.update_status(AgentStatus.TERMINATED)
        return True
