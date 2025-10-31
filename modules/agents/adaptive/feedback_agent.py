#!/usr/bin/env python3
"""
FeedbackAgent — Self-correction and experience-based learning
Structured feedback recording with learning-rate concept
"""
import json
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from ..base_agent import BaseAgent


class FeedbackAgent(BaseAgent):
    """
    Self-correction and experience-based learning.
    Records feedback and applies adjustments for continuous improvement.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="FeedbackAgent", config=config)
        self.feedback_history = []
        self.learning_rate = config.get("learning_rate", 0.1) if config else 0.1
        self.parameters = config.get("parameters", {}) if config else {}
        self.feedback_file = Path.home() / "Desktop" / "Kalki" / "vector_db" / "feedback_history.json"
        self._load_feedback_history()

    def _load_feedback_history(self):
        """Load persisted feedback history from disk"""
        try:
            if self.feedback_file.exists():
                with open(self.feedback_file, 'r') as f:
                    data = json.load(f)
                    self.feedback_history = data.get('feedback_history', [])
                self.logger.info(f"Loaded {len(self.feedback_history)} feedback entries from disk")
        except Exception as e:
            self.logger.warning(f"Failed to load feedback history: {e}")

    def _save_feedback_history(self):
        """Persist feedback history to disk"""
        try:
            self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'feedback_history': self.feedback_history[-1000:],  # Keep last 1000 entries
                'last_updated': datetime.now().isoformat()
            }
            with open(self.feedback_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save feedback history: {e}")

    def record_feedback(self, action: str, expected_outcome: Any, actual_outcome: Any,
                       context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Record feedback from action execution
        """
        try:
            # Calculate error/difference
            error = self._calculate_error(expected_outcome, actual_outcome)

            feedback_entry = {
                "timestamp": datetime.now().isoformat(),
                "action": action,
                "expected_outcome": expected_outcome,
                "actual_outcome": actual_outcome,
                "error": error,
                "context": context or {},
                "learning_rate": self.learning_rate
            }

            self.feedback_history.append(feedback_entry)
            self._save_feedback_history()

            # Apply adjustment if error is significant
            if abs(error) > 0.1:  # Threshold for adjustment
                adjustment = self._calculate_adjustment(error)
                self.apply_adjustment(adjustment, context)

            self.logger.info(f"Recorded feedback for action '{action}': error={error:.3f}")
            return feedback_entry

        except Exception as e:
            self.logger.exception(f"Failed to record feedback: {e}")
            raise

    def _calculate_error(self, expected: Any, actual: Any) -> float:
        """
        Calculate error between expected and actual outcomes
        TODO: Replace with statistical or embedding-distance metric later
        """
        try:
            # Simple error calculation for numeric values
            if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
                return abs(expected - actual) / max(abs(expected), 1)  # Normalized error

            # For strings, use length difference as proxy
            elif isinstance(expected, str) and isinstance(actual, str):
                return abs(len(expected) - len(actual)) / max(len(expected), 1)

            # For booleans, exact match
            elif isinstance(expected, bool) and isinstance(actual, bool):
                return 0.0 if expected == actual else 1.0

            # For other types, assume no error if they're equal
            else:
                return 0.0 if expected == actual else 0.5

        except Exception:
            return 0.5  # Default error

    def _calculate_adjustment(self, error: float) -> Dict[str, Any]:
        """
        Calculate parameter adjustment based on error
        """
        adjustment = {
            "error": error,
            "learning_rate": self.learning_rate,
            "adjustment_factor": error * self.learning_rate,
            "timestamp": datetime.now().isoformat()
        }
        return adjustment

    def apply_adjustment(self, adjustment: Dict[str, Any], context: Optional[Dict[str, Any]] = None):
        """
        Apply parameter adjustment for continuous learning
        """
        try:
            adjustment_factor = adjustment.get("adjustment_factor", 0)

            # Update learning rate based on performance
            if adjustment_factor > 0.2:  # Large error, reduce learning rate
                self.learning_rate = max(0.01, self.learning_rate * 0.9)
            elif adjustment_factor < 0.05:  # Small error, can increase learning rate
                self.learning_rate = min(0.5, self.learning_rate * 1.05)

            # Update parameters (example: adjust thresholds, weights, etc.)
            for param_name, param_value in self.parameters.items():
                if isinstance(param_value, (int, float)):
                    # Apply small adjustment
                    self.parameters[param_name] = param_value * (1 + adjustment_factor * 0.1)

            self.logger.info(f"Applied adjustment: learning_rate={self.learning_rate:.3f}, factor={adjustment_factor:.3f}")

        except Exception as e:
            self.logger.exception(f"Failed to apply adjustment: {e}")

    def get_feedback_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get summary of recent feedback
        """
        try:
            cutoff_time = datetime.now().timestamp() - (hours * 3600)

            recent_feedback = [
                f for f in self.feedback_history
                if datetime.fromisoformat(f["timestamp"]).timestamp() > cutoff_time
            ]

            if not recent_feedback:
                return {"total_entries": 0, "average_error": 0.0, "summary": "No recent feedback"}

            total_error = sum(f.get("error", 0) for f in recent_feedback)
            average_error = total_error / len(recent_feedback)

            # Group by action
            action_errors = {}
            for feedback in recent_feedback:
                action = feedback.get("action", "unknown")
                if action not in action_errors:
                    action_errors[action] = []
                action_errors[action].append(feedback.get("error", 0))

            action_summary = {
                action: {
                    "count": len(errors),
                    "average_error": sum(errors) / len(errors)
                }
                for action, errors in action_errors.items()
            }

            summary = {
                "total_entries": len(recent_feedback),
                "average_error": average_error,
                "learning_rate": self.learning_rate,
                "action_summary": action_summary,
                "period_hours": hours
            }

            return summary

        except Exception as e:
            self.logger.exception(f"Failed to get feedback summary: {e}")
            return {"error": str(e)}

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute feedback and learning tasks"""
        action = task.get("action")

        if action == "record":
            feedback = self.record_feedback(
                task["action_name"],
                task["expected"],
                task["actual"],
                task.get("context")
            )
            return {"status": "success", "feedback": feedback}
        elif action == "summary":
            summary = self.get_feedback_summary(task.get("hours", 24))
            return {"status": "success", "summary": summary}
        elif action == "adjust":
            adjustment = self._calculate_adjustment(task.get("error", 0))
            self.apply_adjustment(adjustment, task.get("context"))
            return {"status": "success", "adjustment": adjustment}
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}