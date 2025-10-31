#!/usr/bin/env python3
"""
MetaHypothesisAgent — Cognitive hypothesis formation and testing loop
Generates unique hypothesis IDs, proper try/except safety, confidence scoring logic
"""
import json
import os
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from ..base_agent import BaseAgent


class MetaHypothesisAgent(BaseAgent):
    """
    Cognitive hypothesis formation and testing loop.
    Generates unique hypothesis IDs with confidence scoring.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="MetaHypothesisAgent", config=config)
        self.hypotheses = {}
        self.hypotheses_file = Path.home() / "Desktop" / "Kalki" / "vector_db" / "hypotheses.json"
        self._load_hypotheses()

    def _load_hypotheses(self):
        """Load persisted hypotheses from disk"""
        try:
            if self.hypotheses_file.exists():
                with open(self.hypotheses_file, 'r') as f:
                    data = json.load(f)
                    self.hypotheses = data.get('hypotheses', {})
                self.logger.info(f"Loaded {len(self.hypotheses)} hypotheses from disk")
        except Exception as e:
            self.logger.warning(f"Failed to load hypotheses: {e}")

    def _save_hypotheses(self):
        """Persist hypotheses to disk"""
        try:
            self.hypotheses_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'hypotheses': self.hypotheses,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.hypotheses_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save hypotheses: {e}")

    def generate_hypothesis(self, observation: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a new hypothesis based on observation and context
        """
        try:
            hypothesis_id = str(uuid.uuid4())

            # Formulate hypothesis using pattern extraction
            hypothesis_text = self._formulate_hypothesis(observation, context)

            # Calculate initial confidence
            confidence = self._calculate_confidence(observation, context)

            hypothesis = {
                "id": hypothesis_id,
                "observation": observation,
                "hypothesis": hypothesis_text,
                "confidence": confidence,
                "context": context or {},
                "status": "proposed",
                "created_at": datetime.now().isoformat(),
                "tested_count": 0,
                "confirmed_count": 0,
                "rejected_count": 0
            }

            self.hypotheses[hypothesis_id] = hypothesis
            self._save_hypotheses()

            self.logger.info(f"Generated hypothesis {hypothesis_id} with confidence {confidence:.2f}")
            return hypothesis

        except Exception as e:
            self.logger.exception(f"Failed to generate hypothesis: {e}")
            raise

    def _formulate_hypothesis(self, observation: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Formulate hypothesis from observation
        TODO: Replace with actual pattern extraction once local LLM integration available
        """
        # Simple rule-based hypothesis formation
        observation_lower = observation.lower()

        if "error" in observation_lower or "fail" in observation_lower:
            return f"System errors may be caused by {self._extract_error_patterns(observation)}"
        elif "slow" in observation_lower or "performance" in observation_lower:
            return f"Performance issues may relate to {self._extract_performance_patterns(observation)}"
        elif "memory" in observation_lower:
            return f"Memory usage patterns suggest {self._extract_memory_patterns(observation)}"
        else:
            return f"Based on observation '{observation}', the system may benefit from optimization"

    def _extract_error_patterns(self, observation: str) -> str:
        """Extract error patterns (placeholder)"""
        return "resource constraints or configuration issues"

    def _extract_performance_patterns(self, observation: str) -> str:
        """Extract performance patterns (placeholder)"""
        return "resource allocation or processing bottlenecks"

    def _extract_memory_patterns(self, observation: str) -> str:
        """Extract memory patterns (placeholder)"""
        return "memory leaks or inefficient data structures"

    def _calculate_confidence(self, observation: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate initial confidence score for hypothesis
        """
        confidence = 0.5  # Base confidence

        # Increase confidence based on observation quality
        if len(observation) > 50:
            confidence += 0.1
        if context and len(context) > 0:
            confidence += 0.1

        # Increase confidence for specific keywords
        keywords = ["error", "fail", "slow", "memory", "performance"]
        matches = sum(1 for keyword in keywords if keyword in observation.lower())
        confidence += matches * 0.05

        return min(confidence, 0.95)  # Cap at 95%

    def test_hypothesis(self, hypothesis_id: str, test_result: bool) -> Dict[str, Any]:
        """
        Update hypothesis based on test results
        """
        try:
            if hypothesis_id not in self.hypotheses:
                raise ValueError(f"Hypothesis {hypothesis_id} not found")

            hypothesis = self.hypotheses[hypothesis_id]
            hypothesis["tested_count"] += 1

            if test_result:
                hypothesis["confirmed_count"] += 1
                hypothesis["status"] = "confirmed" if hypothesis["confirmed_count"] > hypothesis["rejected_count"] else "pending"
            else:
                hypothesis["rejected_count"] += 1
                hypothesis["status"] = "rejected" if hypothesis["rejected_count"] > hypothesis["confirmed_count"] else "pending"

            # Recalculate confidence based on test results
            total_tests = hypothesis["confirmed_count"] + hypothesis["rejected_count"]
            if total_tests > 0:
                hypothesis["confidence"] = hypothesis["confirmed_count"] / total_tests

            hypothesis["last_tested"] = datetime.now().isoformat()
            self._save_hypotheses()

            self.logger.info(f"Updated hypothesis {hypothesis_id}: {hypothesis['status']} (confidence: {hypothesis['confidence']:.2f})")
            return hypothesis

        except Exception as e:
            self.logger.exception(f"Failed to test hypothesis: {e}")
            raise

    def get_hypotheses(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get hypotheses, optionally filtered by status
        """
        hypotheses = list(self.hypotheses.values())

        if status:
            hypotheses = [h for h in hypotheses if h["status"] == status]

        return sorted(hypotheses, key=lambda x: x["created_at"], reverse=True)

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute hypothesis generation and testing tasks"""
        action = task.get("action")

        if action == "generate":
            hypothesis = self.generate_hypothesis(task["observation"], task.get("context"))
            return {"status": "success", "hypothesis": hypothesis}
        elif action == "test":
            hypothesis = self.test_hypothesis(task["hypothesis_id"], task["result"])
            return {"status": "success", "hypothesis": hypothesis}
        elif action == "get":
            hypotheses = self.get_hypotheses(task.get("status"))
            return {"status": "success", "hypotheses": hypotheses}
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}