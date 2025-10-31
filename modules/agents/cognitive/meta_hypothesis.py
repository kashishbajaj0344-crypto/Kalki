"""
Meta Hypothesis Agent (Phase 6)
===============================

Generates and tests hypotheses for problem-solving with scientific method approach.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from modules.logging_config import get_logger

from ..base_agent import BaseAgent, AgentCapability, AgentStatus

logger = get_logger("Kalki.MetaHypothesis")


class MetaHypothesisAgent(BaseAgent):
    """
    Generates and tests hypotheses for problem-solving
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="MetaHypothesisAgent",
            capabilities=[AgentCapability.META_REASONING, AgentCapability.REASONING],
            description="Hypothesis generation and testing with scientific methodology",
            config=config or {}
        )
        self.hypotheses = []

    async def initialize(self) -> bool:
        """Initialize hypothesis generation system"""
        try:
            logger.info("MetaHypothesisAgent initialized with scientific methodology")
            return True
        except Exception as e:
            logger.exception(f"Failed to initialize MetaHypothesisAgent: {e}")
            return False

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute hypothesis tasks"""
        action = task.get("action")
        params = task.get("params", {})

        if action == "generate":
            hypothesis = self.generate_hypothesis(
                params["problem"],
                params["observations"]
            )
            return {"status": "success", "hypothesis": hypothesis}
        elif action == "test":
            hypothesis = self.test_hypothesis(
                params["hypothesis_id"],
                params["test_results"]
            )
            return {"status": "success", "hypothesis": hypothesis}
        elif action == "list":
            return {"status": "success", "hypotheses": self.hypotheses}
        else:
            return {"status": "error", "error": f"Unknown action: {action}"}

    def generate_hypothesis(self, problem: str, observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a hypothesis based on problem and observations
        """
        try:
            hypothesis_id = f"hyp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

            # Enhanced hypothesis generation with pattern analysis
            hypothesis = {
                "hypothesis_id": hypothesis_id,
                "problem": problem,
                "observations": observations,
                "statement": self._formulate_hypothesis(problem, observations),
                "confidence": self._calculate_initial_confidence(observations),
                "status": "untested",
                "evidence_strength": self._assess_evidence_strength(observations),
                "created_at": datetime.utcnow().isoformat()
            }

            self.hypotheses.append(hypothesis)
            logger.info(f"Generated hypothesis {hypothesis_id} with confidence {hypothesis['confidence']:.2f}")
            return hypothesis
        except Exception as e:
            logger.exception(f"Failed to generate hypothesis: {e}")
            raise

    def _formulate_hypothesis(self, problem: str, observations: List[Dict[str, Any]]) -> str:
        """Formulate hypothesis statement with pattern analysis"""
        try:
            if not observations:
                return f"Based on limited data, {problem} requires further investigation."

            # Analyze patterns in observations
            patterns = self._analyze_patterns(observations)

            if patterns.get("correlation_found"):
                return f"Based on observed correlations in {len(observations)} data points, {problem} may be explained by {patterns.get('primary_correlation', 'pattern analysis')}."

            if patterns.get("anomaly_detected"):
                return f"Given the anomalous observations, {problem} might be caused by {patterns.get('anomaly_type', 'unusual conditions')}."

            # Default formulation
            return f"Based on {len(observations)} observations, {problem} may be explained by systematic pattern analysis."

        except Exception:
            return f"Based on {len(observations)} observations, {problem} requires hypothesis testing."

    def _analyze_patterns(self, observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in observations"""
        patterns = {
            "correlation_found": False,
            "anomaly_detected": False,
            "data_points": len(observations)
        }

        try:
            # Simple pattern detection
            if len(observations) >= 3:
                # Check for sequential patterns
                values = [obs.get("value", 0) for obs in observations if "value" in obs]
                if len(values) >= 3:
                    # Check for increasing/decreasing trends
                    increasing = all(values[i] <= values[i+1] for i in range(len(values)-1))
                    decreasing = all(values[i] >= values[i+1] for i in range(len(values)-1))

                    if increasing or decreasing:
                        patterns["correlation_found"] = True
                        patterns["primary_correlation"] = "increasing trend" if increasing else "decreasing trend"

                # Check for anomalies (values significantly different from mean)
                if values:
                    mean_val = sum(values) / len(values)
                    std_dev = (sum((v - mean_val) ** 2 for v in values) / len(values)) ** 0.5

                    for obs in observations:
                        val = obs.get("value", mean_val)
                        if abs(val - mean_val) > 2 * std_dev:  # 2 standard deviations
                            patterns["anomaly_detected"] = True
                            patterns["anomaly_type"] = "statistical outlier"
                            break

        except Exception:
            pass

        return patterns

    def _calculate_initial_confidence(self, observations: List[Dict[str, Any]]) -> float:
        """Calculate initial confidence based on observation quality"""
        try:
            if not observations:
                return 0.3

            # Base confidence on number and quality of observations
            confidence = min(0.8, len(observations) / 10)  # Max 0.8 from quantity

            # Adjust for observation quality
            quality_score = 0
            for obs in observations:
                if obs.get("verified", False):
                    quality_score += 1
                if obs.get("source_reliability", 0.5) > 0.7:
                    quality_score += 0.5

            quality_bonus = min(0.2, quality_score / len(observations))
            confidence += quality_bonus

            return max(0.1, min(0.95, confidence))

        except Exception:
            return 0.5

    def _assess_evidence_strength(self, observations: List[Dict[str, Any]]) -> str:
        """Assess the strength of evidence"""
        try:
            count = len(observations)

            if count >= 10:
                return "strong"
            elif count >= 5:
                return "moderate"
            elif count >= 2:
                return "weak"
            else:
                return "insufficient"

        except Exception:
            return "unknown"

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
            p_value = test_results.get("p_value", 1.0)
            effect_size = test_results.get("effect_size", 0)

            # Determine status based on statistical significance
            if p_value < 0.05:  # 95% confidence
                hypothesis["status"] = "confirmed" if success else "rejected"
            elif p_value < 0.1:  # 90% confidence
                hypothesis["status"] = "tentatively_confirmed" if success else "tentatively_rejected"
            else:
                hypothesis["status"] = "inconclusive"

            # Update confidence based on results
            if success and p_value < 0.05:
                hypothesis["confidence"] = min(0.95, hypothesis["confidence"] + 0.3)
            elif not success and p_value < 0.05:
                hypothesis["confidence"] = max(0.05, hypothesis["confidence"] - 0.4)
            else:
                # Inconclusive results - slight confidence decrease
                hypothesis["confidence"] = max(0.05, hypothesis["confidence"] - 0.1)

            # Add test metadata
            hypothesis["test_results"] = test_results
            hypothesis["tested_at"] = datetime.utcnow().isoformat()
            hypothesis["statistical_significance"] = p_value
            hypothesis["effect_size"] = effect_size

            logger.info(f"Tested hypothesis {hypothesis_id}: {hypothesis['status']} (p={p_value:.3f})")
            return hypothesis
        except Exception as e:
            logger.exception(f"Failed to test hypothesis: {e}")
            raise

    async def shutdown(self) -> bool:
        """Shutdown the hypothesis agent"""
        try:
            logger.info("MetaHypothesisAgent shutting down")
            return True
        except Exception as e:
            logger.exception(f"Shutdown error: {e}")
            return False
