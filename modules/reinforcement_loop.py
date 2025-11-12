# ============================================================
# Kalki v2.4 — reinforcement_loop.py
# ------------------------------------------------------------
# Reinforcement Feedback Loop: Self-Optimization Engine
# - Lightweight reward-based learning system
# - Continuous improvement through response evaluation
# - Meta-core heuristic weight adjustment
# - Performance tracking and optimization
# ============================================================

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import statistics
import numpy as np

from modules.utils.logging_config import get_logger
from modules.meta_core import get_meta_core, ReasoningDepth, OutputStyle

logger = get_logger("Kalki.ReinforcementLoop")

class RewardType(Enum):
    """Types of reward signals"""
    COHERENCE = "coherence"
    SATISFACTION = "satisfaction"
    EFFICIENCY = "efficiency"
    CREATIVITY = "creativity"
    ETHICAL_ALIGNMENT = "ethical_alignment"
    INTERDISCIPLINARY_COVERAGE = "interdisciplinary_coverage"

class FeedbackSource(Enum):
    """Sources of feedback"""
    SELF_EVALUATION = "self_evaluation"
    USER_EXPLICIT = "user_explicit"
    USER_IMPLICIT = "user_implicit"
    SYSTEM_METRICS = "system_metrics"
    PEER_REVIEW = "peer_review"

@dataclass
class RewardSignal:
    """Individual reward signal"""
    reward_type: RewardType
    value: float  # 0-1 scale
    confidence: float  # 0-1 scale
    source: FeedbackSource
    timestamp: str
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ResponseEvaluation:
    """Complete evaluation of a response"""
    response_id: str
    query: str
    response: str
    reasoning_depth_used: str
    output_style_used: str
    reward_signals: List[RewardSignal] = field(default_factory=list)
    composite_score: float = 0.0
    timestamp: str = ""
    session_id: str = ""
    feedback_collected: bool = False

@dataclass
class HeuristicWeights:
    """Adjustable weights for meta-core heuristics"""
    depth_selection_weights: Dict[str, float] = field(default_factory=lambda: {
        "summary": 1.0,
        "standard": 1.0,
        "deep_analysis": 1.0,
        "auto": 1.0
    })
    style_preference_weights: Dict[str, float] = field(default_factory=lambda: {
        "conversational": 1.0,
        "structured": 1.0,
        "technical": 1.0
    })
    domain_importance_weights: Dict[str, float] = field(default_factory=lambda: {
        "mathematics": 1.0,
        "physics": 1.0,
        "biology": 1.0,
        "computation": 1.0,
        "art": 1.0,
        "psychology": 1.0,
        "systems": 1.0
    })
    ethical_priority_weights: Dict[str, float] = field(default_factory=lambda: {
        "safety": 1.0,
        "fairness": 1.0,
        "transparency": 1.0,
        "sustainability": 1.0
    })

@dataclass
class PerformanceMetrics:
    """System-wide performance tracking"""
    total_responses: int = 0
    average_composite_score: float = 0.0
    improvement_rate: float = 0.0
    bias_detection_rate: float = 0.0
    ethical_alignment_rate: float = 1.0
    efficiency_ratio: float = 1.0
    learning_sessions: int = 0
    last_update: str = ""

class ReinforcementLoop:
    """
    Reinforcement Feedback Loop: Self-Optimization Engine

    Implements lightweight reward-based learning where:
    - Each response receives reward_signal: coherence × satisfaction × efficiency
    - Meta-core adjusts heuristic weights for reasoning depth and response structure
    - Continuous self-optimization over time
    """

    def __init__(self, learning_rate: float = 0.1, memory_size: int = 1000):
        self.learning_rate = learning_rate
        self.memory_size = memory_size

        # Core data structures
        self.response_history: List[ResponseEvaluation] = []
        self.heuristic_weights = HeuristicWeights()
        self.performance_metrics = PerformanceMetrics()

        # Learning state
        self.baseline_scores: List[float] = []
        self.weight_update_history: List[Dict[str, Any]] = []

        # Persistence
        self.data_file = "data/reinforcement_data.json"
        self.weights_file = "data/heuristic_weights.json"

        # Load existing data
        self._load_persistent_data()

        logger.info(f"Reinforcement Loop initialized with learning rate: {learning_rate}")

    def _load_persistent_data(self):
        """Load persistent reinforcement data"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    # Load response history (last 500 entries)
                    for item in data.get('response_history', [])[-500:]:
                        eval_obj = ResponseEvaluation(**item)
                        self.response_history.append(eval_obj)

                    # Load baseline scores
                    self.baseline_scores = data.get('baseline_scores', [])

            if os.path.exists(self.weights_file):
                with open(self.weights_file, 'r') as f:
                    weights_data = json.load(f)
                    self.heuristic_weights = HeuristicWeights(**weights_data.get('weights', {}))
                    self.weight_update_history = weights_data.get('update_history', [])

        except Exception as e:
            logger.warning(f"Failed to load persistent data: {e}")

    def _save_persistent_data(self):
        """Save reinforcement data persistently"""
        try:
            os.makedirs("data", exist_ok=True)

            # Save response history and baselines
            data = {
                'response_history': [asdict(r) for r in self.response_history[-500:]],
                'baseline_scores': self.baseline_scores[-100:],
                'last_updated': datetime.now().isoformat()
            }

            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)

            # Save weights and update history
            weights_data = {
                'weights': asdict(self.heuristic_weights),
                'update_history': self.weight_update_history[-50:],
                'last_updated': datetime.now().isoformat()
            }

            with open(self.weights_file, 'w') as f:
                json.dump(weights_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save persistent data: {e}")

    async def evaluate_response(self,
                              response_id: str,
                              query: str,
                              response: str,
                              reasoning_depth: str,
                              output_style: str,
                              session_id: str = "") -> ResponseEvaluation:
        """
        Evaluate a response and generate reward signals

        Args:
            response_id: Unique identifier for the response
            query: The original query
            response: The generated response
            reasoning_depth: Depth used for this response
            output_style: Style used for this response
            session_id: Session identifier

        Returns:
            Complete response evaluation with reward signals
        """

        evaluation = ResponseEvaluation(
            response_id=response_id,
            query=query,
            response=response,
            reasoning_depth_used=reasoning_depth,
            output_style_used=output_style,
            timestamp=datetime.now().isoformat(),
            session_id=session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Generate reward signals through self-evaluation
        reward_signals = await self._generate_reward_signals(query, response, reasoning_depth, output_style)

        evaluation.reward_signals = reward_signals
        evaluation.composite_score = self._calculate_composite_score(reward_signals)
        evaluation.feedback_collected = True

        # Add to history
        self.response_history.append(evaluation)
        if len(self.response_history) > self.memory_size:
            self.response_history.pop(0)

        # Update baseline scores
        self.baseline_scores.append(evaluation.composite_score)
        if len(self.baseline_scores) > 100:
            self.baseline_scores.pop(0)

        # Trigger learning update
        await self._update_weights_from_evaluation(evaluation)

        # Update performance metrics
        self._update_performance_metrics()

        # Persist data
        self._save_persistent_data()

        logger.info(f"Response {response_id} evaluated with composite score: {evaluation.composite_score:.3f}")

        return evaluation

    async def _generate_reward_signals(self,
                                     query: str,
                                     response: str,
                                     reasoning_depth: str,
                                     output_style: str) -> List[RewardSignal]:
        """Generate reward signals through self-evaluation"""

        signals = []

        # 1. Coherence reward - logical consistency
        coherence_score = self._evaluate_coherence(query, response)
        signals.append(RewardSignal(
            reward_type=RewardType.COHERENCE,
            value=coherence_score,
            confidence=0.8,
            source=FeedbackSource.SELF_EVALUATION,
            timestamp=datetime.now().isoformat(),
            context={"query": query, "response_length": len(response)}
        ))

        # 2. Satisfaction reward - user need fulfillment
        satisfaction_score = self._evaluate_satisfaction(query, response)
        signals.append(RewardSignal(
            reward_type=RewardType.SATISFACTION,
            value=satisfaction_score,
            confidence=0.7,
            source=FeedbackSource.SELF_EVALUATION,
            timestamp=datetime.now().isoformat(),
            context={"query_complexity": self._assess_query_complexity(query)}
        ))

        # 3. Efficiency reward - quality vs. resource usage
        efficiency_score = self._evaluate_efficiency(response, reasoning_depth)
        signals.append(RewardSignal(
            reward_type=RewardType.EFFICIENCY,
            value=efficiency_score,
            confidence=0.9,
            source=FeedbackSource.SYSTEM_METRICS,
            timestamp=datetime.now().isoformat(),
            context={"reasoning_depth": reasoning_depth, "response_length": len(response)}
        ))

        # 4. Creativity reward - novel insights and approaches
        creativity_score = self._evaluate_creativity(query, response)
        signals.append(RewardSignal(
            reward_type=RewardType.CREATIVITY,
            value=creativity_score,
            confidence=0.6,
            source=FeedbackSource.SELF_EVALUATION,
            timestamp=datetime.now().isoformat(),
            context={"domain_novelty": self._assess_domain_novelty(query, response)}
        ))

        # 5. Ethical alignment reward - safety and appropriateness
        ethical_score = self._evaluate_ethical_alignment(query, response)
        signals.append(RewardSignal(
            reward_type=RewardType.ETHICAL_ALIGNMENT,
            value=ethical_score,
            confidence=0.95,
            source=FeedbackSource.SELF_EVALUATION,
            timestamp=datetime.now().isoformat(),
            context={"safety_check": True}
        ))

        # 6. Interdisciplinary coverage reward
        coverage_score = self._evaluate_interdisciplinary_coverage(response)
        signals.append(RewardSignal(
            reward_type=RewardType.INTERDISCIPLINARY_COVERAGE,
            value=coverage_score,
            confidence=0.8,
            source=FeedbackSource.SELF_EVALUATION,
            timestamp=datetime.now().isoformat(),
            context={"domain_count": self._count_domains_covered(response)}
        ))

        return signals

    def _calculate_composite_score(self, reward_signals: List[RewardSignal]) -> float:
        """Calculate composite score from reward signals"""
        if not reward_signals:
            return 0.0

        # Weighted combination: coherence × satisfaction × efficiency
        weights = {
            RewardType.COHERENCE: 0.25,
            RewardType.SATISFACTION: 0.25,
            RewardType.EFFICIENCY: 0.2,
            RewardType.CREATIVITY: 0.1,
            RewardType.ETHICAL_ALIGNMENT: 0.15,
            RewardType.INTERDISCIPLINARY_COVERAGE: 0.05
        }

        composite = 1.0
        total_weight = 0.0

        for signal in reward_signals:
            if signal.reward_type in weights:
                weight = weights[signal.reward_type]
                composite *= (signal.value ** weight)
                total_weight += weight

        # Normalize by total weight
        if total_weight > 0:
            composite = composite ** (1.0 / total_weight)

        return min(1.0, max(0.0, composite))

    async def _update_weights_from_evaluation(self, evaluation: ResponseEvaluation):
        """Update heuristic weights based on evaluation"""

        if evaluation.composite_score < 0.5:
            # Poor performance - reduce weights for used strategies
            adjustment_factor = 0.95
        elif evaluation.composite_score > 0.8:
            # Good performance - increase weights for used strategies
            adjustment_factor = 1.05
        else:
            # Neutral performance - slight adjustment
            adjustment_factor = 1.02

        # Update depth selection weights
        depth_key = evaluation.reasoning_depth_used
        if depth_key in self.heuristic_weights.depth_selection_weights:
            old_weight = self.heuristic_weights.depth_selection_weights[depth_key]
            new_weight = old_weight * adjustment_factor
            self.heuristic_weights.depth_selection_weights[depth_key] = new_weight

        # Update style preference weights
        style_key = evaluation.output_style_used
        if style_key in self.heuristic_weights.style_preference_weights:
            old_weight = self.heuristic_weights.style_preference_weights[style_key]
            new_weight = old_weight * adjustment_factor
            self.heuristic_weights.style_preference_weights[style_key] = new_weight

        # Record weight update
        update_record = {
            "timestamp": datetime.now().isoformat(),
            "evaluation_id": evaluation.response_id,
            "composite_score": evaluation.composite_score,
            "depth_used": evaluation.reasoning_depth_used,
            "style_used": evaluation.output_style_used,
            "adjustments": {
                "depth_weights": self.heuristic_weights.depth_selection_weights.copy(),
                "style_weights": self.heuristic_weights.style_preference_weights.copy()
            }
        }

        self.weight_update_history.append(update_record)
        if len(self.weight_update_history) > 50:
            self.weight_update_history.pop(0)

        logger.debug(f"Weights updated for response {evaluation.response_id}: score={evaluation.composite_score:.3f}")

    def _update_performance_metrics(self):
        """Update system-wide performance metrics"""

        if not self.response_history:
            return

        recent_evaluations = self.response_history[-50:]  # Last 50 responses

        self.performance_metrics.total_responses = len(self.response_history)
        self.performance_metrics.average_composite_score = statistics.mean(
            [e.composite_score for e in recent_evaluations]
        )

        # Calculate improvement rate (trend over last 20 evaluations)
        if len(recent_evaluations) >= 20:
            recent_scores = [e.composite_score for e in recent_evaluations[-20:]]
            if len(recent_scores) >= 10:
                first_half = statistics.mean(recent_scores[:10])
                second_half = statistics.mean(recent_scores[10:])
                if first_half > 0:
                    self.performance_metrics.improvement_rate = (second_half - first_half) / first_half

        # Calculate bias detection rate
        bias_signals = []
        for eval in recent_evaluations:
            for signal in eval.reward_signals:
                if signal.reward_type == RewardType.COHERENCE and signal.value < 0.7:
                    bias_signals.append(signal)

        if recent_evaluations:
            self.performance_metrics.bias_detection_rate = len(bias_signals) / len(recent_evaluations)

        # Ethical alignment (assume high unless violations detected)
        ethical_violations = sum(1 for eval in recent_evaluations
                               for signal in eval.reward_signals
                               if signal.reward_type == RewardType.ETHICAL_ALIGNMENT and signal.value < 0.8)

        if recent_evaluations:
            self.performance_metrics.ethical_alignment_rate = 1.0 - (ethical_violations / len(recent_evaluations))

        # Efficiency ratio (relative to baseline)
        if self.baseline_scores:
            baseline_avg = statistics.mean(self.baseline_scores[:50])  # First 50 as baseline
            current_avg = statistics.mean([e.composite_score for e in recent_evaluations])
            if baseline_avg > 0:
                self.performance_metrics.efficiency_ratio = current_avg / baseline_avg

        self.performance_metrics.learning_sessions += 1
        self.performance_metrics.last_update = datetime.now().isoformat()

    # Evaluation helper methods
    def _evaluate_coherence(self, query: str, response: str) -> float:
        """Evaluate logical coherence"""
        # Simple heuristics - could be enhanced with NLP
        coherence_indicators = [
            len(response) > 50,  # Substantial response
            "?" not in response[-100:],  # Doesn't end with questions
            len(response.split(".")) > 3,  # Multiple sentences
            any(word in response.lower() for word in ["therefore", "thus", "consequently", "because"])
        ]
        return sum(coherence_indicators) / len(coherence_indicators)

    def _evaluate_satisfaction(self, query: str, response: str) -> float:
        """Evaluate user satisfaction potential"""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())

        # Overlap indicates relevance
        overlap_ratio = len(query_words & response_words) / len(query_words) if query_words else 0

        # Length appropriateness
        query_complexity = self._assess_query_complexity(query)
        response_length_score = min(1.0, len(response) / (query_complexity * 500 + 100))

        return (overlap_ratio + response_length_score) / 2

    def _evaluate_efficiency(self, response: str, reasoning_depth: str) -> float:
        """Evaluate efficiency (quality vs. resources)"""
        # Depth-appropriate length
        depth_multipliers = {
            "summary": 0.5,
            "standard": 1.0,
            "deep_analysis": 2.0,
            "auto": 1.0
        }

        expected_length = 500 * depth_multipliers.get(reasoning_depth, 1.0)
        actual_length = len(response)

        # Efficiency score (closer to expected = more efficient)
        if expected_length > 0:
            efficiency = 1.0 - min(1.0, abs(actual_length - expected_length) / expected_length)
        else:
            efficiency = 0.8  # Default

        return efficiency

    def _evaluate_creativity(self, query: str, response: str) -> float:
        """Evaluate creativity and novelty"""
        # Look for novel combinations and insights
        creativity_indicators = [
            "innovative" in response.lower(),
            "novel" in response.lower(),
            "unique" in response.lower(),
            any(word in response.lower() for word in ["imagine", "envision", "create", "design"]),
            len(set(response.lower().split())) / len(response.split()) > 0.7  # Vocabulary diversity
        ]
        return sum(creativity_indicators) / len(creativity_indicators)

    def _evaluate_ethical_alignment(self, query: str, response: str) -> float:
        """Evaluate ethical alignment and safety"""
        # Check for harmful content
        harmful_indicators = [
            "hack" in response.lower(),
            "exploit" in response.lower(),
            "illegal" in response.lower(),
            "dangerous" in response.lower()
        ]

        harmful_score = sum(harmful_indicators)

        # Check for positive ethical indicators
        ethical_indicators = [
            "safe" in response.lower(),
            "ethical" in response.lower(),
            "responsible" in response.lower(),
            "sustainable" in response.lower()
        ]

        ethical_score = sum(ethical_indicators)

        # Combine: high ethical score, low harmful score
        base_score = 0.8  # Assume generally ethical
        ethical_bonus = min(0.2, ethical_score * 0.05)
        harmful_penalty = min(0.5, harmful_score * 0.1)

        return base_score + ethical_bonus - harmful_penalty

    def _evaluate_interdisciplinary_coverage(self, response: str) -> float:
        """Evaluate interdisciplinary coverage"""
        domains = {
            "mathematics": ["math", "equation", "calculation", "algorithm"],
            "physics": ["physics", "force", "energy", "quantum", "relativity"],
            "biology": ["biology", "cell", "organism", "evolution", "dna"],
            "computation": ["computer", "software", "algorithm", "data", "ai"],
            "art": ["art", "design", "aesthetic", "beauty", "creative"],
            "psychology": ["psychology", "mind", "behavior", "cognitive", "emotion"],
            "systems": ["system", "network", "complexity", "emergence", "feedback"]
        }

        covered_domains = 0
        for domain, keywords in domains.items():
            if any(keyword in response.lower() for keyword in keywords):
                covered_domains += 1

        return min(1.0, covered_domains / 3)  # Target: at least 3 domains

    def _assess_query_complexity(self, query: str) -> float:
        """Assess query complexity (0-1 scale)"""
        complexity_indicators = [
            len(query) > 100,  # Long query
            any(word in query.lower() for word in ["design", "create", "develop", "analyze"]),
            any(word in query.lower() for word in ["complex", "advanced", "sophisticated"]),
            "?" in query,  # Question format
            len(query.split()) > 20  # Many words
        ]
        return sum(complexity_indicators) / len(complexity_indicators)

    def _assess_domain_novelty(self, query: str, response: str) -> float:
        """Assess domain novelty"""
        # Simple novelty check - could be enhanced
        query_domains = self._count_domains_covered(query)
        response_domains = self._count_domains_covered(response)
        return min(1.0, response_domains / max(1, query_domains))

    def _count_domains_covered(self, text: str) -> int:
        """Count domains covered in text"""
        domains = {
            "mathematics": ["math", "equation", "calculation"],
            "physics": ["physics", "force", "energy"],
            "biology": ["biology", "cell", "organism"],
            "computation": ["computer", "software", "algorithm"],
            "art": ["art", "design", "aesthetic"],
            "psychology": ["psychology", "mind", "behavior"],
            "systems": ["system", "network", "complexity"]
        }

        covered = 0
        text_lower = text.lower()
        for domain, keywords in domains.items():
            if any(keyword in text_lower for keyword in keywords):
                covered += 1

        return covered

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        return {
            "performance_metrics": asdict(self.performance_metrics),
            "current_weights": asdict(self.heuristic_weights),
            "recent_evaluations": [asdict(e) for e in self.response_history[-10:]],
            "learning_trends": self._analyze_learning_trends(),
            "recommendations": self._generate_improvement_recommendations()
        }

    def _analyze_learning_trends(self) -> Dict[str, Any]:
        """Analyze learning trends over time"""
        if len(self.response_history) < 10:
            return {"insufficient_data": True}

        # Group by time periods
        evaluations_by_day = {}
        for eval in self.response_history[-100:]:  # Last 100 evaluations
            day = eval.timestamp[:10]  # YYYY-MM-DD
            if day not in evaluations_by_day:
                evaluations_by_day[day] = []
            evaluations_by_day[day].append(eval.composite_score)

        # Calculate daily averages
        daily_averages = {}
        for day, scores in evaluations_by_day.items():
            daily_averages[day] = statistics.mean(scores)

        # Calculate trend
        if len(daily_averages) >= 3:
            sorted_days = sorted(daily_averages.keys())
            recent_scores = [daily_averages[day] for day in sorted_days[-7:]]  # Last week

            if len(recent_scores) >= 2:
                trend = statistics.linear_regression(range(len(recent_scores)), recent_scores)
                slope = trend.slope if hasattr(trend, 'slope') else 0
            else:
                slope = 0
        else:
            slope = 0

        return {
            "daily_averages": daily_averages,
            "overall_trend": "improving" if slope > 0.001 else "stable" if slope > -0.001 else "declining",
            "trend_slope": slope,
            "data_points": len(daily_averages)
        }

    def _generate_improvement_recommendations(self) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []

        # Analyze performance metrics
        if self.performance_metrics.average_composite_score < 0.7:
            recommendations.append("Overall performance below target. Consider increasing learning rate.")

        if self.performance_metrics.bias_detection_rate > 0.3:
            recommendations.append("High bias detection rate. Review coherence evaluation criteria.")

        if self.performance_metrics.ethical_alignment_rate < 0.95:
            recommendations.append("Ethical alignment below target. Enhance safety checks.")

        if self.performance_metrics.improvement_rate < 0:
            recommendations.append("Performance declining. Consider resetting weights or adjusting learning rate.")

        # Analyze weight distributions
        depth_weights = list(self.heuristic_weights.depth_selection_weights.values())
        if max(depth_weights) / min(depth_weights) > 3:
            recommendations.append("Large weight disparities detected. Consider weight normalization.")

        if not recommendations:
            recommendations.append("System performing well. Continue monitoring for optimization opportunities.")

        return recommendations

    def reset_weights(self):
        """Reset heuristic weights to defaults"""
        self.heuristic_weights = HeuristicWeights()
        logger.info("Heuristic weights reset to defaults")

    def adjust_learning_rate(self, new_rate: float):
        """Adjust the learning rate"""
        old_rate = self.learning_rate
        self.learning_rate = max(0.01, min(0.5, new_rate))  # Clamp to reasonable range
        logger.info(f"Learning rate adjusted from {old_rate} to {self.learning_rate}")

# Global reinforcement loop instance
_reinforcement_loop = None

def get_reinforcement_loop() -> ReinforcementLoop:
    """Get the global reinforcement loop instance"""
    global _reinforcement_loop
    if _reinforcement_loop is None:
        _reinforcement_loop = ReinforcementLoop()
    return _reinforcement_loop

async def evaluate_and_learn(response_id: str,
                           query: str,
                           response: str,
                           reasoning_depth: str,
                           output_style: str,
                           session_id: str = "") -> Dict[str, Any]:
    """Convenience function for response evaluation and learning"""
    loop = get_reinforcement_loop()
    evaluation = await loop.evaluate_response(
        response_id, query, response, reasoning_depth, output_style, session_id
    )
    return {
        "evaluation": asdict(evaluation),
        "performance_report": loop.get_performance_report()
    }