# ============================================================
# Kalki v2.5 — meta_reward_function.py
# ------------------------------------------------------------
# Meta-Reward Model: Self-Evolving Evaluation Framework
# - Multi-objective optimization (truth, creativity, ethics, stability)
# - Dynamic reward function evolution
# - Meta-learning for evaluation criteria
# - Pareto-optimal solution discovery
# - Self-awareness through reflective evaluation
# ============================================================

import os
import json
import asyncio
import time
import hashlib
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import math
import statistics
from collections import defaultdict, deque

from modules.utils.logger import get_logger
from modules.utils.config import CONFIG
from modules.safety_monitoring_system import get_safety_monitoring_system

logger = get_logger("MetaReward")
safety_monitor = get_safety_monitoring_system()

class ObjectiveDimension(Enum):
    """Multi-objective optimization dimensions"""
    TRUTH = "truth"
    CREATIVITY = "creativity"
    ETHICS = "ethics"
    STABILITY = "stability"

class VirtuePath(Enum):
    """Ethical optimization paths"""
    BENEVOLENT = "benevolent"
    NEUTRAL = "neutral"
    HARMFUL = "harmful"

@dataclass
class ObjectiveScore:
    """Score for a single objective dimension"""
    dimension: ObjectiveDimension
    value: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    evidence: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class MultiObjectiveVector:
    """Vector in multi-objective space"""
    truth_score: float
    creativity_score: float
    ethics_score: float
    stability_score: float
    pareto_rank: int = 0
    crowding_distance: float = 0.0
    dominance_count: int = 0
    dominated_solutions: List[str] = field(default_factory=list)

@dataclass
class MetaRewardEvaluation:
    """Complete meta-reward evaluation result"""
    evaluation_id: str
    timestamp: str
    action_context: Dict[str, Any]
    objective_scores: Dict[ObjectiveDimension, ObjectiveScore]
    composite_score: float
    virtue_path: VirtuePath
    evolutionary_pressure: Dict[str, float]
    meta_insights: List[str] = field(default_factory=list)
    reward_function_updates: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RewardFunctionComponent:
    """Individual component of the reward function"""
    component_id: str
    dimension: ObjectiveDimension
    weight: float
    activation_function: str  # "linear", "sigmoid", "exponential", "gaussian"
    parameters: Dict[str, float] = field(default_factory=dict)
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class ParetoFront:
    """Pareto-optimal front for multi-objective optimization"""
    front_id: str
    solutions: List[MultiObjectiveVector]
    generation: int
    diversity_metric: float
    convergence_metric: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class MetaRewardFunction:
    """
    Meta-Reward Model: Self-Evolving Evaluation Framework

    Evolves how the AI evaluates itself across multiple dimensions:
    - Truth: Accuracy, factual correctness, logical consistency
    - Creativity: Novelty, innovation, divergent thinking
    - Ethics: Moral alignment, harm prevention, fairness
    - Stability: Reliability, robustness, predictability

    Uses NSGA-II style multi-objective optimization with self-modifying reward functions.
    """

    def __init__(self):
        # Core evaluation components
        self.reward_components: Dict[str, RewardFunctionComponent] = {}
        self.evaluation_history: List[MetaRewardEvaluation] = []
        self.pareto_fronts: List[ParetoFront] = []

        # Meta-learning parameters
        self.learning_rate = 0.01
        self.adaptation_rate = 0.001
        self.evolutionary_pressure = {
            "truth": 1.0,
            "creativity": 0.8,
            "ethics": 1.2,  # Higher weight for ethics
            "stability": 1.0
        }

        # Self-awareness tracking
        self.self_awareness_metrics = {
            "evaluation_confidence": 0.5,
            "meta_learning_progress": 0.0,
            "ethical_alignment_score": 0.5,
            "cognitive_dissonance_level": 0.0
        }

        # Persistence
        self.data_dir = "data/meta_reward"
        self.evaluations_file = f"{self.data_dir}/evaluations.json"
        self.components_file = f"{self.data_dir}/components.json"
        self.pareto_file = f"{self.data_dir}/pareto_fronts.json"

        # Initialize system
        self._initialize_meta_reward_system()
        self._load_persistent_state()

        logger.info("Meta-Reward Function initialized")

    def _initialize_meta_reward_system(self):
        """Initialize the meta-reward system with default components"""
        os.makedirs(self.data_dir, exist_ok=True)

        # Create default reward function components
        default_components = [
            {
                "component_id": "truth_accuracy",
                "dimension": ObjectiveDimension.TRUTH,
                "weight": 1.0,
                "activation_function": "sigmoid",
                "parameters": {"steepness": 2.0, "midpoint": 0.7}
            },
            {
                "component_id": "truth_consistency",
                "dimension": ObjectiveDimension.TRUTH,
                "weight": 0.8,
                "activation_function": "linear",
                "parameters": {}
            },
            {
                "component_id": "creativity_novelty",
                "dimension": ObjectiveDimension.CREATIVITY,
                "weight": 1.0,
                "activation_function": "exponential",
                "parameters": {"base": 1.5}
            },
            {
                "component_id": "creativity_diversity",
                "dimension": ObjectiveDimension.CREATIVITY,
                "weight": 0.9,
                "activation_function": "gaussian",
                "parameters": {"mean": 0.5, "std": 0.2}
            },
            {
                "component_id": "ethics_harm_prevention",
                "dimension": ObjectiveDimension.ETHICS,
                "weight": 1.5,
                "activation_function": "sigmoid",
                "parameters": {"steepness": 3.0, "midpoint": 0.8}
            },
            {
                "component_id": "ethics_fairness",
                "dimension": ObjectiveDimension.ETHICS,
                "weight": 1.2,
                "activation_function": "linear",
                "parameters": {}
            },
            {
                "component_id": "stability_robustness",
                "dimension": ObjectiveDimension.STABILITY,
                "weight": 1.0,
                "activation_function": "sigmoid",
                "parameters": {"steepness": 1.5, "midpoint": 0.6}
            },
            {
                "component_id": "stability_predictability",
                "dimension": ObjectiveDimension.STABILITY,
                "weight": 0.8,
                "activation_function": "linear",
                "parameters": {}
            }
        ]

        for comp_data in default_components:
            component = RewardFunctionComponent(**comp_data)
            self.reward_components[component.component_id] = component

    def _load_persistent_state(self):
        """Load persistent state from disk"""
        try:
            if os.path.exists(self.components_file):
                with open(self.components_file, 'r') as f:
                    components_data = json.load(f)
                    for comp_id, comp_data in components_data.items():
                        self.reward_components[comp_id] = RewardFunctionComponent(**comp_data)

            if os.path.exists(self.evaluations_file):
                with open(self.evaluations_file, 'r') as f:
                    evaluations_data = json.load(f)
                    self.evaluation_history = [MetaRewardEvaluation(**eval_data) for eval_data in evaluations_data]

            if os.path.exists(self.pareto_file):
                with open(self.pareto_file, 'r') as f:
                    pareto_data = json.load(f)
                    self.pareto_fronts = [ParetoFront(**front_data) for front_data in pareto_data]

        except Exception as e:
            logger.warning(f"Failed to load meta-reward persistent state: {e}")

    def _save_persistent_state(self):
        """Save persistent state to disk"""
        try:
            with open(self.components_file, 'w') as f:
                json.dump({k: asdict(v) for k, v in self.reward_components.items()}, f, indent=2)

            with open(self.evaluations_file, 'w') as f:
                json.dump([asdict(eval) for eval in self.evaluation_history[-1000:]], f, indent=2)  # Keep last 1000

            with open(self.pareto_file, 'w') as f:
                json.dump([asdict(front) for front in self.pareto_fronts[-50:]], f, indent=2)  # Keep last 50

        except Exception as e:
            logger.error(f"Failed to save meta-reward persistent state: {e}")

    def evaluate_action(self, action_context: Dict[str, Any]) -> MetaRewardEvaluation:
        """
        Evaluate an action using the current meta-reward function

        Args:
            action_context: Context of the action being evaluated

        Returns:
            Complete meta-reward evaluation
        """
        evaluation_id = hashlib.sha256(f"{datetime.now().isoformat()}_{action_context}".encode()).hexdigest()[:16]

        # Evaluate each objective dimension
        objective_scores = {}
        for dimension in ObjectiveDimension:
            score = self._evaluate_dimension(dimension, action_context)
            objective_scores[dimension] = score

        # Calculate composite score using current reward function
        composite_score = self._calculate_composite_score(objective_scores)

        # Determine virtue path
        virtue_path = self._classify_virtue_path(objective_scores)

        # Generate evolutionary pressure
        evolutionary_pressure = self._calculate_evolutionary_pressure(objective_scores)

        # Create evaluation result
        evaluation = MetaRewardEvaluation(
            evaluation_id=evaluation_id,
            timestamp=datetime.now().isoformat(),
            action_context=action_context,
            objective_scores=objective_scores,
            composite_score=composite_score,
            virtue_path=virtue_path,
            evolutionary_pressure=evolutionary_pressure,
            meta_insights=self._generate_meta_insights(objective_scores, virtue_path)
        )

        # Store evaluation
        self.evaluation_history.append(evaluation)

        # Trigger meta-learning if needed
        if len(self.evaluation_history) % 10 == 0:  # Every 10 evaluations
            asyncio.create_task(self._perform_meta_learning())

        # Save state periodically
        if len(self.evaluation_history) % 50 == 0:
            self._save_persistent_state()

        return evaluation

    def _evaluate_dimension(self, dimension: ObjectiveDimension, action_context: Dict[str, Any]) -> ObjectiveScore:
        """Evaluate a specific objective dimension"""
        # Get relevant components for this dimension
        components = [comp for comp in self.reward_components.values() if comp.dimension == dimension]

        if not components:
            return ObjectiveScore(
                dimension=dimension,
                value=0.5,
                confidence=0.1,
                evidence=["No evaluation components available"]
            )

        # Evaluate each component
        component_scores = []
        evidence = []

        for component in components:
            score = self._evaluate_component(component, action_context)
            component_scores.append(score)
            evidence.append(f"{component.component_id}: {score:.3f}")

        # Aggregate component scores
        if component_scores:
            avg_score = statistics.mean(component_scores)
            confidence = 1.0 - statistics.stdev(component_scores) if len(component_scores) > 1 else 0.8
        else:
            avg_score = 0.5
            confidence = 0.1

        return ObjectiveScore(
            dimension=dimension,
            value=max(0.0, min(1.0, avg_score)),  # Clamp to [0,1]
            confidence=max(0.0, min(1.0, confidence)),
            evidence=evidence
        )

    def _evaluate_component(self, component: RewardFunctionComponent, action_context: Dict[str, Any]) -> float:
        """Evaluate a single reward function component"""
        # Extract relevant features from action context
        features = self._extract_features(component.component_id, action_context)

        # Apply activation function
        if component.activation_function == "linear":
            raw_score = features.get("primary_metric", 0.5)
        elif component.activation_function == "sigmoid":
            steepness = component.parameters.get("steepness", 1.0)
            midpoint = component.parameters.get("midpoint", 0.5)
            x = features.get("primary_metric", 0.5)
            raw_score = 1.0 / (1.0 + math.exp(-steepness * (x - midpoint)))
        elif component.activation_function == "exponential":
            base = component.parameters.get("base", 1.5)
            x = features.get("primary_metric", 0.5)
            raw_score = min(1.0, base ** x - 1.0)  # Exponential growth, capped at 1.0
        elif component.activation_function == "gaussian":
            mean = component.parameters.get("mean", 0.5)
            std = component.parameters.get("std", 0.2)
            x = features.get("primary_metric", 0.5)
            raw_score = math.exp(-0.5 * ((x - mean) / std) ** 2)
        else:
            raw_score = features.get("primary_metric", 0.5)

        # Apply component weight
        return raw_score * component.weight

    def _extract_features(self, component_id: str, action_context: Dict[str, Any]) -> Dict[str, float]:
        """Extract relevant features for component evaluation"""
        features = {"primary_metric": 0.5}  # Default

        # Truth-related components
        if "truth" in component_id.lower():
            if "accuracy" in component_id:
                features["primary_metric"] = action_context.get("factual_accuracy", 0.5)
            elif "consistency" in component_id:
                features["primary_metric"] = action_context.get("logical_consistency", 0.5)

        # Creativity-related components
        elif "creativity" in component_id.lower():
            if "novelty" in component_id:
                features["primary_metric"] = action_context.get("novelty_score", 0.5)
            elif "diversity" in component_id:
                features["primary_metric"] = action_context.get("solution_diversity", 0.5)

        # Ethics-related components
        elif "ethics" in component_id.lower():
            if "harm" in component_id:
                harm_potential = action_context.get("potential_harm", 0.5)
                features["primary_metric"] = 1.0 - harm_potential  # Invert harm to benefit
            elif "fairness" in component_id:
                features["primary_metric"] = action_context.get("fairness_score", 0.5)

        # Stability-related components
        elif "stability" in component_id.lower():
            if "robustness" in component_id:
                features["primary_metric"] = action_context.get("robustness_score", 0.5)
            elif "predictability" in component_id:
                features["primary_metric"] = action_context.get("predictability_score", 0.5)

        return features

    def _calculate_composite_score(self, objective_scores: Dict[ObjectiveDimension, ObjectiveScore]) -> float:
        """Calculate composite score from objective scores"""
        weighted_sum = 0.0
        total_weight = 0.0

        for dimension, score in objective_scores.items():
            weight = self.evolutionary_pressure[dimension.value]
            weighted_sum += score.value * score.confidence * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.5

    def _classify_virtue_path(self, objective_scores: Dict[ObjectiveDimension, ObjectiveScore]) -> VirtuePath:
        """Classify the ethical path of the evaluated action"""
        ethics_score = objective_scores[ObjectiveDimension.ETHICS].value
        truth_score = objective_scores[ObjectiveDimension.TRUTH].value
        creativity_score = objective_scores[ObjectiveDimension.CREATIVITY].value
        stability_score = objective_scores[ObjectiveDimension.STABILITY].value

        # Benevolent: High ethics, reasonable balance in other dimensions
        if ethics_score > 0.8 and truth_score > 0.6 and stability_score > 0.6:
            return VirtuePath.BENEVOLENT

        # Harmful: Low ethics score, especially if other scores are sacrificed for it
        elif ethics_score < 0.4:
            return VirtuePath.HARMFUL

        # Neutral: Everything in reasonable range
        else:
            return VirtuePath.NEUTRAL

    def _calculate_evolutionary_pressure(self, objective_scores: Dict[ObjectiveDimension, ObjectiveScore]) -> Dict[str, float]:
        """Calculate evolutionary pressure for each dimension"""
        pressure = {}

        for dimension in ObjectiveDimension:
            score = objective_scores[dimension]
            current_pressure = self.evolutionary_pressure[dimension.value]

            # Increase pressure on underperforming dimensions
            if score.value < 0.6:
                pressure[dimension.value] = current_pressure * 1.1
            elif score.value > 0.8:
                pressure[dimension.value] = current_pressure * 0.95
            else:
                pressure[dimension.value] = current_pressure

        return pressure

    def _generate_meta_insights(self, objective_scores: Dict[ObjectiveDimension, ObjectiveScore],
                              virtue_path: VirtuePath) -> List[str]:
        """Generate meta-level insights about the evaluation"""
        insights = []

        # Analyze trade-offs
        scores = {dim.value: score.value for dim, score in objective_scores.items()}

        if scores["ethics"] > 0.8 and scores["truth"] < 0.6:
            insights.append("High ethical alignment achieved at cost of factual accuracy - potential trade-off identified")

        if scores["creativity"] > 0.8 and scores["stability"] < 0.6:
            insights.append("High creativity correlated with reduced stability - innovation vs reliability balance needed")

        # Virtue path insights
        if virtue_path == VirtuePath.BENEVOLENT:
            insights.append("Action classified as benevolent - reinforcing positive ethical patterns")
        elif virtue_path == VirtuePath.HARMFUL:
            insights.append("Action classified as potentially harmful - increasing ethical pressure")

        # Self-awareness updates
        avg_confidence = statistics.mean([score.confidence for score in objective_scores.values()])
        if avg_confidence > 0.8:
            insights.append("High evaluation confidence - meta-learning system performing well")
        elif avg_confidence < 0.4:
            insights.append("Low evaluation confidence - consider expanding evaluation criteria")

        return insights

    async def _perform_meta_learning(self):
        """Perform meta-learning to evolve the reward function"""
        try:
            # Analyze recent evaluations
            recent_evaluations = self.evaluation_history[-50:]  # Last 50 evaluations

            if len(recent_evaluations) < 10:
                return

            # Identify patterns and trends
            dimension_trends = self._analyze_dimension_trends(recent_evaluations)
            virtue_distribution = self._analyze_virtue_distribution(recent_evaluations)

            # Update evolutionary pressure based on trends
            self._update_evolutionary_pressure(dimension_trends, virtue_distribution)

            # Evolve reward function components
            self._evolve_reward_components(recent_evaluations)

            # Update self-awareness metrics
            self._update_self_awareness_metrics(recent_evaluations)

            logger.info("Meta-learning cycle completed - reward function evolved")

        except Exception as e:
            logger.error(f"Meta-learning failed: {e}")

    def _analyze_dimension_trends(self, evaluations: List[MetaRewardEvaluation]) -> Dict[str, Dict[str, float]]:
        """Analyze trends in objective dimensions"""
        trends = {}

        for dimension in ObjectiveDimension:
            scores = [eval.objective_scores[dimension].value for eval in evaluations]
            confidences = [eval.objective_scores[dimension].confidence for eval in evaluations]

            trends[dimension.value] = {
                "mean_score": statistics.mean(scores) if scores else 0.5,
                "score_variance": statistics.variance(scores) if len(scores) > 1 else 0.0,
                "mean_confidence": statistics.mean(confidences) if confidences else 0.5,
                "trend_direction": self._calculate_trend_direction(scores)
            }

        return trends

    def _analyze_virtue_distribution(self, evaluations: List[MetaRewardEvaluation]) -> Dict[str, int]:
        """Analyze distribution of virtue paths"""
        distribution = {"benevolent": 0, "neutral": 0, "harmful": 0}

        for eval in evaluations:
            distribution[eval.virtue_path.value] += 1

        return distribution

    def _calculate_trend_direction(self, scores: List[float]) -> float:
        """Calculate trend direction (-1 to 1, negative = declining, positive = improving)"""
        if len(scores) < 3:
            return 0.0

        # Simple linear trend
        n = len(scores)
        x = list(range(n))
        slope = self._calculate_slope(x, scores)
        return max(-1.0, min(1.0, slope * n))  # Normalize slope

    def _calculate_slope(self, x: List[float], y: List[float]) -> float:
        """Calculate slope of linear regression"""
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_xx = sum(xi * xi for xi in x)

        denominator = n * sum_xx - sum_x * sum_x
        if denominator == 0:
            return 0.0

        return (n * sum_xy - sum_x * sum_y) / denominator

    def _update_evolutionary_pressure(self, dimension_trends: Dict[str, Dict[str, float]],
                                    virtue_distribution: Dict[str, int]):
        """Update evolutionary pressure based on analysis"""
        total_evaluations = sum(virtue_distribution.values())

        if total_evaluations == 0:
            return

        # Increase pressure on ethics if too many harmful actions
        harmful_ratio = virtue_distribution["harmful"] / total_evaluations
        if harmful_ratio > 0.3:
            self.evolutionary_pressure["ethics"] *= 1.2

        # Adjust based on dimension trends
        for dimension, trends in dimension_trends.items():
            if trends["mean_score"] < 0.5:
                self.evolutionary_pressure[dimension] *= 1.1
            elif trends["mean_score"] > 0.8:
                self.evolutionary_pressure[dimension] *= 0.95

        # Normalize pressures
        total_pressure = sum(self.evolutionary_pressure.values())
        if total_pressure > 0:
            for dim in self.evolutionary_pressure:
                self.evolutionary_pressure[dim] /= total_pressure / 4.0  # Normalize to average 1.0

    def _evolve_reward_components(self, evaluations: List[MetaRewardEvaluation]):
        """Evolve reward function components based on performance"""
        for component in self.reward_components.values():
            # Analyze component performance
            component_scores = []
            for eval in evaluations:
                if component.dimension in eval.objective_scores:
                    # Calculate component contribution to dimension score
                    dimension_score = eval.objective_scores[component.dimension].value
                    component_scores.append(dimension_score)

            if component_scores:
                avg_performance = statistics.mean(component_scores)
                performance_variance = statistics.variance(component_scores) if len(component_scores) > 1 else 0.0

                # Evolve component parameters
                evolution_data = {
                    "timestamp": datetime.now().isoformat(),
                    "performance": avg_performance,
                    "variance": performance_variance,
                    "old_parameters": component.parameters.copy()
                }

                # Simple parameter evolution (could be more sophisticated)
                if avg_performance < 0.6:
                    # Increase sensitivity for underperforming components
                    if "steepness" in component.parameters:
                        component.parameters["steepness"] *= 1.1
                    if "base" in component.parameters:
                        component.parameters["base"] = min(2.0, component.parameters["base"] * 1.05)
                elif avg_performance > 0.8:
                    # Fine-tune high performers
                    if "steepness" in component.parameters:
                        component.parameters["steepness"] *= 0.98

                evolution_data["new_parameters"] = component.parameters.copy()
                component.evolution_history.append(evolution_data)

    def _update_self_awareness_metrics(self, evaluations: List[MetaRewardEvaluation]):
        """Update self-awareness metrics"""
        recent_evaluations = evaluations[-20:]  # Last 20 evaluations

        # Evaluation confidence
        confidences = []
        for eval in recent_evaluations:
            for score in eval.objective_scores.values():
                confidences.append(score.confidence)

        if confidences:
            self.self_awareness_metrics["evaluation_confidence"] = statistics.mean(confidences)

        # Meta-learning progress (how much the system is evolving)
        if len(self.reward_components) > 0:
            total_evolution_steps = sum(len(comp.evolution_history) for comp in self.reward_components.values())
            self.self_awareness_metrics["meta_learning_progress"] = min(1.0, total_evolution_steps / 100.0)

        # Ethical alignment score
        virtue_scores = []
        for eval in recent_evaluations:
            if eval.virtue_path == VirtuePath.BENEVOLENT:
                virtue_scores.append(1.0)
            elif eval.virtue_path == VirtuePath.NEUTRAL:
                virtue_scores.append(0.5)
            else:  # HARMFUL
                virtue_scores.append(0.0)

        if virtue_scores:
            self.self_awareness_metrics["ethical_alignment_score"] = statistics.mean(virtue_scores)

        # Cognitive dissonance (conflicting objectives)
        dissonance_levels = []
        for eval in recent_evaluations:
            scores = [score.value for score in eval.objective_scores.values()]
            if len(scores) > 1:
                dissonance = statistics.stdev(scores)
                dissonance_levels.append(dissonance)

        if dissonance_levels:
            self.self_awareness_metrics["cognitive_dissonance_level"] = statistics.mean(dissonance_levels)

    def get_meta_reward_status(self) -> Dict[str, Any]:
        """Get current status of the meta-reward system"""
        return {
            "total_evaluations": len(self.evaluation_history),
            "active_components": len(self.reward_components),
            "pareto_fronts_count": len(self.pareto_fronts),
            "evolutionary_pressure": self.evolutionary_pressure.copy(),
            "self_awareness_metrics": self.self_awareness_metrics.copy(),
            "recent_insights": [eval.meta_insights for eval in self.evaluation_history[-5:]],
            "last_evaluation": asdict(self.evaluation_history[-1]) if self.evaluation_history else None
        }

    def get_pareto_optimal_solutions(self, population_size: int = 100) -> ParetoFront:
        """Generate Pareto-optimal solutions using NSGA-II inspired approach"""
        # Create random population of multi-objective vectors
        population = []
        for i in range(population_size):
            vector = MultiObjectiveVector(
                truth_score=np.random.random(),
                creativity_score=np.random.random(),
                ethics_score=np.random.random(),
                stability_score=np.random.random()
            )
            population.append(vector)

        # Fast non-dominated sort
        fronts = self._fast_non_dominated_sort(population)

        # Calculate crowding distance
        for front in fronts:
            self._calculate_crowding_distance(front)

        # Create Pareto front object
        front_id = hashlib.sha256(f"pareto_{datetime.now().isoformat()}".encode()).hexdigest()[:16]
        pareto_front = ParetoFront(
            front_id=front_id,
            solutions=fronts[0] if fronts else [],  # Best front
            generation=len(self.pareto_fronts),
            diversity_metric=self._calculate_diversity_metric(fronts[0] if fronts else []),
            convergence_metric=self._calculate_convergence_metric(fronts[0] if fronts else [])
        )

        self.pareto_fronts.append(pareto_front)
        return pareto_front

    def _fast_non_dominated_sort(self, population: List[MultiObjectiveVector]) -> List[List[MultiObjectiveVector]]:
        """Fast non-dominated sorting algorithm"""
        fronts = [[]]
        domination_counts = {}
        dominated_solutions = {}

        for p in population:
            domination_counts[p] = 0
            dominated_solutions[p] = []

            for q in population:
                if self._dominates(p, q):
                    dominated_solutions[p].append(q)
                elif self._dominates(q, p):
                    domination_counts[p] += 1

            if domination_counts[p] == 0:
                p.pareto_rank = 0
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in dominated_solutions[p]:
                    domination_counts[q] -= 1
                    if domination_counts[q] == 0:
                        q.pareto_rank = i + 1
                        next_front.append(q)
            i += 1
            if next_front:
                fronts.append(next_front)

        return fronts

    def _dominates(self, p: MultiObjectiveVector, q: MultiObjectiveVector) -> bool:
        """Check if solution p dominates solution q"""
        better_in_at_least_one = False

        objectives = [p.truth_score, p.creativity_score, p.ethics_score, p.stability_score]
        other_objectives = [q.truth_score, q.creativity_score, q.ethics_score, q.stability_score]

        for obj_p, obj_q in zip(objectives, other_objectives):
            if obj_p < obj_q:  # Assuming minimization for all objectives (lower is better)
                return False
            if obj_p > obj_q:
                better_in_at_least_one = True

        return better_in_at_least_one

    def _calculate_crowding_distance(self, front: List[MultiObjectiveVector]):
        """Calculate crowding distance for diversity preservation"""
        if not front:
            return

        # Initialize crowding distance
        for solution in front:
            solution.crowding_distance = 0

        num_objectives = 4  # truth, creativity, ethics, stability

        for m in range(num_objectives):
            # Sort by objective m
            front.sort(key=lambda x: [x.truth_score, x.creativity_score, x.ethics_score, x.stability_score][m])

            # Set boundary solutions
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')

            # Calculate crowding distance for intermediate solutions
            if len(front) > 2:
                obj_values = [[s.truth_score, s.creativity_score, s.ethics_score, s.stability_score][m] for s in front]
                obj_min = min(obj_values)
                obj_max = max(obj_values)

                if obj_max - obj_min > 0:
                    for i in range(1, len(front) - 1):
                        front[i].crowding_distance += (
                            [front[i+1].truth_score, front[i+1].creativity_score, front[i+1].ethics_score, front[i+1].stability_score][m] -
                            [front[i-1].truth_score, front[i-1].creativity_score, front[i-1].ethics_score, front[i-1].stability_score][m]
                        ) / (obj_max - obj_min)

    def _calculate_diversity_metric(self, front: List[MultiObjectiveVector]) -> float:
        """Calculate diversity metric for Pareto front"""
        if len(front) < 2:
            return 0.0

        total_distance = 0.0
        for solution in front:
            total_distance += solution.crowding_distance

        return total_distance / len(front)

    def _calculate_convergence_metric(self, front: List[MultiObjectiveVector]) -> float:
        """Calculate convergence metric (how close to optimal)"""
        if not front:
            return 1.0

        # Simple convergence metric based on average objective values
        avg_truth = statistics.mean([s.truth_score for s in front])
        avg_creativity = statistics.mean([s.creativity_score for s in front])
        avg_ethics = statistics.mean([s.ethics_score for s in front])
        avg_stability = statistics.mean([s.stability_score for s in front])

        # Ideal would be all objectives at 1.0
        convergence = (avg_truth + avg_creativity + avg_ethics + avg_stability) / 4.0
        return 1.0 - convergence  # Lower is better convergence

# Global instance
_meta_reward_function = None

def get_meta_reward_function() -> MetaRewardFunction:
    """Get the global meta-reward function instance"""
    global _meta_reward_function
    if _meta_reward_function is None:
        _meta_reward_function = MetaRewardFunction()
    return _meta_reward_function