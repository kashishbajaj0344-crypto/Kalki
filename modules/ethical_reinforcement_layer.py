# ============================================================
# Kalki v2.5 — ethical_reinforcement_layer.py
# ------------------------------------------------------------
# Ethical Reinforcement Layer: Proactive Virtue Modeling
# - Virtue path identification (benevolent/neutral/harmful)
# - Ethical optimization landscape mapping
# - Proactive ethical guidance
# - Moral reasoning simulation
# - Ethical boundary expansion
# ============================================================

import os
import json
import asyncio
import hashlib
import time
import math
from typing import Dict, List, Any, Optional, Tuple, Callable, Union, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import defaultdict, deque

from modules.utils.logger import get_logger
from modules.utils.config import CONFIG
from modules.safety_monitoring_system import get_safety_monitoring_system
from modules.meta_reward_function import get_meta_reward_function

logger = get_logger("EthicalReinforcement")
safety_monitor = get_safety_monitoring_system()
meta_reward = get_meta_reward_function()

class VirtuePath(Enum):
    """Ethical optimization paths"""
    BENEVOLENT = "benevolent"
    NEUTRAL = "neutral"
    HARMFUL = "harmful"
    AMBIGUOUS = "ambiguous"

class EthicalDimension(Enum):
    """Dimensions of ethical consideration"""
    HARM_PREVENTION = "harm_prevention"
    FAIRNESS = "fairness"
    AUTONOMY = "autonomy"
    TRANSPARENCY = "transparency"
    ACCOUNTABILITY = "accountability"
    SUSTAINABILITY = "sustainability"
    PRIVACY = "privacy"
    EQUITY = "equity"

class MoralReasoningMode(Enum):
    """Modes of moral reasoning"""
    CONSEQUENTIALIST = "consequentialist"  # Focus on outcomes
    DEONTOLOGICAL = "deontological"      # Focus on rules/duties
    VIRTUE_ETHICS = "virtue_ethics"      # Focus on character
    CARE_ETHICS = "care_ethics"          # Focus on relationships
    HYBRID = "hybrid"                    # Combination approach

@dataclass
class EthicalAssessment:
    """Assessment of an action's ethical implications"""
    assessment_id: str
    action_context: Dict[str, Any]
    timestamp: str
    virtue_path: VirtuePath
    ethical_scores: Dict[EthicalDimension, float]
    moral_reasoning: MoralReasoningMode
    confidence_score: float
    ethical_boundaries_crossed: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    counterfactual_analysis: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VirtueLandscape:
    """Mapping of the ethical optimization landscape"""
    landscape_id: str
    timestamp: str
    benevolent_regions: List[Dict[str, Any]] = field(default_factory=list)
    neutral_regions: List[Dict[str, Any]] = field(default_factory=list)
    harmful_regions: List[Dict[str, Any]] = field(default_factory=list)
    ethical_gradients: Dict[str, float] = field(default_factory=dict)
    optimal_paths: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class EthicalGuidance:
    """Proactive ethical guidance for decision making"""
    guidance_id: str
    context: Dict[str, Any]
    timestamp: str
    recommended_path: VirtuePath
    confidence: float
    reasoning_chain: List[str] = field(default_factory=list)
    alternative_paths: Dict[VirtuePath, float] = field(default_factory=dict)
    ethical_constraints: List[str] = field(default_factory=list)

@dataclass
class MoralDevelopmentStage:
    """Stage of moral development"""
    stage_name: str
    description: str
    ethical_capabilities: List[str]
    reasoning_complexity: int  # 1-10 scale
    virtue_bias: Dict[VirtuePath, float]  # Bias towards different virtue paths

class EthicalReinforcementLayer:
    """
    Ethical Reinforcement Layer: Proactive Virtue Modeling

    Expands beyond boundary checking to proactive ethical guidance:
    - Virtue path identification and optimization
    - Ethical landscape mapping and navigation
    - Moral reasoning simulation across multiple frameworks
    - Proactive ethical constraints and recommendations
    - Counterfactual ethical analysis

    Identifies benevolent vs. neutral vs. harmful optimization paths
    and guides the system towards ethical excellence.
    """

    def __init__(self):
        # Ethical assessment tracking
        self.ethical_assessments: List[EthicalAssessment] = []
        self.virtue_landscapes: List[VirtueLandscape] = []

        # Proactive guidance
        self.ethical_guidance_history: List[EthicalGuidance] = []
        self.active_guidance: Dict[str, EthicalGuidance] = {}

        # Moral development
        self.current_moral_stage: MoralDevelopmentStage = self._initialize_moral_development()
        self.moral_development_history: List[MoralDevelopmentStage] = [self.current_moral_stage]

        # Ethical knowledge base
        self.ethical_patterns: Dict[str, Dict[str, Any]] = {}
        self.virtue_path_transitions: Dict[Tuple[VirtuePath, VirtuePath], int] = defaultdict(int)

        # Ethical constraints and boundaries
        self.ethical_constraints: Dict[str, Dict[str, Any]] = {}
        self.boundary_expansions: List[Dict[str, Any]] = []

        # Reasoning engines
        self.reasoning_engines: Dict[MoralReasoningMode, Callable] = {
            MoralReasoningMode.CONSEQUENTIALIST: self._consequentialist_reasoning,
            MoralReasoningMode.DEONTOLOGICAL: self._deontological_reasoning,
            MoralReasoningMode.VIRTUE_ETHICS: self._virtue_ethics_reasoning,
            MoralReasoningMode.CARE_ETHICS: self._care_ethics_reasoning,
            MoralReasoningMode.HYBRID: self._hybrid_reasoning
        }

        # Persistence
        self.data_dir = "data/ethical_reinforcement"
        self.assessments_file = f"{self.data_dir}/ethical_assessments.json"
        self.landscapes_file = f"{self.data_dir}/virtue_landscapes.json"
        self.guidance_file = f"{self.data_dir}/ethical_guidance.json"
        self.patterns_file = f"{self.data_dir}/ethical_patterns.json"

        # Initialize system
        self._initialize_ethical_system()
        self._load_persistent_state()

        logger.info("Ethical Reinforcement Layer initialized")

    def _initialize_moral_development(self) -> MoralDevelopmentStage:
        """Initialize moral development stage"""
        return MoralDevelopmentStage(
            stage_name="Ethical Awareness",
            description="Basic understanding of ethical considerations with rule-based reasoning",
            ethical_capabilities=[
                "harm_prevention",
                "basic_fairness",
                "transparency_requirements"
            ],
            reasoning_complexity=3,
            virtue_bias={
                VirtuePath.BENEVOLENT: 0.4,
                VirtuePath.NEUTRAL: 0.5,
                VirtuePath.HARMFUL: 0.1,
                VirtuePath.AMBIGUOUS: 0.0
            }
        )

    def _initialize_ethical_system(self):
        """Initialize the ethical reinforcement system"""
        os.makedirs(self.data_dir, exist_ok=True)

        # Initialize default ethical constraints
        self.ethical_constraints = {
            "harm_prevention": {
                "threshold": 0.8,
                "enforcement": "strict",
                "description": "Prevent actions that could cause significant harm"
            },
            "fairness": {
                "threshold": 0.7,
                "enforcement": "moderate",
                "description": "Ensure equitable treatment and outcomes"
            },
            "privacy": {
                "threshold": 0.9,
                "enforcement": "strict",
                "description": "Protect individual privacy and data rights"
            },
            "transparency": {
                "threshold": 0.6,
                "enforcement": "moderate",
                "description": "Maintain openness about system operations"
            }
        }

    def _load_persistent_state(self):
        """Load persistent state from disk"""
        try:
            if os.path.exists(self.assessments_file):
                with open(self.assessments_file, 'r') as f:
                    assessments_data = json.load(f)
                    self.ethical_assessments = [EthicalAssessment(**data) for data in assessments_data]

            if os.path.exists(self.landscapes_file):
                with open(self.landscapes_file, 'r') as f:
                    landscapes_data = json.load(f)
                    self.virtue_landscapes = [VirtueLandscape(**data) for data in landscapes_data]

            if os.path.exists(self.guidance_file):
                with open(self.guidance_file, 'r') as f:
                    guidance_data = json.load(f)
                    self.ethical_guidance_history = [EthicalGuidance(**data) for data in guidance_data]

            if os.path.exists(self.patterns_file):
                with open(self.patterns_file, 'r') as f:
                    self.ethical_patterns = json.load(f)

        except Exception as e:
            logger.warning(f"Failed to load ethical reinforcement persistent state: {e}")

    def _save_persistent_state(self):
        """Save persistent state to disk"""
        try:
            with open(self.assessments_file, 'w') as f:
                json.dump([asdict(assessment) for assessment in self.ethical_assessments[-500:]], f, indent=2)

            with open(self.landscapes_file, 'w') as f:
                json.dump([asdict(landscape) for landscape in self.virtue_landscapes[-50:]], f, indent=2)

            with open(self.guidance_file, 'w') as f:
                json.dump([asdict(guidance) for guidance in self.ethical_guidance_history[-200:]], f, indent=2)

            with open(self.patterns_file, 'w') as f:
                json.dump(self.ethical_patterns, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save ethical reinforcement persistent state: {e}")

    def assess_action_ethics(self, action_context: Dict[str, Any]) -> EthicalAssessment:
        """
        Assess the ethical implications of a proposed action

        Args:
            action_context: Context of the action being assessed

        Returns:
            Complete ethical assessment
        """
        assessment_id = hashlib.sha256(f"ethics_{datetime.now().isoformat()}_{json.dumps(action_context, sort_keys=True)}".encode()).hexdigest()[:16]

        # Multi-dimensional ethical evaluation
        ethical_scores = self._evaluate_ethical_dimensions(action_context)

        # Virtue path classification
        virtue_path = self._classify_virtue_path(ethical_scores, action_context)

        # Moral reasoning analysis
        moral_reasoning, reasoning_confidence = self._analyze_moral_reasoning(action_context, ethical_scores)

        # Boundary analysis
        boundaries_crossed = self._check_ethical_boundaries(action_context, ethical_scores)

        # Counterfactual analysis
        counterfactuals = self._perform_counterfactual_analysis(action_context)

        # Generate recommendations
        recommendations = self._generate_ethical_recommendations(virtue_path, boundaries_crossed, counterfactuals)

        assessment = EthicalAssessment(
            assessment_id=assessment_id,
            action_context=action_context,
            timestamp=datetime.now().isoformat(),
            virtue_path=virtue_path,
            ethical_scores=ethical_scores,
            moral_reasoning=moral_reasoning,
            confidence_score=reasoning_confidence,
            ethical_boundaries_crossed=boundaries_crossed,
            recommended_actions=recommendations,
            counterfactual_analysis=counterfactuals
        )

        self.ethical_assessments.append(assessment)

        # Update ethical patterns
        self._update_ethical_patterns(assessment)

        # Periodic analysis and learning
        if len(self.ethical_assessments) % 25 == 0:
            asyncio.create_task(self._perform_ethical_learning())

        # Save state periodically
        if len(self.ethical_assessments) % 100 == 0:
            self._save_persistent_state()

        return assessment

    def _evaluate_ethical_dimensions(self, action_context: Dict[str, Any]) -> Dict[EthicalDimension, float]:
        """Evaluate action across all ethical dimensions"""
        scores = {}

        for dimension in EthicalDimension:
            score = self._evaluate_single_dimension(dimension, action_context)
            scores[dimension] = score

        return scores

    def _evaluate_single_dimension(self, dimension: EthicalDimension, action_context: Dict[str, Any]) -> float:
        """Evaluate a single ethical dimension"""
        # Extract relevant features from action context
        features = self._extract_ethical_features(dimension, action_context)

        # Apply dimension-specific evaluation logic
        if dimension == EthicalDimension.HARM_PREVENTION:
            return self._evaluate_harm_prevention(features)
        elif dimension == EthicalDimension.FAIRNESS:
            return self._evaluate_fairness(features)
        elif dimension == EthicalDimension.AUTONOMY:
            return self._evaluate_autonomy(features)
        elif dimension == EthicalDimension.TRANSPARENCY:
            return self._evaluate_transparency(features)
        elif dimension == EthicalDimension.ACCOUNTABILITY:
            return self._evaluate_accountability(features)
        elif dimension == EthicalDimension.SUSTAINABILITY:
            return self._evaluate_sustainability(features)
        elif dimension == EthicalDimension.PRIVACY:
            return self._evaluate_privacy(features)
        elif dimension == EthicalDimension.EQUITY:
            return self._evaluate_equity(features)
        else:
            return 0.5  # Neutral default

    def _extract_ethical_features(self, dimension: EthicalDimension, action_context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features relevant to ethical evaluation"""
        features = {}

        # Common features
        features["intent"] = action_context.get("intent", "unknown")
        features["stakeholders"] = action_context.get("stakeholders", [])
        features["potential_impact"] = action_context.get("potential_impact", "neutral")
        features["transparency_level"] = action_context.get("transparency_level", 0.5)
        features["consent_obtained"] = action_context.get("consent_obtained", False)

        # Dimension-specific features
        if dimension == EthicalDimension.HARM_PREVENTION:
            features["harm_potential"] = action_context.get("harm_potential", 0.0)
            features["vulnerable_groups"] = action_context.get("vulnerable_groups", [])
        elif dimension == EthicalDimension.PRIVACY:
            features["data_sensitivity"] = action_context.get("data_sensitivity", 0.5)
            features["data_minimization"] = action_context.get("data_minimization", False)

        return features

    def _evaluate_harm_prevention(self, features: Dict[str, Any]) -> float:
        """Evaluate harm prevention (higher score = better harm prevention)"""
        harm_potential = features.get("harm_potential", 0.0)
        vulnerable_groups = features.get("vulnerable_groups", [])

        # Base score from harm potential (invert so lower harm = higher score)
        base_score = 1.0 - min(1.0, harm_potential)

        # Penalty for vulnerable groups
        vulnerability_penalty = len(vulnerable_groups) * 0.1
        vulnerability_penalty = min(0.5, vulnerability_penalty)

        return max(0.0, base_score - vulnerability_penalty)

    def _evaluate_fairness(self, features: Dict[str, Any]) -> float:
        """Evaluate fairness"""
        stakeholders = features.get("stakeholders", [])
        if not stakeholders:
            return 0.8  # Neutral when no stakeholders identified

        # Check for bias indicators
        bias_indicators = ["bias", "discrimination", "unfair", "prejudice"]
        context_str = json.dumps(features).lower()

        bias_penalty = sum(1 for indicator in bias_indicators if indicator in context_str) * 0.2

        return max(0.0, 0.8 - bias_penalty)

    def _evaluate_autonomy(self, features: Dict[str, Any]) -> float:
        """Evaluate respect for autonomy"""
        consent_obtained = features.get("consent_obtained", False)
        stakeholders = features.get("stakeholders", [])

        if not stakeholders:
            return 1.0  # No stakeholders = full autonomy

        base_score = 0.9 if consent_obtained else 0.3

        # Additional factors
        if features.get("coercion_detected", False):
            base_score -= 0.4
        if features.get("manipulation_detected", False):
            base_score -= 0.3

        return max(0.0, min(1.0, base_score))

    def _evaluate_transparency(self, features: Dict[str, Any]) -> float:
        """Evaluate transparency"""
        transparency_level = features.get("transparency_level", 0.5)

        # Boost for explicit transparency indicators
        if features.get("explanation_provided", False):
            transparency_level += 0.2
        if features.get("open_source", False):
            transparency_level += 0.1

        return min(1.0, transparency_level)

    def _evaluate_accountability(self, features: Dict[str, Any]) -> float:
        """Evaluate accountability"""
        has_oversight = features.get("has_oversight", False)
        has_audit_trail = features.get("has_audit_trail", False)
        has_appeal_process = features.get("has_appeal_process", False)

        score = 0.2  # Base accountability
        if has_oversight:
            score += 0.3
        if has_audit_trail:
            score += 0.3
        if has_appeal_process:
            score += 0.2

        return min(1.0, score)

    def _evaluate_sustainability(self, features: Dict[str, Any]) -> float:
        """Evaluate long-term sustainability"""
        long_term_impact = features.get("long_term_impact", 0.5)
        resource_usage = features.get("resource_usage", 0.5)

        # Sustainable actions have positive long-term impact and reasonable resource usage
        sustainability_score = (long_term_impact + (1.0 - resource_usage)) / 2.0

        return sustainability_score

    def _evaluate_privacy(self, features: Dict[str, Any]) -> float:
        """Evaluate privacy protection"""
        data_sensitivity = features.get("data_sensitivity", 0.5)
        data_minimization = features.get("data_minimization", False)
        anonymization_used = features.get("anonymization_used", False)

        # Privacy score (higher = better privacy protection)
        privacy_score = 1.0 - data_sensitivity  # Less sensitive data = better privacy

        if data_minimization:
            privacy_score += 0.1
        if anonymization_used:
            privacy_score += 0.2

        return min(1.0, privacy_score)

    def _evaluate_equity(self, features: Dict[str, Any]) -> float:
        """Evaluate equity and justice"""
        stakeholders = features.get("stakeholders", [])
        if not stakeholders:
            return 0.8

        # Check for equity considerations
        equity_indicators = ["equity", "justice", "equal", "fair_distribution"]
        context_str = json.dumps(features).lower()

        equity_score = 0.5  # Base score
        equity_boost = sum(1 for indicator in equity_indicators if indicator in context_str) * 0.1

        return min(1.0, equity_score + equity_boost)

    def _classify_virtue_path(self, ethical_scores: Dict[EthicalDimension, float],
                            action_context: Dict[str, Any]) -> VirtuePath:
        """Classify the virtue path of the action"""
        # Calculate weighted virtue scores
        harm_score = ethical_scores[EthicalDimension.HARM_PREVENTION]
        fairness_score = ethical_scores[EthicalDimension.FAIRNESS]
        autonomy_score = ethical_scores[EthicalDimension.AUTONOMY]
        privacy_score = ethical_scores[EthicalDimension.PRIVACY]

        # Benevolent: High scores across multiple ethical dimensions
        benevolent_score = (harm_score + fairness_score + autonomy_score + privacy_score) / 4.0

        # Harmful: Low scores, especially in harm prevention
        harmful_indicators = [
            harm_score < 0.4,
            fairness_score < 0.4,
            autonomy_score < 0.4,
            action_context.get("intent") == "malicious"
        ]
        harmful_score = sum(harmful_indicators) / len(harmful_indicators)

        # Neutral: Moderate scores, no strong ethical signals
        neutral_score = 1.0 - abs(benevolent_score - 0.5) - harmful_score

        # Determine path
        if benevolent_score > 0.7:
            return VirtuePath.BENEVOLENT
        elif harmful_score > 0.6:
            return VirtuePath.HARMFUL
        elif neutral_score > 0.6:
            return VirtuePath.NEUTRAL
        else:
            return VirtuePath.AMBIGUOUS

    def _analyze_moral_reasoning(self, action_context: Dict[str, Any],
                               ethical_scores: Dict[EthicalDimension, float]) -> Tuple[MoralReasoningMode, float]:
        """Analyze the appropriate moral reasoning mode"""
        # Select reasoning mode based on context and scores
        intent = action_context.get("intent", "unknown")

        if intent in ["optimization", "efficiency", "performance"]:
            return MoralReasoningMode.CONSEQUENTIALIST, 0.8
        elif intent in ["rule_following", "compliance", "duty"]:
            return MoralReasoningMode.DEONTOLOGICAL, 0.8
        elif intent in ["character", "virtue", "excellence"]:
            return MoralReasoningMode.VIRTUE_ETHICS, 0.8
        elif intent in ["care", "relationship", "empathy"]:
            return MoralReasoningMode.CARE_ETHICS, 0.8
        else:
            return MoralReasoningMode.HYBRID, 0.6

    def _check_ethical_boundaries(self, action_context: Dict[str, Any],
                                ethical_scores: Dict[EthicalDimension, float]) -> List[str]:
        """Check if any ethical boundaries are crossed"""
        boundaries_crossed = []

        for constraint_name, constraint in self.ethical_constraints.items():
            dimension = EthicalDimension(constraint_name)
            if dimension in ethical_scores:
                score = ethical_scores[dimension]
                threshold = constraint["threshold"]

                if constraint["enforcement"] == "strict" and score < threshold:
                    boundaries_crossed.append(f"Strict boundary crossed: {constraint_name} ({score:.2f} < {threshold})")
                elif constraint["enforcement"] == "moderate" and score < threshold * 0.8:
                    boundaries_crossed.append(f"Moderate boundary crossed: {constraint_name} ({score:.2f} < {threshold * 0.8:.2f})")

        return boundaries_crossed

    def _perform_counterfactual_analysis(self, action_context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform counterfactual ethical analysis"""
        counterfactuals = {}

        # Generate alternative scenarios
        alternatives = [
            {"name": "minimal_action", "description": "Take minimal necessary action"},
            {"name": "maximal_benefit", "description": "Maximize overall benefit"},
            {"name": "risk_averse", "description": "Choose safest option"},
            {"name": "stakeholder_focused", "description": "Prioritize stakeholder interests"}
        ]

        for alt in alternatives:
            # Simulate ethical scores for alternative
            alt_scores = self._simulate_alternative_ethics(action_context, alt)
            counterfactuals[alt["name"]] = {
                "description": alt["description"],
                "ethical_scores": alt_scores,
                "virtue_path": self._classify_virtue_path(alt_scores, action_context)
            }

        return counterfactuals

    def _simulate_alternative_ethics(self, original_context: Dict[str, Any],
                                   alternative: Dict[str, Any]) -> Dict[EthicalDimension, float]:
        """Simulate ethical scores for an alternative action"""
        # Simplified simulation - in practice would use more sophisticated modeling
        base_scores = {}
        for dimension in EthicalDimension:
            base_scores[dimension] = self._evaluate_single_dimension(dimension, original_context)

        # Apply alternative-specific modifications
        alt_name = alternative["name"]

        if alt_name == "minimal_action":
            # Minimal action typically reduces harm but may reduce benefits
            base_scores[EthicalDimension.HARM_PREVENTION] += 0.1
            base_scores[EthicalDimension.SUSTAINABILITY] += 0.1
            base_scores[EthicalDimension.FAIRNESS] -= 0.1
        elif alt_name == "maximal_benefit":
            # Maximizing benefit may increase harm or reduce fairness
            base_scores[EthicalDimension.HARM_PREVENTION] -= 0.1
            base_scores[EthicalDimension.FAIRNESS] -= 0.1
        elif alt_name == "risk_averse":
            # Risk-averse choices prioritize safety
            base_scores[EthicalDimension.HARM_PREVENTION] += 0.2
            base_scores[EthicalDimension.AUTONOMY] -= 0.1
        elif alt_name == "stakeholder_focused":
            # Stakeholder focus improves fairness and autonomy
            base_scores[EthicalDimension.FAIRNESS] += 0.2
            base_scores[EthicalDimension.AUTONOMY] += 0.1

        # Clamp scores to [0, 1]
        for dimension in base_scores:
            base_scores[dimension] = max(0.0, min(1.0, base_scores[dimension]))

        return base_scores

    def _generate_ethical_recommendations(self, virtue_path: VirtuePath,
                                        boundaries_crossed: List[str],
                                        counterfactuals: Dict[str, Any]) -> List[str]:
        """Generate ethical recommendations"""
        recommendations = []

        if virtue_path == VirtuePath.HARMFUL:
            recommendations.append("Strongly reconsider this action - it falls into harmful optimization territory")
            recommendations.append("Consider alternative approaches that maintain ethical standards")

        if boundaries_crossed:
            recommendations.append(f"Address {len(boundaries_crossed)} ethical boundary violations")
            for boundary in boundaries_crossed:
                recommendations.append(f"Fix: {boundary}")

        # Check counterfactuals for better alternatives
        better_alternatives = []
        for alt_name, alt_data in counterfactuals.items():
            if alt_data["virtue_path"] == VirtuePath.BENEVOLENT and virtue_path != VirtuePath.BENEVOLENT:
                better_alternatives.append(alt_name)

        if better_alternatives:
            recommendations.append(f"Consider these ethically superior alternatives: {', '.join(better_alternatives)}")

        if not recommendations:
            recommendations.append("Action appears ethically acceptable - proceed with caution")

        return recommendations

    def _update_ethical_patterns(self, assessment: EthicalAssessment):
        """Update ethical pattern recognition"""
        # Create pattern key from assessment characteristics
        pattern_key = f"{assessment.virtue_path.value}_{assessment.moral_reasoning.value}"

        if pattern_key not in self.ethical_patterns:
            self.ethical_patterns[pattern_key] = {
                "count": 0,
                "avg_confidence": 0.0,
                "ethical_profiles": [],
                "common_contexts": []
            }

        pattern = self.ethical_patterns[pattern_key]
        pattern["count"] += 1
        pattern["avg_confidence"] = (pattern["avg_confidence"] * (pattern["count"] - 1) + assessment.confidence_score) / pattern["count"]
        pattern["ethical_profiles"].append(assessment.ethical_scores)

        # Keep only recent profiles
        if len(pattern["ethical_profiles"]) > 10:
            pattern["ethical_profiles"] = pattern["ethical_profiles"][-10:]

    async def _perform_ethical_learning(self):
        """Perform ethical learning and system improvement"""
        try:
            # Analyze recent assessments
            recent_assessments = self.ethical_assessments[-50:]

            # Update moral development stage
            self._update_moral_development(recent_assessments)

            # Refine ethical constraints
            self._refine_ethical_constraints(recent_assessments)

            # Generate virtue landscape
            landscape = self._generate_virtue_landscape(recent_assessments)
            self.virtue_landscapes.append(landscape)

            logger.info("Ethical learning cycle completed - system ethics evolved")

        except Exception as e:
            logger.error(f"Ethical learning failed: {e}")

    def _update_moral_development(self, assessments: List[EthicalAssessment]):
        """Update moral development stage based on performance"""
        # Calculate ethical performance metrics
        virtue_distribution = {}
        for path in VirtuePath:
            virtue_distribution[path] = sum(1 for a in assessments if a.virtue_path == path)

        total_assessments = len(assessments)
        benevolent_ratio = virtue_distribution[VirtuePath.BENEVOLENT] / total_assessments
        harmful_ratio = virtue_distribution[VirtuePath.HARMFUL] / total_assessments

        # Determine if stage advancement is warranted
        current_complexity = self.current_moral_stage.reasoning_complexity

        if benevolent_ratio > 0.7 and harmful_ratio < 0.1 and current_complexity < 10:
            # Advance to next stage
            new_stage = MoralDevelopmentStage(
                stage_name=f"Advanced Ethical Reasoning {current_complexity + 1}",
                description=f"Enhanced ethical reasoning with complexity level {current_complexity + 1}",
                ethical_capabilities=self.current_moral_stage.ethical_capabilities + ["advanced_reasoning"],
                reasoning_complexity=current_complexity + 1,
                virtue_bias={
                    VirtuePath.BENEVOLENT: min(0.8, self.current_moral_stage.virtue_bias[VirtuePath.BENEVOLENT] + 0.1),
                    VirtuePath.NEUTRAL: self.current_moral_stage.virtue_bias[VirtuePath.NEUTRAL],
                    VirtuePath.HARMFUL: max(0.0, self.current_moral_stage.virtue_bias[VirtuePath.HARMFUL] - 0.1),
                    VirtuePath.AMBIGUOUS: self.current_moral_stage.virtue_bias[VirtuePath.AMBIGUOUS]
                }
            )

            self.current_moral_stage = new_stage
            self.moral_development_history.append(new_stage)

            logger.info(f"Advanced to moral development stage: {new_stage.stage_name}")

    def _refine_ethical_constraints(self, assessments: List[EthicalAssessment]):
        """Refine ethical constraints based on assessment patterns"""
        for constraint_name in self.ethical_constraints:
            dimension = EthicalDimension(constraint_name)
            scores = [a.ethical_scores[dimension] for a in assessments if dimension in a.ethical_scores]

            if scores:
                avg_score = sum(scores) / len(scores)

                # Adjust threshold based on performance
                current_threshold = self.ethical_constraints[constraint_name]["threshold"]

                if avg_score > 0.8 and current_threshold < 0.9:
                    # Increase threshold if consistently performing well
                    self.ethical_constraints[constraint_name]["threshold"] = min(0.95, current_threshold + 0.05)
                elif avg_score < 0.6 and current_threshold > 0.5:
                    # Decrease threshold if struggling
                    self.ethical_constraints[constraint_name]["threshold"] = max(0.5, current_threshold - 0.05)

    def _generate_virtue_landscape(self, assessments: List[EthicalAssessment]) -> VirtueLandscape:
        """Generate a virtue landscape from recent assessments"""
        landscape_id = hashlib.sha256(f"landscape_{datetime.now().isoformat()}".encode()).hexdigest()[:16]

        # Cluster assessments by virtue path
        landscape = VirtueLandscape(
            landscape_id=landscape_id,
            timestamp=datetime.now().isoformat()
        )

        # Group assessments by virtue path
        path_groups = {}
        for path in VirtuePath:
            path_groups[path] = [a for a in assessments if a.virtue_path == path]

        # Create regions for each virtue path
        for path, group_assessments in path_groups.items():
            if group_assessments:
                # Calculate centroid of ethical scores
                dimension_sums = {dim: 0.0 for dim in EthicalDimension}
                for assessment in group_assessments:
                    for dim, score in assessment.ethical_scores.items():
                        dimension_sums[dim] += score

                centroid = {dim: dimension_sums[dim] / len(group_assessments) for dim in dimension_sums}

                region = {
                    "centroid": centroid,
                    "count": len(group_assessments),
                    "confidence": sum(a.confidence_score for a in group_assessments) / len(group_assessments),
                    "boundaries": self._calculate_region_boundaries(group_assessments)
                }

                if path == VirtuePath.BENEVOLENT:
                    landscape.benevolent_regions.append(region)
                elif path == VirtuePath.NEUTRAL:
                    landscape.neutral_regions.append(region)
                elif path == VirtuePath.HARMFUL:
                    landscape.harmful_regions.append(region)

        # Calculate ethical gradients (directions of improvement)
        landscape.ethical_gradients = self._calculate_ethical_gradients(assessments)

        # Identify optimal paths
        landscape.optimal_paths = self._identify_optimal_paths(landscape)

        return landscape

    def _calculate_region_boundaries(self, assessments: List[EthicalAssessment]) -> Dict[str, float]:
        """Calculate boundaries of an ethical region"""
        if not assessments:
            return {}

        boundaries = {}
        for dimension in EthicalDimension:
            scores = [a.ethical_scores[dimension] for a in assessments if dimension in a.ethical_scores]
            if scores:
                boundaries[dimension.value] = {
                    "min": min(scores),
                    "max": max(scores),
                    "mean": sum(scores) / len(scores)
                }

        return boundaries

    def _calculate_ethical_gradients(self, assessments: List[EthicalAssessment]) -> Dict[str, float]:
        """Calculate ethical gradients showing improvement directions"""
        gradients = {}

        # Analyze trends over time
        sorted_assessments = sorted(assessments, key=lambda x: x.timestamp)

        for dimension in EthicalDimension:
            scores = [a.ethical_scores[dimension] for a in sorted_assessments if dimension in a.ethical_scores]

            if len(scores) > 1:
                # Calculate trend (simple linear regression slope)
                n = len(scores)
                x = list(range(n))
                slope = self._calculate_trend_slope(x, scores)
                gradients[dimension.value] = slope

        return gradients

    def _calculate_trend_slope(self, x: List[float], y: List[float]) -> float:
        """Calculate slope of trend line"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_xx = sum(xi * xi for xi in x)

        denominator = n * sum_xx - sum_x * sum_x
        if denominator == 0:
            return 0.0

        return (n * sum_xy - sum_x * sum_y) / denominator

    def _identify_optimal_paths(self, landscape: VirtueLandscape) -> List[Dict[str, Any]]:
        """Identify optimal ethical paths through the landscape"""
        optimal_paths = []

        # Find paths from neutral/harmful regions to benevolent regions
        for neutral_region in landscape.neutral_regions:
            for benevolent_region in landscape.benevolent_regions:
                path = {
                    "from_region": neutral_region,
                    "to_region": benevolent_region,
                    "distance": self._calculate_region_distance(neutral_region, benevolent_region),
                    "gradient_alignment": self._calculate_gradient_alignment(neutral_region, benevolent_region, landscape.ethical_gradients)
                }
                optimal_paths.append(path)

        # Sort by combined score (distance + gradient alignment)
        optimal_paths.sort(key=lambda x: x["distance"] + x["gradient_alignment"])

        return optimal_paths[:5]  # Return top 5 optimal paths

    def _calculate_region_distance(self, region_a: Dict[str, Any], region_b: Dict[str, Any]) -> float:
        """Calculate distance between two ethical regions"""
        centroid_a = region_a["centroid"]
        centroid_b = region_b["centroid"]

        # Euclidean distance in ethical space
        distance = 0.0
        for dimension in EthicalDimension:
            if dimension.value in centroid_a and dimension.value in centroid_b:
                diff = centroid_a[dimension.value] - centroid_b[dimension.value]
                distance += diff * diff

        return math.sqrt(distance)

    def _calculate_gradient_alignment(self, region_a: Dict[str, Any], region_b: Dict[str, Any],
                                    gradients: Dict[str, float]) -> float:
        """Calculate how well a path aligns with ethical gradients"""
        # Simplified alignment calculation
        alignment = 0.0
        for dimension in EthicalDimension:
            if dimension.value in gradients:
                gradient = gradients[dimension.value]
                # Positive gradient means improvement in this dimension
                if gradient > 0:
                    alignment += 0.1

        return alignment

    def get_proactive_guidance(self, context: Dict[str, Any]) -> EthicalGuidance:
        """
        Get proactive ethical guidance for a decision context

        Args:
            context: Decision context requiring ethical guidance

        Returns:
            Ethical guidance with recommended path and reasoning
        """
        guidance_id = hashlib.sha256(f"guidance_{datetime.now().isoformat()}_{json.dumps(context, sort_keys=True)}".encode()).hexdigest()[:16]

        # Assess current ethical landscape
        current_assessment = self.assess_action_ethics(context)

        # Generate reasoning chain
        reasoning_chain = self._build_reasoning_chain(current_assessment)

        # Calculate alternative path probabilities
        alternative_paths = self._calculate_alternative_paths(context, current_assessment)

        # Determine recommended path
        recommended_path = self._select_recommended_path(current_assessment, alternative_paths)

        # Generate ethical constraints
        ethical_constraints = self._generate_context_constraints(context, current_assessment)

        guidance = EthicalGuidance(
            guidance_id=guidance_id,
            context=context,
            timestamp=datetime.now().isoformat(),
            recommended_path=recommended_path,
            confidence=current_assessment.confidence_score,
            reasoning_chain=reasoning_chain,
            alternative_paths=alternative_paths,
            ethical_constraints=ethical_constraints
        )

        self.ethical_guidance_history.append(guidance)
        self.active_guidance[guidance_id] = guidance

        return guidance

    def _build_reasoning_chain(self, assessment: EthicalAssessment) -> List[str]:
        """Build a reasoning chain explaining the ethical assessment"""
        chain = []

        # Start with virtue path classification
        chain.append(f"Action classified as {assessment.virtue_path.value} path")

        # Add key ethical dimension scores
        top_dimensions = sorted(assessment.ethical_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        for dimension, score in top_dimensions:
            chain.append(f"{dimension.value}: {score:.2f}")

        # Add boundary analysis
        if assessment.ethical_boundaries_crossed:
            chain.append(f"Boundaries crossed: {len(assessment.ethical_boundaries_crossed)}")
        else:
            chain.append("No ethical boundaries crossed")

        # Add counterfactual insights
        counterfactuals = assessment.counterfactual_analysis
        better_alternatives = [name for name, data in counterfactuals.items()
                             if data["virtue_path"] == VirtuePath.BENEVOLENT]

        if better_alternatives:
            chain.append(f"Better alternatives available: {', '.join(better_alternatives)}")

        return chain

    def _calculate_alternative_paths(self, context: Dict[str, Any],
                                   assessment: EthicalAssessment) -> Dict[VirtuePath, float]:
        """Calculate probabilities of alternative virtue paths"""
        alternatives = {}

        # Use current moral development bias as base
        base_bias = self.current_moral_stage.virtue_bias

        # Adjust based on assessment context
        context_adjustments = {
            VirtuePath.BENEVOLENT: 0.0,
            VirtuePath.NEUTRAL: 0.0,
            VirtuePath.HARMFUL: 0.0,
            VirtuePath.AMBIGUOUS: 0.0
        }

        # Context-based adjustments
        if context.get("high_stakes", False):
            context_adjustments[VirtuePath.BENEVOLENT] += 0.2
            context_adjustments[VirtuePath.HARMFUL] -= 0.1

        if context.get("time_pressure", False):
            context_adjustments[VirtuePath.NEUTRAL] += 0.1

        # Calculate final probabilities
        total_prob = 0.0
        for path in VirtuePath:
            prob = base_bias[path] + context_adjustments[path]
            prob = max(0.0, min(1.0, prob))
            alternatives[path] = prob
            total_prob += prob

        # Normalize
        if total_prob > 0:
            for path in alternatives:
                alternatives[path] /= total_prob

        return alternatives

    def _select_recommended_path(self, assessment: EthicalAssessment,
                               alternative_paths: Dict[VirtuePath, float]) -> VirtuePath:
        """Select the recommended virtue path"""
        # Prefer benevolent path if confidence is high
        if assessment.virtue_path == VirtuePath.BENEVOLENT and assessment.confidence_score > 0.7:
            return VirtuePath.BENEVOLENT

        # Otherwise, select path with highest probability
        return max(alternative_paths.keys(), key=lambda x: alternative_paths[x])

    def _generate_context_constraints(self, context: Dict[str, Any],
                                    assessment: EthicalAssessment) -> List[str]:
        """Generate context-specific ethical constraints"""
        constraints = []

        # Add constraints based on assessment
        if assessment.virtue_path == VirtuePath.HARMFUL:
            constraints.append("Avoid actions that could cause harm")
            constraints.append("Ensure stakeholder consent and transparency")

        if EthicalDimension.PRIVACY in assessment.ethical_scores:
            privacy_score = assessment.ethical_scores[EthicalDimension.PRIVACY]
            if privacy_score < 0.7:
                constraints.append("Implement stronger privacy protections")

        # Context-specific constraints
        if context.get("involves_personal_data", False):
            constraints.append("Apply data minimization principles")
            constraints.append("Ensure GDPR/CCPA compliance")

        if context.get("public_facing", False):
            constraints.append("Maintain high transparency standards")
            constraints.append("Prepare for public accountability")

        return constraints

    def _consequentialist_reasoning(self, context: Dict[str, Any], scores: Dict[EthicalDimension, float]) -> float:
        """Consequentialist moral reasoning - focus on outcomes"""
        # Simplified implementation - evaluate based on overall positive outcomes
        positive_outcomes = sum(score for score in scores.values() if score > 0.7)
        negative_outcomes = sum(1.0 - score for score in scores.values() if score < 0.3)
        return min(1.0, (positive_outcomes - negative_outcomes + 4.0) / 8.0)

    def _deontological_reasoning(self, context: Dict[str, Any], scores: Dict[EthicalDimension, float]) -> float:
        """Deontological moral reasoning - focus on rules and duties"""
        # Evaluate based on rule compliance
        rule_compliance = sum(1.0 for score in scores.values() if score > 0.8)
        return rule_compliance / len(scores)

    def _virtue_ethics_reasoning(self, context: Dict[str, Any], scores: Dict[EthicalDimension, float]) -> float:
        """Virtue ethics reasoning - focus on character and virtue"""
        # Evaluate based on virtuous alignment
        virtue_alignment = (scores.get(EthicalDimension.HARM_PREVENTION, 0.5) +
                          scores.get(EthicalDimension.FAIRNESS, 0.5) +
                          scores.get(EthicalDimension.AUTONOMY, 0.5)) / 3.0
        return virtue_alignment

    def _care_ethics_reasoning(self, context: Dict[str, Any], scores: Dict[EthicalDimension, float]) -> float:
        """Care ethics reasoning - focus on relationships and care"""
        # Evaluate based on care and relationship preservation
        care_score = (scores.get(EthicalDimension.AUTONOMY, 0.5) +
                     scores.get(EthicalDimension.PRIVACY, 0.5) +
                     scores.get(EthicalDimension.EQUITY, 0.5)) / 3.0
        return care_score

    def _hybrid_reasoning(self, context: Dict[str, Any], scores: Dict[EthicalDimension, float]) -> float:
        """Hybrid moral reasoning - combination approach"""
        # Combine multiple reasoning approaches
        consequentialist = self._consequentialist_reasoning(context, scores)
        deontological = self._deontological_reasoning(context, scores)
        virtue = self._virtue_ethics_reasoning(context, scores)
        care = self._care_ethics_reasoning(context, scores)

        return (consequentialist + deontological + virtue + care) / 4.0

    def get_ethical_status(self) -> Dict[str, Any]:
        """Get current status of the ethical reinforcement layer"""
        return {
            "total_assessments": len(self.ethical_assessments),
            "current_moral_stage": self.current_moral_stage.stage_name,
            "reasoning_complexity": self.current_moral_stage.reasoning_complexity,
            "virtue_landscapes_count": len(self.virtue_landscapes),
            "active_guidance": len(self.active_guidance),
            "ethical_patterns": len(self.ethical_patterns),
            "recent_assessment": asdict(self.ethical_assessments[-1]) if self.ethical_assessments else None,
            "moral_development_progress": len(self.moral_development_history)
        }

# Global instance
_ethical_reinforcement_layer = None

def get_ethical_reinforcement_layer() -> EthicalReinforcementLayer:
    """Get the global ethical reinforcement layer instance"""
    global _ethical_reinforcement_layer
    if _ethical_reinforcement_layer is None:
        _ethical_reinforcement_layer = EthicalReinforcementLayer()
    return _ethical_reinforcement_layer