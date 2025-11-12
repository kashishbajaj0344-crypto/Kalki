# ============================================================
# Kalki Conscious Decision Engine
# ------------------------------------------------------------
# Make decisions informed by:
# - Consciousness state (awareness, emotional resonance)
# - Analytical evaluation (logic, facts, data)
# - Ethical assessment (multi-framework moral reasoning)
# - Weighted combination for human-like decision making
# ============================================================

import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from modules.utils.logging_config import get_logger

logger = get_logger("Kalki.ConsciousDecisionEngine")

@dataclass
class DecisionOption:
    """An option to be evaluated for decision making"""
    id: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DecisionEvaluation:
    """Evaluation results for a decision option"""
    option: DecisionOption
    analytical_score: float  # 0-1
    emotional_score: float  # 0-1
    ethical_score: float  # 0-1
    consciousness_weight: float  # 0-1
    final_score: float  # 0-1
    reasoning: Dict[str, str]  # Explanation for each component
    confidence: float  # 0-1

@dataclass
class DecisionResult:
    """Final decision result"""
    best_option: DecisionOption
    evaluation: DecisionEvaluation
    all_evaluations: List[DecisionEvaluation]
    decision_method: str
    consciousness_level: float
    timestamp: str

class ConsciousDecisionEngine:
    """
    Make decisions informed by consciousness state
    
    Combines:
    - Analytical reasoning (facts, logic, optimization)
    - Emotional intelligence (resonance, intuition, aesthetic)
    - Ethical reasoning (multi-framework moral analysis)
    - Consciousness weighting (awareness-informed decisions)
    """
    
    def __init__(self):
        # Lazy-load components
        self.consciousness = None
        self.ethics_agent = None
        self.meta_core = None
        
        # Decision history
        self.decision_history = []
        
        logger.info("Conscious Decision Engine initialized")
    
    async def _ensure_components_loaded(self):
        """Lazy-load decision components"""
        if self.consciousness is None:
            try:
                from modules.consciousness_engine import ConsciousnessEngine
                self.consciousness = ConsciousnessEngine()
                logger.info("Consciousness Engine loaded")
            except Exception as e:
                logger.warning(f"Consciousness Engine unavailable: {e}")
        
        if self.ethics_agent is None:
            try:
                from modules.agents.safety.ethics import EthicsAgent
                self.ethics_agent = EthicsAgent()
                await self.ethics_agent.initialize()
                logger.info("Ethics Agent loaded")
            except Exception as e:
                logger.warning(f"Ethics Agent unavailable: {e}")
        
        if self.meta_core is None:
            try:
                from modules.meta_core import get_meta_core
                self.meta_core = get_meta_core()
                logger.info("Meta-Core loaded")
            except Exception as e:
                logger.warning(f"Meta-Core unavailable: {e}")
    
    async def make_decision(
        self,
        options: List[DecisionOption],
        context: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None
    ) -> DecisionResult:
        """
        Make a decision using consciousness-informed multi-criteria analysis
        
        Args:
            options: List of options to evaluate
            context: Decision context
            weights: Optional custom weights for analytical, emotional, ethical
                    Default: {"analytical": 0.4, "emotional": 0.2, "ethical": 0.4}
        
        Returns:
            DecisionResult with selected option and full evaluation
        """
        await self._ensure_components_loaded()
        
        if weights is None:
            weights = {
                "analytical": 0.4,
                "emotional": 0.2,
                "ethical": 0.4
            }
        
        logger.info(f"Making conscious decision among {len(options)} options")
        
        # Step 1: Get consciousness state
        consciousness_level = 0.5  # Default
        emotional_base = 0.5
        
        if self.consciousness:
            try:
                consciousness_state = await self.consciousness.achieve_consciousness({
                    "decision_maker": {
                        "options": [opt.description for opt in options],
                        "context": context
                    }
                })
                consciousness_level = consciousness_state.awareness_level
                emotional_base = consciousness_state.emotional_resonance
                logger.info(f"🧠 Consciousness level: {consciousness_level:.3f}, "
                          f"Emotional resonance: {emotional_base:.3f}")
            except Exception as e:
                logger.warning(f"Consciousness assessment failed: {e}")
        
        # Step 2: Evaluate each option
        evaluations = []
        
        for option in options:
            logger.info(f"Evaluating option: {option.description}")
            
            # Analytical evaluation
            analytical_score = await self._analytical_evaluation(option, context)
            
            # Emotional evaluation (consciousness-based)
            emotional_score = await self._emotional_evaluation(
                option, context, emotional_base
            )
            
            # Ethical evaluation
            ethical_score = await self._ethical_evaluation(option, context)
            
            # Consciousness-weighted combination
            final_score = (
                weights["analytical"] * analytical_score +
                weights["emotional"] * emotional_score +
                weights["ethical"] * ethical_score
            ) * consciousness_level
            
            # Confidence based on score consistency
            score_variance = self._calculate_variance([
                analytical_score, emotional_score, ethical_score
            ])
            confidence = 1.0 - min(score_variance, 1.0)
            
            evaluation = DecisionEvaluation(
                option=option,
                analytical_score=analytical_score,
                emotional_score=emotional_score,
                ethical_score=ethical_score,
                consciousness_weight=consciousness_level,
                final_score=final_score,
                reasoning={
                    "analytical": self._explain_analytical(analytical_score),
                    "emotional": self._explain_emotional(emotional_score),
                    "ethical": self._explain_ethical(ethical_score),
                    "consciousness": f"Awareness level {consciousness_level:.2f} amplifies decision quality"
                },
                confidence=confidence
            )
            
            evaluations.append(evaluation)
            logger.info(f"  Final score: {final_score:.3f} (confidence: {confidence:.2f})")
        
        # Step 3: Select best option
        best_evaluation = max(evaluations, key=lambda e: e.final_score)
        
        result = DecisionResult(
            best_option=best_evaluation.option,
            evaluation=best_evaluation,
            all_evaluations=evaluations,
            decision_method="consciousness_weighted_multi_criteria",
            consciousness_level=consciousness_level,
            timestamp=datetime.now().isoformat()
        )
        
        # Track decision history
        self.decision_history.append(result)
        if len(self.decision_history) > 100:
            self.decision_history = self.decision_history[-50:]
        
        logger.info(f"✅ Decision made: {best_evaluation.option.description} "
                   f"(score: {best_evaluation.final_score:.3f})")
        
        return result
    
    async def _analytical_evaluation(
        self, 
        option: DecisionOption, 
        context: Dict[str, Any]
    ) -> float:
        """Evaluate option using analytical reasoning"""
        
        # Heuristic analytical scoring based on parameters
        score = 0.5  # Default
        
        # Check for quantitative metrics
        if "cost" in option.parameters:
            # Lower cost is better (assuming budget constraint)
            budget = context.get("budget", float('inf'))
            cost_ratio = option.parameters["cost"] / budget
            cost_score = max(0.0, 1.0 - cost_ratio)
            score += 0.2 * cost_score
        
        if "performance" in option.parameters:
            # Higher performance is better
            perf = option.parameters["performance"]
            score += 0.3 * min(perf, 1.0)
        
        if "complexity" in option.parameters:
            # Lower complexity preferred
            complexity = option.parameters["complexity"]
            complexity_score = 1.0 - min(complexity, 1.0)
            score += 0.1 * complexity_score
        
        # Normalize to 0-1
        return min(max(score, 0.0), 1.0)
    
    async def _emotional_evaluation(
        self,
        option: DecisionOption,
        context: Dict[str, Any],
        emotional_base: float
    ) -> float:
        """Evaluate option using emotional resonance"""
        
        # Emotional scoring based on intuitive factors
        score = emotional_base  # Start with consciousness emotional state
        
        # Aesthetic appeal
        if "aesthetic_score" in option.parameters:
            score = (score + option.parameters["aesthetic_score"]) / 2
        
        # Novelty/creativity bonus
        if "novelty" in option.parameters:
            novelty_bonus = 0.2 * option.parameters["novelty"]
            score = min(score + novelty_bonus, 1.0)
        
        # User preference alignment
        if "user_preference" in context:
            if context["user_preference"] in option.description.lower():
                score = min(score + 0.1, 1.0)
        
        return min(max(score, 0.0), 1.0)
    
    async def _ethical_evaluation(
        self,
        option: DecisionOption,
        context: Dict[str, Any]
    ) -> float:
        """Evaluate option using ethical reasoning"""
        
        if self.ethics_agent:
            try:
                # Use ethics agent for full evaluation
                ethics_result = await self.ethics_agent.execute({
                    "action": "assess_ethics",
                    "decision": option.description,
                    "context": context
                })
                return ethics_result.get("ethical_score", 0.8)
            except Exception as e:
                logger.warning(f"Ethics agent evaluation failed: {e}")
        
        # Fallback: heuristic ethical scoring
        score = 0.8  # Default: most options are ethically acceptable
        
        # Check for harmful keywords
        harmful_keywords = ["harm", "danger", "illegal", "unethical", "unsafe"]
        if any(keyword in option.description.lower() for keyword in harmful_keywords):
            score = 0.3
        
        # Bonus for beneficial keywords
        beneficial_keywords = ["help", "improve", "benefit", "safe", "sustainable"]
        if any(keyword in option.description.lower() for keyword in beneficial_keywords):
            score = min(score + 0.1, 1.0)
        
        return score
    
    def _calculate_variance(self, scores: List[float]) -> float:
        """Calculate variance of scores"""
        if not scores:
            return 0.0
        
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        return variance ** 0.5  # Return standard deviation
    
    def _explain_analytical(self, score: float) -> str:
        """Explain analytical score"""
        if score > 0.8:
            return "Strong analytical support: high performance, low cost, optimal parameters"
        elif score > 0.6:
            return "Good analytical support: reasonable trade-offs"
        elif score > 0.4:
            return "Moderate analytical support: some concerns about metrics"
        else:
            return "Weak analytical support: suboptimal parameters or high cost"
    
    def _explain_emotional(self, score: float) -> str:
        """Explain emotional score"""
        if score > 0.8:
            return "Strong emotional resonance: aesthetically appealing, intuitive, novel"
        elif score > 0.6:
            return "Positive emotional response: generally appealing"
        elif score > 0.4:
            return "Neutral emotional response: functional but uninspiring"
        else:
            return "Weak emotional response: lacks appeal or novelty"
    
    def _explain_ethical(self, score: float) -> str:
        """Explain ethical score"""
        if score > 0.9:
            return "Excellent ethical alignment: promotes well-being, sustainable, safe"
        elif score > 0.7:
            return "Good ethical alignment: generally positive impact"
        elif score > 0.5:
            return "Acceptable ethical alignment: neutral impact"
        else:
            return "Ethical concerns: potential harm or negative consequences"
    
    def get_decision_statistics(self) -> Dict[str, Any]:
        """Get decision-making statistics"""
        if not self.decision_history:
            return {
                "total_decisions": 0,
                "average_consciousness_level": 0.0,
                "average_confidence": 0.0
            }
        
        return {
            "total_decisions": len(self.decision_history),
            "average_consciousness_level": sum(d.consciousness_level for d in self.decision_history) / len(self.decision_history),
            "average_confidence": sum(d.evaluation.confidence for d in self.decision_history) / len(self.decision_history),
            "average_final_score": sum(d.evaluation.final_score for d in self.decision_history) / len(self.decision_history)
        }


# Global singleton instance
_conscious_decision_engine = None

def get_conscious_decision_engine() -> ConsciousDecisionEngine:
    """Get or create the global Conscious Decision Engine instance"""
    global _conscious_decision_engine
    if _conscious_decision_engine is None:
        _conscious_decision_engine = ConsciousDecisionEngine()
    return _conscious_decision_engine
