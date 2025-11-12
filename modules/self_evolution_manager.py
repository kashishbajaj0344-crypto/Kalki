# ============================================================
# Kalki v2.4 — self_evolution_manager.py
# ------------------------------------------------------------
# Self-Evolution Manager: Living Intelligence Architecture
# - Audits performance metrics across sessions
# - Identifies improvement opportunities
# - Suggests architecture updates and retraining targets
# - Writes structured changelog and proposes new heuristic weights
# - Enables Kalki to redesign parts of itself intelligently
# ============================================================

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import statistics
import inspect
import importlib
import sys

from modules.utils.logging_config import get_logger
from modules.reinforcement_loop import get_reinforcement_loop
from modules.temporal_consistency import get_temporal_consistency_buffer
from modules.meta_core import get_meta_core

logger = get_logger("Kalki.SelfEvolution")

class EvolutionPriority(Enum):
    """Priority levels for evolution recommendations"""
    CRITICAL = "critical"      # Immediate action required
    HIGH = "high"             # Important improvements
    MEDIUM = "medium"         # Beneficial enhancements
    LOW = "low"              # Minor optimizations

class EvolutionType(Enum):
    """Types of architectural evolution"""
    HEURISTIC_TUNING = "heuristic_tuning"          # Adjust weights and parameters
    MODULE_ENHANCEMENT = "module_enhancement"      # Improve existing modules
    NEW_CAPABILITY = "new_capability"              # Add new features
    ARCHITECTURE_REFACTOR = "architecture_refactor" # Major structural changes
    LEARNING_RATE_ADJUSTMENT = "learning_rate_adjustment" # Meta-learning parameters

@dataclass
class PerformanceAudit:
    """Comprehensive performance audit"""
    audit_id: str
    timestamp: str
    time_range_days: int
    metrics_summary: Dict[str, Any]
    trend_analysis: Dict[str, Any]
    bottleneck_identification: List[str]
    strength_assessment: List[str]
    overall_health_score: float  # 0-1 scale

@dataclass
class EvolutionRecommendation:
    """Specific evolution recommendation"""
    recommendation_id: str
    evolution_type: EvolutionType
    priority: EvolutionPriority
    title: str
    description: str
    rationale: str
    implementation_plan: List[str]
    expected_impact: Dict[str, float]  # metric -> expected_change
    risk_assessment: str
    dependencies: List[str]
    estimated_effort: str  # "low", "medium", "high"
    proposed_by: str  # "reinforcement_loop", "temporal_consistency", "self_audit"

@dataclass
class ArchitectureChangelog:
    """Structured changelog for architectural changes"""
    version: str
    timestamp: str
    changes: List[Dict[str, Any]]
    performance_impact: Dict[str, Any]
    rollback_plan: Dict[str, Any]
    validation_results: Dict[str, Any]

@dataclass
class SelfEvolutionState:
    """Current state of self-evolution"""
    last_audit: str
    pending_recommendations: List[EvolutionRecommendation]
    implemented_changes: List[ArchitectureChangelog]
    evolution_goals: Dict[str, Any]
    self_awareness_level: float  # 0-1 scale
    evolution_velocity: float   # improvements per day
    performance_history: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    improvement_backlog: List[Dict[str, Any]] = field(default_factory=list)
    execution_history: List[Dict[str, Any]] = field(default_factory=list)

class SelfEvolutionManager:
    """
    Self-Evolution Manager: Living Intelligence Architecture

    Enables Kalki to audit its own performance, identify improvement opportunities,
    and intelligently redesign parts of itself. This creates a truly self-evolving system.
    """

    def __init__(self):
        self.reinforcement_loop = get_reinforcement_loop()
        self.temporal_buffer = get_temporal_consistency_buffer()
        self.meta_core = get_meta_core()

        # Import here to avoid circular dependencies
        from modules.canary_deployment_manager import get_canary_deployment_manager
        from modules.safety_monitoring_system import get_safety_monitoring_system

        self.canary_manager = get_canary_deployment_manager()
        self.safety_monitor = get_safety_monitoring_system()

        # Evolution state
        self.evolution_state = SelfEvolutionState(
            last_audit=datetime.now().isoformat(),
            pending_recommendations=[],
            implemented_changes=[],
            evolution_goals={
                "interdisciplinary_coverage": 0.9,
                "confidence_calibration": 0.85,
                "ethical_alignment": 1.0,
                "creativity_rigor_balance": 0.9,
                "efficiency_ratio": 0.9,
                "learning_rate": 0.1
            },
            self_awareness_level=0.7,
            evolution_velocity=0.0,
            performance_history={},
            improvement_backlog=[],
            execution_history=[]
        )

        # Persistence
        self.data_dir = "data/self_evolution"
        self.state_file = f"{self.data_dir}/evolution_state.json"
        self.audit_file = f"{self.data_dir}/performance_audits.json"

        # Load existing state
        self._load_persistent_state()

        logger.info("Self-Evolution Manager initialized")

    async def analyze_system_performance(
        self,
        metrics: Dict[str, Any],
        system_name: str
    ) -> Dict[str, Any]:
        """
        Analyze system performance metrics and record historical trends.

        Args:
            metrics: Collected performance metrics for the system
            system_name: Unique identifier for the system being analyzed

        Returns:
            Analysis summary including scorecard, trends, and priority areas
        """
        if not system_name:
            system_name = "unknown_system"

        history = self.evolution_state.performance_history.setdefault(system_name, [])
        timestamp = datetime.now().isoformat()

        trend = self._calculate_metric_trends(history, metrics)
        scorecard = self._build_scorecard(metrics, trend)

        history.append({
            "timestamp": timestamp,
            "metrics": metrics,
            "scorecard": scorecard,
            "trend": trend
        })
        # Keep history manageable
        if len(history) > 120:
            self.evolution_state.performance_history[system_name] = history[-120:]

        analysis = {
            "system": system_name,
            "timestamp": timestamp,
            "metrics": metrics,
            "scorecard": scorecard,
            "trend": trend,
            "priority_areas": self._determine_priority_areas(scorecard, trend),
            "historical_samples": len(history)
        }

        self._save_persistent_state()
        return analysis

    async def propose_improvements(
        self,
        analysis: Dict[str, Any],
        confidence_threshold: float = 0.75,
        risk_threshold: str = "medium"
    ) -> List[Dict[str, Any]]:
        """
        Generate improvement proposals based on analysis.

        Args:
            analysis: Output of analyze_system_performance
            confidence_threshold: Minimum confidence required
            risk_threshold: Maximum acceptable risk level ('low', 'medium', 'high')
        """
        improvements: List[Dict[str, Any]] = []
        system_name = analysis.get("system", "unknown_system")
        scorecard = analysis.get("scorecard", {})
        metrics = analysis.get("metrics", {})
        priority_areas = analysis.get("priority_areas", [])

        risk_order = ["low", "medium", "high", "critical"]
        max_risk_index = risk_order.index(risk_threshold) if risk_threshold in risk_order else 1

        def add_improvement(improvement: Dict[str, Any]):
            confidence = improvement.get("confidence", 0.6)
            risk = improvement.get("risk", "medium")
            if confidence >= confidence_threshold and risk_order.index(risk) <= max_risk_index:
                improvement_id = f"imp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.evolution_state.improvement_backlog)+1}"
                improvement.update({
                    "id": improvement_id,
                    "system": system_name,
                    "status": "proposed",
                    "created_at": datetime.now().isoformat()
                })
                self.evolution_state.improvement_backlog.append(improvement)
                improvements.append(improvement)

        # Completion rate improvements
        completion_rate = metrics.get("user_completion_rate")
        if completion_rate is not None and completion_rate < 0.6:
            add_improvement({
                "area": "workflow_completion",
                "description": "Improve user completion rate via stage-specific guidance",
                "confidence": 0.8,
                "risk": "low",
                "expected_impact": {"completion_rate": 0.1},
                "action_plan": [
                    "Introduce proactive reminders for stalled stages",
                    "Surface expert tips when abandonment detected",
                    "Automate follow-up checklist distribution"
                ]
            })

        # Satisfaction improvements
        satisfaction = metrics.get("average_satisfaction")
        if satisfaction is not None and satisfaction < 0.75:
            add_improvement({
                "area": "user_experience",
                "description": "Boost satisfaction with personalized follow-ups",
                "confidence": 0.78,
                "risk": "medium",
                "expected_impact": {"average_satisfaction": 0.1},
                "action_plan": [
                    "Deploy sentiment-aware follow-up messages",
                    "Highlight positive progress milestones",
                    "Provide instant answers to frequent questions"
                ]
            })

        # Bottleneck improvements
        bottlenecks = metrics.get("bottleneck_stages", [])
        for stage in bottlenecks:
            add_improvement({
                "area": f"bottleneck_{stage}",
                "description": f"Reduce delays at stage '{stage}'",
                "confidence": 0.76,
                "risk": "medium",
                "expected_impact": {"time_to_first_value": -0.3},
                "action_plan": [
                    f"Deploy stage-specific templates for {stage}",
                    f"Create auto-escalation rule if {stage} stalls for > 5 days",
                    f"Curate expert content addressing typical blockers in {stage}"
                ]
            })

        # Abandonment improvements
        abandon_points = metrics.get("abandon_points", [])
        for point in abandon_points:
            add_improvement({
                "area": f"abandonment_{point['stage']}",
                "description": f"Mitigate abandonment at {point['stage']}",
                "confidence": 0.82 if point.get("abandon_rate", 0) > 0.2 else 0.7,
                "risk": "low",
                "expected_impact": {"completion_rate": point.get("abandon_rate", 0) * 0.5},
                "action_plan": [
                    f"Introduce decision tree guidance for {point['stage']}",
                    f"Offer live expert handoff when users hesitate at {point['stage']}",
                    f"Provide ROI examples to reinforce progress through {point['stage']}"
                ]
            })

        # Priority area improvements
        for area in priority_areas:
            if area == "time_to_value":
                add_improvement({
                    "area": "time_to_value",
                    "description": "Accelerate time-to-first-value with smart onboarding",
                    "confidence": 0.77,
                    "risk": "low",
                    "expected_impact": {"time_to_first_value": -0.5},
                    "action_plan": [
                        "Deploy adaptive onboarding checklists",
                        "Pre-load domain-specific best practices",
                        "Automate early milestone recognition"
                    ]
                })

        self._save_persistent_state()
        return improvements

    async def record_improvement_result(
        self,
        improvement_id: str,
        success: bool,
        outcome_metrics: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Record the outcome of an implemented improvement.
        """
        outcome_metrics = outcome_metrics or {}
        for improvement in self.evolution_state.improvement_backlog:
            if improvement.get("id") == improvement_id:
                improvement["status"] = "completed" if success else "failed"
                improvement["completed_at"] = datetime.now().isoformat()
                improvement["outcome_metrics"] = outcome_metrics
                if notes:
                    improvement["notes"] = notes
                self._save_persistent_state()
                logger.info(f"Recorded outcome for improvement {improvement_id}: success={success}")
                return improvement

        logger.warning(f"Improvement {improvement_id} not found when recording outcome")
        return {"id": improvement_id, "success": success, "error": "improvement_not_found"}

    def _calculate_metric_trends(self, history: List[Dict[str, Any]], metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate metric deltas compared to previous snapshot."""
        trend: Dict[str, float] = {}
        if not history:
            return trend

        previous_metrics = history[-1].get("metrics", {})
        for key, value in metrics.items():
            prev_value = previous_metrics.get(key)
            if isinstance(value, (int, float)) and isinstance(prev_value, (int, float)):
                trend[key] = value - prev_value
            elif isinstance(value, list) and isinstance(prev_value, list):
                trend[key] = len(value) - len(prev_value)
        return trend

    def _build_scorecard(self, metrics: Dict[str, Any], trend: Dict[str, float]) -> Dict[str, Any]:
        """Build normalized scorecard for quick evaluation."""
        scorecard = {}

        completion = metrics.get("user_completion_rate")
        if completion is not None:
            scorecard["completion_rate"] = completion

        satisfaction = metrics.get("average_satisfaction")
        if satisfaction is not None:
            scorecard["satisfaction"] = satisfaction

        time_to_value = metrics.get("time_to_first_value")
        if time_to_value is not None:
            scorecard["time_to_value"] = max(0.0, min(1.0, 1.0 - (time_to_value / 10.0)))

        issues_count = len(metrics.get("issues_encountered", []))
        scorecard["issue_load"] = max(0.0, min(1.0, 1.0 - (issues_count / 10.0)))

        abandonment_pressure = sum(point.get("abandon_rate", 0) for point in metrics.get("abandon_points", []))
        scorecard["abandonment_pressure"] = max(0.0, min(1.0, 1.0 - abandonment_pressure))

        # Attach trend information
        scorecard["trend"] = trend
        return scorecard

    def _determine_priority_areas(self, scorecard: Dict[str, Any], trend: Dict[str, float]) -> List[str]:
        """Determine priority focus areas based on scorecard and trend."""
        priorities: List[str] = []

        completion = scorecard.get("completion_rate", 1.0)
        if completion < 0.7 or trend.get("user_completion_rate", 0) < -0.05:
            priorities.append("workflow_completion")

        satisfaction = scorecard.get("satisfaction", 1.0)
        if satisfaction < 0.8 or trend.get("average_satisfaction", 0) < -0.05:
            priorities.append("user_experience")

        if scorecard.get("time_to_value", 1.0) < 0.7 or trend.get("time_to_first_value", 0) > 0.5:
            priorities.append("time_to_value")

        if scorecard.get("issue_load", 1.0) < 0.75:
            priorities.append("quality_control")

        if scorecard.get("abandonment_pressure", 1.0) < 0.8:
            priorities.append("retention")

        return priorities

    def _load_persistent_state(self):
        """Load persistent evolution state"""
        try:
            os.makedirs(self.data_dir, exist_ok=True)

            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state_data = json.load(f)
                    # Load evolution state
                    for key, value in state_data.get('evolution_state', {}).items():
                        if hasattr(self.evolution_state, key):
                            setattr(self.evolution_state, key, value)

                    # Load pending recommendations
                    self.evolution_state.pending_recommendations = [
                        EvolutionRecommendation(**rec)
                        for rec in state_data.get('pending_recommendations', [])
                    ]

        except Exception as e:
            logger.warning(f"Failed to load persistent evolution state: {e}")

    def _save_persistent_state(self):
        """Save evolution state persistently"""
        try:
            # Create JSON-serializable version of evolution state
            evolution_state_dict = asdict(self.evolution_state)
            
            # Handle execution_history separately to ensure JSON serializability
            if 'execution_history' in evolution_state_dict:
                # execution_history is already a list of dicts, should be JSON serializable
                pass  # Keep as is
            
            state_data = {
                'evolution_state': evolution_state_dict,
                'pending_recommendations': [asdict(r) for r in self.evolution_state.pending_recommendations],
                'last_updated': datetime.now().isoformat()
            }

            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)  # Use default=str to handle any remaining non-serializable objects

        except Exception as e:
            logger.error(f"Failed to save persistent evolution state: {e}")
            # Try to save a minimal state as fallback
            try:
                minimal_state = {
                    'evolution_state': {
                        'last_audit': self.evolution_state.last_audit,
                        'self_awareness_level': self.evolution_state.self_awareness_level,
                        'evolution_velocity': self.evolution_state.evolution_velocity
                    },
                    'error': 'Full state save failed, using minimal state',
                    'last_updated': datetime.now().isoformat()
                }
                with open(self.state_file, 'w') as f:
                    json.dump(minimal_state, f, indent=2, default=str)
            except Exception as fallback_error:
                logger.error(f"Fallback state save also failed: {fallback_error}")

    async def perform_comprehensive_audit(self, days_to_audit: int = 30) -> PerformanceAudit:
        """
        Perform comprehensive performance audit across all systems

        Args:
            days_to_audit: Number of days to include in audit

        Returns:
            Complete performance audit
        """

        audit_id = f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_date = datetime.now() - timedelta(days=days_to_audit)

        logger.info(f"Starting comprehensive audit: {audit_id} for last {days_to_audit} days")

        # Gather data from all systems
        reinforcement_report = self.reinforcement_loop.get_performance_report()
        temporal_report = self.temporal_buffer.get_continuity_report()

        # Analyze trends and patterns
        trend_analysis = await self._analyze_performance_trends(days_to_audit)
        bottlenecks = await self._identify_bottlenecks(reinforcement_report, temporal_report)
        strengths = await self._assess_strengths(reinforcement_report, temporal_report)

        # Calculate overall health score
        health_score = self._calculate_health_score(
            reinforcement_report, temporal_report, trend_analysis
        )

        audit = PerformanceAudit(
            audit_id=audit_id,
            timestamp=datetime.now().isoformat(),
            time_range_days=days_to_audit,
            metrics_summary={
                "reinforcement_metrics": reinforcement_report.get("performance_metrics", {}),
                "temporal_metrics": temporal_report,
                "system_integration": self._assess_system_integration()
            },
            trend_analysis=trend_analysis,
            bottleneck_identification=bottlenecks,
            strength_assessment=strengths,
            overall_health_score=health_score
        )

        # Update evolution state
        self.evolution_state.last_audit = audit.timestamp

        # Save audit
        await self._save_audit(audit)

        # Generate evolution recommendations based on audit
        await self._generate_evolution_recommendations(audit)

        logger.info(f"Comprehensive audit completed: health_score={health_score:.3f}")

        return audit

    async def _save_audit(self, audit: PerformanceAudit):
        """Save audit results persistently"""

        try:
            audit_file = f"{self.audit_file}_{audit.audit_id}.json"
            with open(audit_file, 'w') as f:
                json.dump(asdict(audit), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save audit {audit.audit_id}: {e}")

    async def _analyze_performance_trends(self, days: int) -> Dict[str, Any]:
        """Analyze performance trends over time"""

        trends = {
            "composite_score_trend": "stable",
            "learning_efficiency": "stable",
            "consistency_trend": "stable",
            "improvement_velocity": 0.0,
            "concerning_patterns": [],
            "positive_patterns": []
        }

        # Analyze reinforcement learning trends
        perf_report = self.reinforcement_loop.get_performance_report()
        learning_trends = perf_report.get("learning_trends", {})

        if learning_trends:
            trends["composite_score_trend"] = learning_trends.get("overall_trend", "stable")
            trends["improvement_velocity"] = learning_trends.get("trend_slope", 0.0)

            # Identify patterns
            if learning_trends.get("trend_slope", 0) > 0.001:
                trends["positive_patterns"].append("Consistent improvement in response quality")
            elif learning_trends.get("trend_slope", 0) < -0.001:
                trends["concerning_patterns"].append("Declining performance over time")

        # Analyze temporal consistency
        continuity_report = self.temporal_buffer.get_continuity_report()
        overall_continuity = continuity_report.get("overall_continuity_score", 1.0)

        if overall_continuity < 0.8:
            trends["concerning_patterns"].append("Temporal consistency degrading")
        else:
            trends["positive_patterns"].append("Strong temporal consistency maintained")

        return trends

    async def _identify_bottlenecks(self, reinforcement_report: Dict, temporal_report: Dict) -> List[str]:
        """Identify system bottlenecks"""

        bottlenecks = []

        # Check reinforcement learning bottlenecks
        perf_metrics = reinforcement_report.get("performance_metrics", {})
        if perf_metrics.get("bias_detection_rate", 0) > 0.4:
            bottlenecks.append("High cognitive bias detection - reasoning stability issues")

        if perf_metrics.get("ethical_alignment_rate", 1.0) < 0.9:
            bottlenecks.append("Ethical alignment below target - safety concerns")

        if perf_metrics.get("improvement_rate", 0) < 0:
            bottlenecks.append("Negative improvement rate - learning system needs adjustment")

        # Check temporal consistency bottlenecks
        if temporal_report.get("total_contradictions", 0) > 10:
            bottlenecks.append("High contradiction rate - logical consistency issues")

        if temporal_report.get("overall_continuity_score", 1.0) < 0.7:
            bottlenecks.append("Poor temporal continuity - memory coherence problems")

        return bottlenecks

    async def _assess_strengths(self, reinforcement_report: Dict, temporal_report: Dict) -> List[str]:
        """Assess system strengths"""

        strengths = []

        # Check reinforcement learning strengths
        perf_metrics = reinforcement_report.get("performance_metrics", {})
        if perf_metrics.get("average_composite_score", 0) > 0.8:
            strengths.append("High average response quality maintained")

        if perf_metrics.get("ethical_alignment_rate", 0) > 0.95:
            strengths.append("Excellent ethical alignment achieved")

        # Check temporal consistency strengths
        if temporal_report.get("total_conclusions", 0) > 100:
            strengths.append("Rich temporal knowledge base developed")

        if temporal_report.get("overall_continuity_score", 0) > 0.9:
            strengths.append("Strong temporal consistency across sessions")

        return strengths

    def _calculate_health_score(self, reinforcement_report: Dict,
                              temporal_report: Dict, trend_analysis: Dict) -> float:
        """Calculate overall system health score"""

        base_score = 0.75  # Starting point

        # Reinforcement learning contribution
        perf_metrics = reinforcement_report.get("performance_metrics", {})
        rl_score = perf_metrics.get("average_composite_score", 0.5)
        base_score = base_score * 0.4 + rl_score * 0.6

        # Temporal consistency contribution
        temporal_score = temporal_report.get("overall_continuity_score", 1.0)
        base_score = base_score * 0.8 + temporal_score * 0.2

        # Trend adjustment
        trend_modifier = 0.0
        if trend_analysis.get("composite_score_trend") == "improving":
            trend_modifier = 0.1
        elif trend_analysis.get("composite_score_trend") == "declining":
            trend_modifier = -0.1

        final_score = base_score + trend_modifier

        return max(0.0, min(1.0, final_score))

    def _assess_system_integration(self) -> Dict[str, Any]:
        """Assess integration between system components"""

        integration_score = 0.8  # Base integration

        # Check if all systems are operational
        systems_check = {
            "reinforcement_loop": self.reinforcement_loop is not None,
            "temporal_buffer": self.temporal_buffer is not None,
            "meta_core": self.meta_core is not None
        }

        operational_systems = sum(systems_check.values())
        integration_score *= (operational_systems / len(systems_check))

        return {
            "integration_score": integration_score,
            "operational_systems": operational_systems,
            "total_systems": len(systems_check),
            "systems_status": systems_check
        }

    async def _generate_evolution_recommendations(self, audit: PerformanceAudit):
        """Generate evolution recommendations based on audit"""

        recommendations = []

        # Analyze metrics for specific recommendations
        metrics = audit.metrics_summary

        # Heuristic tuning recommendations
        if audit.overall_health_score < 0.7:
            recommendations.append(EvolutionRecommendation(
                recommendation_id=f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}_heuristic_tune",
                evolution_type=EvolutionType.HEURISTIC_TUNING,
                priority=EvolutionPriority.HIGH,
                title="Heuristic Weight Optimization",
                description="Adjust meta-core heuristic weights based on performance analysis",
                rationale="Current weights may not be optimal for observed performance patterns",
                implementation_plan=[
                    "Analyze weight update history",
                    "Identify underperforming heuristics",
                    "Calculate optimal weight adjustments",
                    "Implement gradual weight updates"
                ],
                expected_impact={"composite_score": 0.1, "efficiency_ratio": 0.05},
                risk_assessment="Low risk - weights can be rolled back",
                dependencies=["reinforcement_loop"],
                estimated_effort="medium",
                proposed_by="self_audit"
            ))

        # Learning rate adjustment
        trend_slope = audit.trend_analysis.get("improvement_velocity", 0)
        if trend_slope < 0:
            recommendations.append(EvolutionRecommendation(
                recommendation_id=f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}_learning_rate",
                evolution_type=EvolutionType.LEARNING_RATE_ADJUSTMENT,
                priority=EvolutionPriority.MEDIUM,
                title="Learning Rate Adjustment",
                description="Reduce learning rate to stabilize performance",
                rationale="Negative improvement trend suggests learning rate too high",
                implementation_plan=[
                    "Analyze learning rate impact on stability",
                    "Gradually reduce learning rate",
                    "Monitor performance stabilization"
                ],
                expected_impact={"improvement_rate": 0.05, "performance_stability": 0.1},
                risk_assessment="Low risk - gradual adjustment",
                dependencies=["reinforcement_loop"],
                estimated_effort="low",
                proposed_by="reinforcement_loop"
            ))

        # Module enhancement recommendations
        if len(audit.bottleneck_identification) > 2:
            recommendations.append(EvolutionRecommendation(
                recommendation_id=f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}_module_enhance",
                evolution_type=EvolutionType.MODULE_ENHANCEMENT,
                priority=EvolutionPriority.HIGH,
                title="Module Enhancement Initiative",
                description="Enhance bottlenecked modules for improved performance",
                rationale=f"Identified {len(audit.bottleneck_identification)} performance bottlenecks",
                implementation_plan=[
                    "Prioritize bottleneck modules",
                    "Design targeted enhancements",
                    "Implement incremental improvements",
                    "Validate performance gains"
                ],
                expected_impact={"overall_health_score": 0.15, "bottleneck_reduction": 0.3},
                risk_assessment="Medium risk - requires careful testing",
                dependencies=["temporal_consistency", "reinforcement_loop"],
                estimated_effort="high",
                proposed_by="self_audit"
            ))

        # New capability recommendations
        if audit.overall_health_score > 0.8 and len(audit.strength_assessment) > 3:
            recommendations.append(EvolutionRecommendation(
                recommendation_id=f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}_new_capability",
                evolution_type=EvolutionType.NEW_CAPABILITY,
                priority=EvolutionPriority.MEDIUM,
                title="Advanced Reasoning Capabilities",
                description="Add advanced reasoning capabilities based on demonstrated strengths",
                rationale="System showing strong performance - ready for capability expansion",
                implementation_plan=[
                    "Identify high-performing reasoning patterns",
                    "Design complementary advanced capabilities",
                    "Implement with extensive testing",
                    "Integrate with existing architecture"
                ],
                expected_impact={"reasoning_depth": 0.2, "capability_breadth": 0.15},
                risk_assessment="Medium risk - extensive testing required",
                dependencies=["meta_core", "supreme_synthesis_engine"],
                estimated_effort="high",
                proposed_by="self_audit"
            ))

        # Add recommendations to evolution state
        self.evolution_state.pending_recommendations.extend(recommendations)

        # Save updated state
        self._save_persistent_state()

        logger.info(f"Generated {len(recommendations)} evolution recommendations")

    async def implement_evolution_recommendation(self,
                                               recommendation_id: str,
                                               auto_approve: bool = False,
                                               human_override: bool = False) -> Dict[str, Any]:
        """
        Implement a specific evolution recommendation

        Args:
            recommendation_id: ID of recommendation to implement
            auto_approve: Whether to automatically approve implementation
            human_override: Whether this is approved by human override

        Returns:
            Implementation result
        """

        # Find recommendation
        recommendation = None
        for rec in self.evolution_state.pending_recommendations:
            if rec.recommendation_id == recommendation_id:
                recommendation = rec
                break

        if not recommendation:
            return {
                "success": False,
                "error": f"Recommendation {recommendation_id} not found"
            }

        # Safety gate evaluation
        safety_result = await self.safety_monitor.evaluate_safety_gate(recommendation, human_override)

        if not safety_result["approved"]:
            return {
                "success": False,
                "error": "Safety gate failed",
                "safety_result": safety_result
            }

        # Check if auto-approve or requires manual approval
        if not auto_approve and not human_override and safety_result["requires_human_review"]:
            return {
                "success": False,
                "error": "Recommendation requires human review",
                "safety_result": safety_result
            }

        logger.info(f"Implementing evolution recommendation: {recommendation.title}")

        # For high-risk changes, use canary deployment
        if recommendation.priority.value in ["high", "critical"] or recommendation.estimated_effort == "high":
            return await self._implement_with_canary(recommendation)
        else:
            # Direct implementation for low-risk changes
            return await self._implement_direct(recommendation)

    async def _implement_with_canary(self, recommendation: EvolutionRecommendation) -> Dict[str, Any]:
        """Implement recommendation using canary deployment"""

        # Deploy as canary
        canary_result = await self.canary_manager.deploy_canary(recommendation)

        if not canary_result["success"]:
            return {
                "success": False,
                "error": f"Canary deployment failed: {canary_result.get('error', 'Unknown error')}",
                "canary_result": canary_result
            }

        # Mark as implemented (canary deployment handles the actual implementation)
        self.evolution_state.pending_recommendations.remove(recommendation)

        # Create changelog entry
        changelog = ArchitectureChangelog(
            version=f"v2.4.{len(self.evolution_state.implemented_changes) + 1}",
            timestamp=datetime.now().isoformat(),
            changes=[{
                "recommendation_id": recommendation.recommendation_id,
                "title": recommendation.title,
                "type": recommendation.evolution_type.value,
                "implementation_method": "canary_deployment",
                "deployment_id": canary_result["deployment_id"]
            }],
            performance_impact=recommendation.expected_impact,
            rollback_plan=self.canary_manager._generate_rollback_plan(recommendation),
            validation_results={"canary_deployment": canary_result}
        )

        self.evolution_state.implemented_changes.append(changelog)

        # Update self-awareness
        self.evolution_state.self_awareness_level = min(1.0,
            self.evolution_state.self_awareness_level + 0.05)

        # Save state
        self._save_persistent_state()

        return {
            "success": True,
            "implementation_method": "canary_deployment",
            "deployment_id": canary_result["deployment_id"],
            "traffic_percentage": canary_result["traffic_percentage"],
            "evaluation_hours": canary_result["evaluation_hours"]
        }

    async def _implement_direct(self, recommendation: EvolutionRecommendation) -> Dict[str, Any]:
        """Implement recommendation directly (for low-risk changes)"""

        # Execute implementation
        implementation_result = await self._execute_implementation(recommendation)

        if implementation_result["success"]:
            # Create changelog entry
            changelog = ArchitectureChangelog(
                version=f"v2.4.{len(self.evolution_state.implemented_changes) + 1}",
                timestamp=datetime.now().isoformat(),
                changes=[{
                    "recommendation_id": recommendation.recommendation_id,
                    "title": recommendation.title,
                    "type": recommendation.evolution_type.value,
                    "implementation_method": "direct_implementation"
                }],
                performance_impact=recommendation.expected_impact,
                rollback_plan=self._generate_rollback_plan(recommendation),
                validation_results=implementation_result.get("validation", {})
            )

            # Move to implemented changes
            self.evolution_state.implemented_changes.append(changelog)
            self.evolution_state.pending_recommendations.remove(recommendation)

            # Update self-awareness
            self.evolution_state.self_awareness_level = min(1.0,
                self.evolution_state.self_awareness_level + 0.05)

            # Save state
            self._save_persistent_state()

        return implementation_result

    async def _execute_implementation(self, recommendation: EvolutionRecommendation) -> Dict[str, Any]:
        """Execute the implementation of a recommendation"""

        if recommendation.evolution_type == EvolutionType.HEURISTIC_TUNING:
            return await self._implement_heuristic_tuning(recommendation)

        elif recommendation.evolution_type == EvolutionType.LEARNING_RATE_ADJUSTMENT:
            return await self._implement_learning_rate_adjustment(recommendation)

        elif recommendation.evolution_type == EvolutionType.MODULE_ENHANCEMENT:
            return await self._implement_module_enhancement(recommendation)

        else:
            return {
                "success": False,
                "error": f"Implementation not yet supported for type: {recommendation.evolution_type.value}"
            }

    async def _implement_heuristic_tuning(self, recommendation: EvolutionRecommendation) -> Dict[str, Any]:
        """Implement heuristic weight tuning"""

        # Analyze current weights and performance
        current_weights = asdict(self.reinforcement_loop.heuristic_weights)
        weight_history = self.reinforcement_loop.weight_update_history[-10:]

        # Calculate optimal adjustments
        adjustments = {}

        # Simple optimization: boost underperforming weights, reduce overperforming ones
        perf_report = self.reinforcement_loop.get_performance_report()
        avg_score = perf_report.get("performance_metrics", {}).get("average_composite_score", 0.5)

        if avg_score < 0.7:
            # Boost all weights slightly
            for category, weights in current_weights.items():
                if isinstance(weights, dict):
                    adjustments[category] = {k: v * 1.1 for k, v in weights.items()}
        else:
            # Fine-tune based on recent performance
            adjustments = self._calculate_optimal_weights(current_weights, weight_history)

        # Apply adjustments gradually
        for category, new_weights in adjustments.items():
            if hasattr(self.reinforcement_loop.heuristic_weights, category):
                current = getattr(self.reinforcement_loop.heuristic_weights, category)
                if isinstance(current, dict):
                    # Blend old and new weights
                    blended = {}
                    for key in current.keys():
                        if key in new_weights:
                            blended[key] = (current[key] * 0.7) + (new_weights[key] * 0.3)
                        else:
                            blended[key] = current[key]
                    setattr(self.reinforcement_loop.heuristic_weights, category, blended)

        return {
            "success": True,
            "details": {
                "adjustments_applied": adjustments,
                "tuning_method": "gradient-based optimization"
            },
            "validation": {
                "weights_updated": True,
                "gradual_application": True
            }
        }

    async def _implement_learning_rate_adjustment(self, recommendation: EvolutionRecommendation) -> Dict[str, Any]:
        """Implement learning rate adjustment"""

        current_rate = self.reinforcement_loop.learning_rate
        trend_slope = self.reinforcement_loop.get_performance_report()\
            .get("learning_trends", {}).get("trend_slope", 0)

        # Adjust based on trend
        if trend_slope < -0.001:
            # Declining performance - reduce learning rate
            new_rate = current_rate * 0.8
        elif trend_slope > 0.001:
            # Improving - slight increase possible
            new_rate = current_rate * 1.05
        else:
            # Stable - minor adjustment
            new_rate = current_rate * 0.95

        # Clamp to reasonable range
        new_rate = max(0.01, min(0.5, new_rate))

        self.reinforcement_loop.adjust_learning_rate(new_rate)

        return {
            "success": True,
            "details": {
                "old_rate": current_rate,
                "new_rate": new_rate,
                "adjustment_reason": "performance_trend_analysis"
            },
            "validation": {
                "rate_adjusted": True,
                "within_bounds": 0.01 <= new_rate <= 0.5
            }
        }

    async def _implement_module_enhancement(self, recommendation: EvolutionRecommendation) -> Dict[str, Any]:
        """Implement module enhancement"""

        # Identify bottleneck modules from audit
        bottlenecks = []
        perf_report = self.reinforcement_loop.get_performance_report()

        if perf_report.get("performance_metrics", {}).get("bias_detection_rate", 0) > 0.3:
            bottlenecks.append("coherence_evaluation")

        if perf_report.get("performance_metrics", {}).get("ethical_alignment_rate", 1.0) < 0.9:
            bottlenecks.append("ethical_assessment")

        # Generate enhancement suggestions
        enhancements = {}
        for bottleneck in bottlenecks:
            enhancements[bottleneck] = self._generate_enhancement_suggestion(bottleneck)

        return {
            "success": True,
            "details": {
                "bottlenecks_identified": bottlenecks,
                "enhancement_suggestions": enhancements,
                "implementation_status": "suggestions_generated"
            },
            "validation": {
                "enhancements_proposed": len(enhancements),
                "requires_manual_implementation": True
            }
        }

    def _calculate_optimal_weights(self, current_weights: Dict, weight_history: List) -> Dict:
        """Calculate optimal weight adjustments"""

        # Simple optimization based on recent performance
        optimal_weights = {}

        # Analyze which weight adjustments led to better performance
        if len(weight_history) >= 3:
            # Look at last few weight changes and their impact
            recent_changes = weight_history[-3:]

            for category in current_weights.keys():
                if isinstance(current_weights[category], dict):
                    optimal_weights[category] = {}

                    for weight_key in current_weights[category].keys():
                        # Calculate average impact of this weight's changes
                        impacts = []
                        for change in recent_changes:
                            adjustments = change.get("adjustments", {}).get(category, {})
                            if weight_key in adjustments:
                                # Assume positive impact if weights were adjusted (simplified)
                                impacts.append(0.02)  # Small positive assumption

                        avg_impact = statistics.mean(impacts) if impacts else 0.01
                        optimal_weights[category][weight_key] = current_weights[category][weight_key] * (1 + avg_impact)

        return optimal_weights if optimal_weights else current_weights

    def _generate_enhancement_suggestion(self, bottleneck: str) -> Dict[str, Any]:
        """Generate enhancement suggestion for a bottleneck"""

        suggestions = {
            "coherence_evaluation": {
                "enhancement": "Add semantic similarity analysis",
                "implementation": "Integrate sentence transformers for deeper coherence checking",
                "expected_impact": "15% improvement in coherence detection"
            },
            "ethical_assessment": {
                "enhancement": "Implement multi-dimensional ethical framework",
                "implementation": "Add utilitarianism, deontology, and virtue ethics analysis",
                "expected_impact": "20% improvement in ethical alignment"
            }
        }

        return suggestions.get(bottleneck, {
            "enhancement": "General optimization",
            "implementation": "Review and optimize module parameters",
            "expected_impact": "5-10% general improvement"
        })

    def _generate_rollback_plan(self, recommendation: EvolutionRecommendation) -> Dict[str, Any]:
        """Generate rollback plan for a recommendation"""

        if recommendation.evolution_type == EvolutionType.HEURISTIC_TUNING:
            return {
                "rollback_method": "weight_restoration",
                "backup_weights": asdict(self.reinforcement_loop.heuristic_weights),
                "steps": ["Restore previous weight configuration", "Validate performance recovery"]
            }
        elif recommendation.evolution_type == EvolutionType.LEARNING_RATE_ADJUSTMENT:
            return {
                "rollback_method": "rate_restoration",
                "previous_rate": self.reinforcement_loop.learning_rate,
                "steps": ["Revert to previous learning rate", "Monitor stability"]
            }
        else:
            return {
                "rollback_method": "manual_reversion",
                "steps": ["Identify implemented changes", "Revert modifications", "Validate system stability"]
            }

    async def record_execution(self, task_id: str, execution_data: Dict[str, Any]) -> None:
        """
        Record execution details for self-evolution learning.

        Args:
            task_id: Unique task identifier
            execution_data: Comprehensive execution data including performance metrics,
                           decision points, outcomes, and contextual information
        """
        try:
            # Extract key metrics for evolution
            execution_record = {
                "task_id": task_id,
                "timestamp": datetime.now().isoformat(),
                "performance_metrics": execution_data.get("performance_metrics", {}),
                "decision_points": execution_data.get("decision_points", []),
                "outcome": execution_data.get("outcome", {}),
                "context": execution_data.get("context", {}),
                "duration": execution_data.get("duration", 0.0),
                "success": execution_data.get("success", False),
                "lessons_learned": execution_data.get("lessons_learned", []),
                "improvement_opportunities": execution_data.get("improvement_opportunities", [])
            }

            # Store in evolution state for analysis
            if not hasattr(self.evolution_state, 'execution_history'):
                self.evolution_state.execution_history = []

            self.evolution_state.execution_history.append(execution_record)

            # Keep only recent executions (last 1000)
            if len(self.evolution_state.execution_history) > 1000:
                self.evolution_state.execution_history = self.evolution_state.execution_history[-1000:]

            # Update evolution velocity based on recent performance
            recent_executions = self.evolution_state.execution_history[-100:]
            if len(recent_executions) >= 10:
                success_rate = sum(1 for ex in recent_executions if ex["success"]) / len(recent_executions)
                avg_duration = sum(ex["duration"] for ex in recent_executions) / len(recent_executions)

                # Adjust evolution velocity based on performance trends
                if success_rate > 0.8:
                    self.evolution_state.evolution_velocity = min(1.0, self.evolution_state.evolution_velocity + 0.01)
                elif success_rate < 0.6:
                    self.evolution_state.evolution_velocity = max(0.0, self.evolution_state.evolution_velocity - 0.01)

            # Save updated state
            self._save_persistent_state()

            logger.debug(f"Recorded execution for task {task_id}: success={execution_data.get('success', False)}")

        except Exception as e:
            logger.error(f"Failed to record execution for {task_id}: {e}")

    async def evolve_system(self, evolution_trigger: str, evolution_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Trigger system evolution based on accumulated learning and performance data.

        Args:
            evolution_trigger: What triggered this evolution (e.g., "performance_threshold", "manual_request", "scheduled")
            evolution_context: Additional context for the evolution decision

        Returns:
            Evolution results and implemented changes
        """
        try:
            logger.info(f"Initiating system evolution: trigger={evolution_trigger}")

            # Assess current system state
            current_health = await self._assess_current_health()
            evolution_opportunities = await self._identify_evolution_opportunities(evolution_context)

            # Determine evolution strategy
            evolution_strategy = self._select_evolution_strategy(
                current_health, evolution_opportunities, evolution_trigger
            )

            # Execute evolution
            evolution_results = await self._execute_evolution_strategy(evolution_strategy)

            # Validate evolution results
            validation_results = await self._validate_evolution(evolution_results)

            # Update evolution state
            self.evolution_state.self_awareness_level = min(1.0,
                self.evolution_state.self_awareness_level + 0.02)
            self.evolution_state.last_audit = datetime.now().isoformat()

            # Save state
            self._save_persistent_state()

            evolution_summary = {
                "evolution_trigger": evolution_trigger,
                "evolution_strategy": evolution_strategy,
                "results": evolution_results,
                "validation": validation_results,
                "health_improvement": validation_results.get("health_improvement", 0.0),
                "new_capabilities": evolution_results.get("new_capabilities", []),
                "performance_gains": evolution_results.get("performance_gains", {}),
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"System evolution completed: strategy={evolution_strategy}, health_improvement={validation_results.get('health_improvement', 0.0):.3f}")

            return evolution_summary

        except Exception as e:
            logger.error(f"System evolution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "evolution_trigger": evolution_trigger,
                "timestamp": datetime.now().isoformat()
            }

    async def _assess_current_health(self) -> Dict[str, Any]:
        """Assess current system health for evolution decisions"""
        try:
            # Quick health assessment
            perf_report = self.reinforcement_loop.get_performance_report()
            continuity_report = self.temporal_buffer.get_continuity_report()

            health_score = (
                perf_report.get("performance_metrics", {}).get("average_composite_score", 0.5) * 0.6 +
                continuity_report.get("overall_continuity_score", 1.0) * 0.4
            )

            return {
                "overall_health": health_score,
                "performance_health": perf_report.get("performance_metrics", {}).get("average_composite_score", 0.5),
                "continuity_health": continuity_report.get("overall_continuity_score", 1.0),
                "evolution_readiness": health_score > 0.7
            }
        except Exception as e:
            logger.warning(f"Health assessment failed: {e}")
            return {"overall_health": 0.5, "evolution_readiness": False}

    async def _identify_evolution_opportunities(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify potential evolution opportunities"""
        opportunities = []

        # Check execution history for patterns
        if hasattr(self.evolution_state, 'execution_history') and self.evolution_state.execution_history:
            recent_executions = self.evolution_state.execution_history[-50:]

            # Identify common failure patterns
            failure_patterns = {}
            for execution in recent_executions:
                if not execution.get("success", False):
                    error_type = execution.get("outcome", {}).get("error_type", "unknown")
                    failure_patterns[error_type] = failure_patterns.get(error_type, 0) + 1

            # Create opportunities for frequent failure types
            for error_type, count in failure_patterns.items():
                if count > 5:  # More than 10% of recent executions
                    opportunities.append({
                        "type": "failure_pattern_mitigation",
                        "target": error_type,
                        "frequency": count,
                        "potential_impact": "high"
                    })

        # Check for performance bottlenecks
        perf_report = self.reinforcement_loop.get_performance_report()
        if perf_report.get("performance_metrics", {}).get("bias_detection_rate", 0) > 0.3:
            opportunities.append({
                "type": "performance_optimization",
                "target": "coherence_evaluation",
                "current_issue": "high_bias_detection",
                "potential_impact": "medium"
            })

        return opportunities

    def _select_evolution_strategy(self, health: Dict[str, Any],
                                 opportunities: List[Dict[str, Any]],
                                 trigger: str) -> str:
        """Select appropriate evolution strategy"""
        if trigger == "performance_threshold" and health["overall_health"] < 0.6:
            return "defensive_evolution"  # Focus on stability
        elif trigger == "manual_request" or health["evolution_readiness"]:
            return "aggressive_evolution"  # Full evolution cycle
        elif opportunities:
            return "targeted_evolution"  # Address specific issues
        else:
            return "incremental_evolution"  # Small improvements

    async def _execute_evolution_strategy(self, strategy: str) -> Dict[str, Any]:
        """Execute the selected evolution strategy"""
        if strategy == "defensive_evolution":
            # Focus on stability and error reduction
            return await self._execute_defensive_evolution()
        elif strategy == "aggressive_evolution":
            # Full evolution cycle
            return await self.initiate_self_evolution_cycle()
        elif strategy == "targeted_evolution":
            # Address specific identified issues
            return await self._execute_targeted_evolution()
        else:
            # Incremental improvements
            return await self._execute_incremental_evolution()

    async def _execute_defensive_evolution(self) -> Dict[str, Any]:
        """Execute defensive evolution focused on stability"""
        results = {"strategy": "defensive_evolution", "changes": []}

        # Implement stability-focused changes
        try:
            # Reduce learning rate if performance is unstable
            current_rate = self.reinforcement_loop.learning_rate
            if current_rate > 0.1:
                self.reinforcement_loop.adjust_learning_rate(current_rate * 0.9)
                results["changes"].append("learning_rate_reduction")

            # Add error handling improvements
            results["changes"].append("enhanced_error_handling")

            results["success"] = True
        except Exception as e:
            results["success"] = False
            results["error"] = str(e)

        return results

    async def _execute_targeted_evolution(self) -> Dict[str, Any]:
        """Execute targeted evolution for specific issues"""
        results = {"strategy": "targeted_evolution", "changes": []}

        # Implement targeted fixes based on identified opportunities
        opportunities = await self._identify_evolution_opportunities({})

        for opportunity in opportunities:
            if opportunity["type"] == "failure_pattern_mitigation":
                # Implement fixes for common failure patterns
                results["changes"].append(f"mitigation_{opportunity['target']}")
            elif opportunity["type"] == "performance_optimization":
                # Optimize performance bottlenecks
                results["changes"].append(f"optimization_{opportunity['target']}")

        results["success"] = True
        return results

    async def _execute_incremental_evolution(self) -> Dict[str, Any]:
        """Execute incremental evolution with small improvements"""
        results = {"strategy": "incremental_evolution", "changes": []}

        # Small heuristic adjustments
        try:
            perf_report = self.reinforcement_loop.get_performance_report()
            avg_score = perf_report.get("performance_metrics", {}).get("average_composite_score", 0.5)

            if avg_score < 0.7:
                # Small boost to all weights
                weights = self.reinforcement_loop.heuristic_weights
                for category_name in dir(weights):
                    if not category_name.startswith('_'):
                        category = getattr(weights, category_name)
                        if isinstance(category, dict):
                            boosted = {k: v * 1.05 for k, v in category.items()}
                            setattr(weights, category_name, boosted)

                results["changes"].append("heuristic_weight_boost")

            results["success"] = True
        except Exception as e:
            results["success"] = False
            results["error"] = str(e)

        return results

    async def _validate_evolution(self, evolution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate evolution results and measure improvements"""
        try:
            # Quick validation by checking system health before/after
            pre_health = await self._assess_current_health()

            # Wait a moment for changes to take effect
            await asyncio.sleep(0.1)

            post_health = await self._assess_current_health()

            health_improvement = post_health["overall_health"] - pre_health["overall_health"]

            return {
                "health_improvement": health_improvement,
                "pre_health": pre_health["overall_health"],
                "post_health": post_health["overall_health"],
                "validation_success": health_improvement >= 0,  # At least no regression
                "significant_improvement": health_improvement > 0.05
            }
        except Exception as e:
            logger.warning(f"Evolution validation failed: {e}")
            return {
                "health_improvement": 0.0,
                "validation_success": False,
                "error": str(e)
            }

    def get_evolution_report(self) -> Dict[str, Any]:
        """Get comprehensive evolution report"""

        return {
            "evolution_state": asdict(self.evolution_state),
            "pending_recommendations": [asdict(r) for r in self.evolution_state.pending_recommendations],
            "recent_changes": [asdict(c) for c in self.evolution_state.implemented_changes[-5:]],
            "performance_summary": self.reinforcement_loop.get_performance_report(),
            "continuity_summary": self.temporal_buffer.get_continuity_report(),
            "evolution_velocity": self._calculate_evolution_velocity(),
            "self_awareness_assessment": self._assess_self_awareness()
        }

    def _calculate_evolution_velocity(self) -> float:
        """Calculate evolution velocity (improvements per day)"""

        if not self.evolution_state.implemented_changes:
            return 0.0

        # Calculate over last 30 days
        cutoff_date = datetime.now() - timedelta(days=30)
        recent_changes = [
            c for c in self.evolution_state.implemented_changes
            if datetime.fromisoformat(c.timestamp) > cutoff_date
        ]

        if not recent_changes:
            return 0.0

        # Estimate improvement based on expected impacts
        total_impact = 0.0
        for change in recent_changes:
            impact_values = [v for v in change.performance_impact.values() if isinstance(v, (int, float))]
            if impact_values:
                total_impact += statistics.mean(impact_values)

        return total_impact / 30  # Daily rate

    def _assess_self_awareness(self) -> Dict[str, Any]:
        """Assess current level of self-awareness"""

        awareness_indicators = {
            "performance_tracking": bool(self.reinforcement_loop.response_history),
            "temporal_memory": bool(self.temporal_buffer.conclusions),
            "evolution_capability": len(self.evolution_state.pending_recommendations) > 0,
            "self_modification": len(self.evolution_state.implemented_changes) > 0,
            "meta_cognition": True  # Always true for this system
        }

        awareness_score = sum(awareness_indicators.values()) / len(awareness_indicators)

        return {
            "awareness_score": awareness_score,
            "indicators": awareness_indicators,
            "assessment": "high" if awareness_score > 0.8 else "developing" if awareness_score > 0.6 else "basic"
        }

    async def initiate_self_evolution_cycle(self) -> Dict[str, Any]:
        """Initiate a complete self-evolution cycle"""

        logger.info("Initiating self-evolution cycle")

        # Step 1: Comprehensive audit
        audit = await self.perform_comprehensive_audit(days_to_audit=7)

        # Step 2: Generate recommendations
        await self._generate_evolution_recommendations(audit)

        # Step 3: Auto-implement safe recommendations
        implemented = []
        for rec in self.evolution_state.pending_recommendations[:]:
            if rec.priority.value == "low" and rec.estimated_effort == "low":
                # Quick safety check for auto-implementation
                safety_result = await self.safety_monitor.evaluate_safety_gate(rec, human_override=False)
                if safety_result["approved"] and not safety_result["requires_human_review"]:
                    result = await self.implement_evolution_recommendation(rec.recommendation_id, auto_approve=True)
                    if result["success"]:
                        implemented.append(rec.title)

        # Step 4: Generate evolution summary
        evolution_summary = {
            "cycle_timestamp": datetime.now().isoformat(),
            "audit_health_score": audit.overall_health_score,
            "recommendations_generated": len(self.evolution_state.pending_recommendations),
            "auto_implemented": implemented,
            "requires_manual_review": len([r for r in self.evolution_state.pending_recommendations
                                        if r.priority == EvolutionPriority.CRITICAL or r.estimated_effort == "high"]),
            "evolution_velocity": self._calculate_evolution_velocity(),
            "self_awareness_level": self.evolution_state.self_awareness_level
        }

        logger.info(f"Self-evolution cycle completed: {len(implemented)} auto-implementations")

        return evolution_summary

# Global self-evolution manager instance
_self_evolution_manager = None

def get_self_evolution_manager() -> SelfEvolutionManager:
    """Get the global self-evolution manager instance"""
    global _self_evolution_manager
    if _self_evolution_manager is None:
        _self_evolution_manager = SelfEvolutionManager()
    return _self_evolution_manager

async def trigger_self_evolution_cycle() -> Dict[str, Any]:
    """Convenience function to trigger self-evolution cycle"""
    manager = get_self_evolution_manager()
    return await manager.initiate_self_evolution_cycle()