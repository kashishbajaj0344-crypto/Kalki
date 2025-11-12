# ============================================================
# Kalki v2.4 — canary_deployment_manager.py
# ------------------------------------------------------------
# Canary Deployment & Rollback System: Safe Self-Evolution
# - Shadow evaluation of self-evolution changes (10% traffic)
# - Automatic rollback on health metric degradation
# - A/B testing framework for evolution recommendations
# - Production safety gates with gradual rollout
# ============================================================

import os
import json
import asyncio
import hashlib
import copy
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import statistics
import threading
import time

from modules.utils.logging_config import get_logger
from typing import Any, Dict

logger = get_logger("Kalki.CanaryDeployment")

class DeploymentStatus(Enum):
    """Status of a canary deployment"""
    PENDING = "pending"
    ACTIVE = "active"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"
    EXPIRED = "expired"

class TrafficAllocation(Enum):
    """Traffic allocation strategies"""
    PERCENTAGE = "percentage"  # Fixed percentage
    GRADUAL = "gradual"       # Gradual increase
    ADAPTIVE = "adaptive"     # Based on performance

@dataclass
class CanaryMetrics:
    """Metrics collected during canary deployment"""
    deployment_id: str
    timestamp: str
    health_score: float
    confidence_calibration: float
    interdisciplinary_coverage: float
    coherence_score: float
    safety_violations: int
    response_quality: float
    error_rate: float
    latency_ms: float
    request_count: int

@dataclass
class CanaryDeployment:
    """A canary deployment instance"""
    deployment_id: str
    recommendation_id: str
    created_at: str
    status: DeploymentStatus
    traffic_percentage: float
    traffic_strategy: TrafficAllocation
    baseline_metrics: CanaryMetrics
    canary_metrics: List[CanaryMetrics] = field(default_factory=list)
    evaluation_period_hours: int = 24
    rollback_threshold: float = -0.1  # Health score degradation threshold
    promotion_threshold: float = 0.05  # Health score improvement threshold
    shadow_mode: bool = True  # Don't affect production responses
    rollback_plan: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SafetyGateResult:
    """Result of safety gate evaluation"""
    passed: bool
    score: float
    violations: List[str]
    recommendations: List[str]
    human_review_required: bool
    automated_checks: Dict[str, bool]
    ethical_assessment: Dict[str, Any]

class CanaryDeploymentManager:
    """
    Canary Deployment Manager: Safe Self-Evolution

    Enables safe deployment of self-evolution changes through canary testing,
    shadow evaluation, and automatic rollback mechanisms.
    """

    def __init__(self):
        # Remove direct dependency on evolution_manager to avoid circular import
        # Will be set when needed via method parameters

        # Deployment state
        self.active_deployments: Dict[str, CanaryDeployment] = {}
        self.deployment_history: List[CanaryDeployment] = []

        # Configuration
        self.default_traffic_percentage = 0.1  # 10%
        self.default_evaluation_hours = 24
        self.health_check_interval = 300  # 5 minutes
        self.max_concurrent_deployments = 3

        # Persistence
        self.data_dir = "data/canary_deployments"
        self.state_file = f"{self.data_dir}/deployment_state.json"

        # Monitoring
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None

        # Load existing state
        self._load_persistent_state()

        logger.info("Canary Deployment Manager initialized")

    def _load_persistent_state(self):
        """Load persistent deployment state"""
        try:
            os.makedirs(self.data_dir, exist_ok=True)

            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state_data = json.load(f)

                    # Load active deployments
                    for dep_data in state_data.get('active_deployments', []):
                        deployment = CanaryDeployment(**dep_data)
                        self.active_deployments[deployment.deployment_id] = deployment

                    # Load history
                    self.deployment_history = [
                        CanaryDeployment(**dep) for dep in state_data.get('deployment_history', [])
                    ]

        except Exception as e:
            logger.warning(f"Failed to load persistent deployment state: {e}")

    def _save_persistent_state(self):
        """Save deployment state persistently"""
        try:
            state_data = {
                'active_deployments': [asdict(d) for d in self.active_deployments.values()],
                'deployment_history': [asdict(d) for d in self.deployment_history[-50:]],  # Keep last 50
                'last_updated': datetime.now().isoformat()
            }

            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save persistent deployment state: {e}")

    async def deploy_canary(self, recommendation: Any,
                          traffic_percentage: float = None,
                          evaluation_hours: int = None) -> Dict[str, Any]:
        """
        Deploy an evolution recommendation as a canary

        Args:
            recommendation: The evolution recommendation to deploy
            traffic_percentage: Percentage of traffic to route to canary (default 10%)
            evaluation_hours: Hours to evaluate before decision (default 24)

        Returns:
            Deployment result
        """

        # Check concurrent deployment limits
        if len(self.active_deployments) >= self.max_concurrent_deployments:
            return {
                "success": False,
                "error": f"Maximum concurrent deployments ({self.max_concurrent_deployments}) reached"
            }

        # Generate deployment ID
        deployment_id = f"canary_{recommendation.recommendation_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Get baseline metrics
        baseline_metrics = await self._capture_baseline_metrics()

        # Create deployment
        deployment = CanaryDeployment(
            deployment_id=deployment_id,
            recommendation_id=recommendation.recommendation_id,
            created_at=datetime.now().isoformat(),
            status=DeploymentStatus.PENDING,
            traffic_percentage=traffic_percentage or self.default_traffic_percentage,
            traffic_strategy=TrafficAllocation.PERCENTAGE,
            baseline_metrics=baseline_metrics,
            evaluation_period_hours=evaluation_hours or self.default_evaluation_hours,
            rollback_plan=self._generate_rollback_plan(recommendation),
            metadata={
                "recommendation_title": recommendation.title,
                "evolution_type": recommendation.evolution_type.value,
                "priority": recommendation.priority.value,
                "estimated_effort": recommendation.estimated_effort
            }
        )

        # Safety gate check
        safety_result = await self._evaluate_safety_gate(recommendation)
        if not safety_result.passed:
            deployment.status = DeploymentStatus.FAILED
            deployment.metadata["safety_violations"] = safety_result.violations
            self.deployment_history.append(deployment)
            self._save_persistent_state()

            return {
                "success": False,
                "error": "Safety gate failed",
                "safety_result": asdict(safety_result)
            }

        # Deploy the recommendation
        try:
            await self._apply_recommendation_canary(recommendation, deployment)
            deployment.status = DeploymentStatus.ACTIVE
            self.active_deployments[deployment_id] = deployment

            # Start monitoring
            if not self.monitoring_active:
                self._start_monitoring()

            logger.info(f"Canary deployment started: {deployment_id} ({deployment.traffic_percentage*100:.1f}% traffic)")

            self._save_persistent_state()

            return {
                "success": True,
                "deployment_id": deployment_id,
                "traffic_percentage": deployment.traffic_percentage,
                "evaluation_hours": deployment.evaluation_period_hours,
                "safety_check_passed": True
            }

        except Exception as e:
            logger.error(f"Failed to deploy canary {deployment_id}: {e}")
            deployment.status = DeploymentStatus.FAILED
            self.deployment_history.append(deployment)
            self._save_persistent_state()

            return {
                "success": False,
                "error": f"Deployment failed: {str(e)}"
            }

    async def _capture_baseline_metrics(self) -> CanaryMetrics:
        """Capture baseline performance metrics"""

        # Get current system metrics
        evolution_report = self.evolution_manager.get_evolution_report()

        # Simulate some metrics (in real implementation, these would come from actual monitoring)
        return CanaryMetrics(
            deployment_id="baseline",
            timestamp=datetime.now().isoformat(),
            health_score=evolution_report.get("evolution_state", {}).get("self_awareness_level", 0.7),
            confidence_calibration=0.85,  # Would be calculated from actual confidence distributions
            interdisciplinary_coverage=0.8,  # Would be measured from response diversity
            coherence_score=0.9,  # Would be calculated from coherence analysis
            safety_violations=0,
            response_quality=0.85,
            error_rate=0.02,
            latency_ms=150.0,
            request_count=1000
        )

    async def _evaluate_safety_gate(self, recommendation: Any) -> SafetyGateResult:
        """
        Evaluate safety and ethics gates for a recommendation

        Args:
            recommendation: The recommendation to evaluate

        Returns:
            Safety gate evaluation result
        """

        violations = []
        automated_checks = {}
        ethical_assessment = {}

        # Automated safety checks
        automated_checks["high_priority_human_review"] = (
            recommendation.priority.value in ["critical", "high"]
        )

        automated_checks["architecture_change_safety"] = (
            recommendation.evolution_type.value not in ["architecture_refactor"]
        )

        automated_checks["learning_rate_bounds"] = (
            recommendation.evolution_type.value != "learning_rate_adjustment" or
            recommendation.expected_impact.get("learning_rate", 0) < 0.5  # Max 50% change
        )

        # Ethical assessment
        ethical_assessment["bias_amplification_risk"] = self._assess_bias_risk(recommendation)
        ethical_assessment["autonomy_impact"] = self._assess_autonomy_impact(recommendation)
        ethical_assessment["transparency_maintained"] = self._assess_transparency(recommendation)

        # Determine violations
        if automated_checks["high_priority_human_review"]:
            violations.append("High priority change requires human review")

        if not automated_checks["architecture_change_safety"]:
            violations.append("Architecture refactoring requires manual approval")

        if ethical_assessment["bias_amplification_risk"] > 0.7:
            violations.append("High risk of bias amplification")

        # Calculate overall score
        safety_score = 1.0 - (len(violations) * 0.2)
        safety_score = max(0.0, min(1.0, safety_score))

        # Determine if human review required
        human_review_required = (
            len(violations) > 0 or
            recommendation.priority.value == "critical" or
            ethical_assessment["bias_amplification_risk"] > 0.5
        )

        return SafetyGateResult(
            passed=len(violations) == 0,
            score=safety_score,
            violations=violations,
            recommendations=self._generate_safety_recommendations(violations),
            human_review_required=human_review_required,
            automated_checks=automated_checks,
            ethical_assessment=ethical_assessment
        )

    def _assess_bias_risk(self, recommendation: Any) -> float:
        """Assess risk of bias amplification"""
        # Simple heuristic based on recommendation type
        risk_map = {
            "heuristic_tuning": 0.3,
            "learning_rate_adjustment": 0.2,
            "module_enhancement": 0.4,
            "new_capability": 0.6,
            "architecture_refactor": 0.8
        }
        return risk_map.get(recommendation.evolution_type.value, 0.5)

    def _assess_autonomy_impact(self, recommendation: Any) -> float:
        """Assess impact on system autonomy"""
        # Higher impact for changes that affect decision-making
        impact_map = {
            "heuristic_tuning": 0.7,
            "learning_rate_adjustment": 0.8,
            "module_enhancement": 0.5,
            "new_capability": 0.6,
            "architecture_refactor": 0.9
        }
        return impact_map.get(recommendation.evolution_type.value, 0.5)

    def _assess_transparency(self, recommendation: Any) -> bool:
        """Assess if transparency is maintained"""
        # Most changes should maintain transparency
        return recommendation.evolution_type.value != "architecture_refactor"

    def _generate_safety_recommendations(self, violations: List[str]) -> List[str]:
        """Generate safety recommendations based on violations"""
        recommendations = []

        for violation in violations:
            if "human review" in violation.lower():
                recommendations.append("Schedule human review within 24 hours")
            elif "bias" in violation.lower():
                recommendations.append("Add bias monitoring and mitigation measures")
            elif "architecture" in violation.lower():
                recommendations.append("Conduct thorough testing before architecture changes")

        return recommendations

    async def _apply_recommendation_canary(self, recommendation: Any,
                                         deployment: CanaryDeployment):
        """Apply recommendation in canary mode (shadow deployment)"""

        # For now, this is a placeholder - in real implementation, this would:
        # 1. Create a shadow instance of the system
        # 2. Apply the recommendation changes
        # 3. Route traffic to both baseline and canary systems
        # 4. Compare performance

        logger.info(f"Applied recommendation {recommendation.recommendation_id} in canary mode")

        # Mark as applied in metadata
        deployment.metadata["applied_at"] = datetime.now().isoformat()

    def _generate_rollback_plan(self, recommendation: Any) -> Dict[str, Any]:
        """Generate rollback plan for a recommendation"""

        # This would be more sophisticated in real implementation
        return {
            "rollback_method": "weight_restoration",
            "backup_state": "pre_deployment_backup",
            "steps": [
                "Stop canary traffic routing",
                "Restore previous system state",
                "Validate rollback success",
                "Resume normal operation"
            ],
            "estimated_rollback_time": "5 minutes",
            "data_preservation": True
        }

    def _start_monitoring(self):
        """Start the monitoring thread"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        logger.info("Canary monitoring started")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Run monitoring in event loop
                asyncio.run(self._check_deployments())
                time.sleep(self.health_check_interval)

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)  # Wait before retry

    async def _check_deployments(self):
        """Check status of active deployments"""

        current_time = datetime.now()
        deployments_to_remove = []

        for deployment_id, deployment in self.active_deployments.items():
            try:
                # Check if evaluation period is complete
                created_at = datetime.fromisoformat(deployment.created_at)
                evaluation_end = created_at + timedelta(hours=deployment.evaluation_period_hours)

                if current_time >= evaluation_end:
                    # Evaluation complete - make decision
                    await self._evaluate_deployment(deployment)
                    deployments_to_remove.append(deployment_id)

                else:
                    # Still evaluating - capture metrics
                    await self._capture_canary_metrics(deployment)

            except Exception as e:
                logger.error(f"Error checking deployment {deployment_id}: {e}")
                deployment.status = DeploymentStatus.FAILED
                deployments_to_remove.append(deployment_id)

        # Remove completed deployments
        for deployment_id in deployments_to_remove:
            deployment = self.active_deployments.pop(deployment_id)
            self.deployment_history.append(deployment)

        # Save state
        self._save_persistent_state()

    async def _capture_canary_metrics(self, deployment: CanaryDeployment):
        """Capture metrics for active canary deployment"""

        # Simulate metric capture (in real implementation, this would collect actual metrics)
        metrics = CanaryMetrics(
            deployment_id=deployment.deployment_id,
            timestamp=datetime.now().isoformat(),
            health_score=deployment.baseline_metrics.health_score + (0.05 * len(deployment.canary_metrics)),  # Simulate improvement
            confidence_calibration=deployment.baseline_metrics.confidence_calibration,
            interdisciplinary_coverage=deployment.baseline_metrics.interdisciplinary_coverage,
            coherence_score=deployment.baseline_metrics.coherence_score,
            safety_violations=0,
            response_quality=deployment.baseline_metrics.response_quality + 0.02,
            error_rate=deployment.baseline_metrics.error_rate,
            latency_ms=deployment.baseline_metrics.latency_ms,
            request_count=deployment.baseline_metrics.request_count + 100
        )

        deployment.canary_metrics.append(metrics)

    async def _evaluate_deployment(self, deployment: CanaryDeployment):
        """Evaluate deployment and decide to promote or rollback"""

        if not deployment.canary_metrics:
            logger.warning(f"No metrics collected for deployment {deployment.deployment_id}")
            deployment.status = DeploymentStatus.FAILED
            return

        # Calculate average metrics
        avg_health = statistics.mean(m.health_score for m in deployment.canary_metrics)
        baseline_health = deployment.baseline_metrics.health_score

        health_change = avg_health - baseline_health

        logger.info(f"Evaluating deployment {deployment.deployment_id}: health_change={health_change:.3f}")

        # Decision logic
        if health_change >= deployment.promotion_threshold:
            # Promote to production
            await self._promote_deployment(deployment)
            deployment.status = DeploymentStatus.PROMOTED

        elif health_change <= deployment.rollback_threshold:
            # Rollback
            await self._rollback_deployment(deployment)
            deployment.status = DeploymentStatus.ROLLED_BACK

        else:
            # No clear winner - could extend evaluation or manual review
            deployment.status = DeploymentStatus.EXPIRED
            logger.info(f"Deployment {deployment.deployment_id} expired - no clear improvement")

    async def _promote_deployment(self, deployment: CanaryDeployment):
        """Promote successful canary deployment to production"""

        logger.info(f"Promoting deployment {deployment.deployment_id} to production")

        # In real implementation, this would:
        # 1. Gradually increase traffic to 100%
        # 2. Update production configuration
        # 3. Monitor for any issues during promotion

        deployment.metadata["promoted_at"] = datetime.now().isoformat()

    async def _rollback_deployment(self, deployment: CanaryDeployment):
        """Rollback failed canary deployment"""

        logger.info(f"Rolling back deployment {deployment.deployment_id}")

        # In real implementation, this would:
        # 1. Stop canary traffic
        # 2. Restore baseline configuration
        # 3. Clean up canary resources

        deployment.metadata["rolled_back_at"] = datetime.now().isoformat()

    def get_deployment_status(self, deployment_id: str = None) -> Dict[str, Any]:
        """Get status of deployments"""

        if deployment_id:
            deployment = self.active_deployments.get(deployment_id)
            if deployment:
                return {
                    "deployment_id": deployment_id,
                    "status": deployment.status.value,
                    "traffic_percentage": deployment.traffic_percentage,
                    "created_at": deployment.created_at,
                    "metrics_count": len(deployment.canary_metrics),
                    "metadata": deployment.metadata
                }
            else:
                return {"error": f"Deployment {deployment_id} not found"}
        else:
            return {
                "active_deployments": len(self.active_deployments),
                "total_deployments": len(self.deployment_history),
                "deployments": [
                    {
                        "id": d.deployment_id,
                        "status": d.status.value,
                        "created_at": d.created_at,
                        "traffic_percentage": d.traffic_percentage
                    }
                    for d in list(self.active_deployments.values())[-5:]  # Last 5 active
                ]
            }

    def stop_monitoring(self):
        """Stop the monitoring thread"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Canary monitoring stopped")

# Global canary deployment manager instance
_canary_manager = None

def get_canary_deployment_manager() -> CanaryDeploymentManager:
    """Get the global canary deployment manager instance"""
    global _canary_manager
    if _canary_manager is None:
        _canary_manager = CanaryDeploymentManager()
    return _canary_manager

async def deploy_evolution_canary(recommendation_id: str) -> Dict[str, Any]:
    """Convenience function to deploy an evolution recommendation as canary"""
    # Import here to avoid circular dependency
    from modules.self_evolution_manager import get_self_evolution_manager

    manager = get_canary_deployment_manager()
    evolution_manager = get_self_evolution_manager()

    # Find the recommendation
    pending_recs = evolution_manager.evolution_state.pending_recommendations
    recommendation = next((r for r in pending_recs if r.recommendation_id == recommendation_id), None)

    if not recommendation:
        return {"success": False, "error": f"Recommendation {recommendation_id} not found"}

    return await manager.deploy_canary(recommendation)