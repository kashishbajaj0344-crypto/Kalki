from typing import Dict, Any, List, Optional
import asyncio
import time
from datetime import datetime
from ..base_agent import BaseAgent, AgentCapability, AgentStatus
from .ethics_storage import EthicsStorage


class RiskAssessmentAgent(BaseAgent):
    """Enhanced risk assessment and management agent with cross-validation and adaptive scoring"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="RiskAssessmentAgent",
            capabilities=[AgentCapability.RISK_ASSESSMENT],
            description="Assesses and manages risks with adaptive scoring and cross-validation",
            config=config or {}
        )

        # Risk categories with adaptive weights
        self.risk_categories = {
            "operational": {"weight": 0.25, "description": "System operation and performance risks"},
            "security": {"weight": 0.30, "description": "Security and privacy risks"},
            "ethical": {"weight": 0.20, "description": "Ethical and moral risks"},
            "financial": {"weight": 0.15, "description": "Financial and resource risks"},
            "reputational": {"weight": 0.10, "description": "Reputation and trust risks"}
        }

        # Risk factors with severity mappings
        self.risk_factors = {
            "data_breach": {"category": "security", "base_severity": 0.9, "indicators": ["breach", "leak", "unauthorized access"]},
            "system_failure": {"category": "operational", "base_severity": 0.8, "indicators": ["crash", "downtime", "failure"]},
            "bias_amplification": {"category": "ethical", "base_severity": 0.7, "indicators": ["bias", "discrimination", "unfair"]},
            "resource_exhaustion": {"category": "operational", "base_severity": 0.6, "indicators": ["overload", "exhaustion", "capacity"]},
            "privacy_violation": {"category": "security", "base_severity": 0.8, "indicators": ["privacy", "personal data", "surveillance"]},
            "financial_loss": {"category": "financial", "base_severity": 0.7, "indicators": ["cost", "loss", "expense"]},
            "reputation_damage": {"category": "reputational", "base_severity": 0.6, "indicators": ["reputation", "trust", "credibility"]}
        }

        # Initialize risk storage
        self.risk_storage = EthicsStorage()  # Reuse ethics storage for risk patterns

        # Cross-agent references
        self.ethics_agent = None
        self.simulation_agent = None

        # Adaptive scoring parameters
        self.risk_history = []
        self.factor_weights = {}
        self.context_adjustments = {}

    async def initialize(self) -> bool:
        try:
            self.logger.info(f"{self.name} initialized with {len(self.risk_categories)} risk categories")
            self.logger.info(f"Risk factors: {list(self.risk_factors.keys())}")

            # Load historical risk patterns for adaptation
            patterns = self.risk_storage.get_risk_patterns()
            self._update_factor_weights(patterns)

            return True
        except Exception as e:
            self.logger.exception(f"Failed to initialize {self.name}: {e}")
            return False

    def set_ethics_agent(self, ethics_agent):
        """Set reference to EthicsAgent for cross-validation"""
        self.ethics_agent = ethics_agent
        self.logger.info("Connected to EthicsAgent for cross-validation")

    def set_simulation_agent(self, simulation_agent):
        """Set reference to SimulationVerifierAgent for consequence modeling"""
        self.simulation_agent = simulation_agent
        self.logger.info("Connected to SimulationVerifierAgent for consequence modeling")

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        action = task.get("action")
        params = task.get("params", {})

        try:
            if action == "assess":
                return await self._assess_risk(params)
            elif action == "monitor":
                return await self._monitor_risks(params)
            elif action == "mitigate":
                return await self._mitigate_risks(params)
            elif action == "analyze_trends":
                return await self._analyze_risk_trends(params)
            elif action == "get_risk_history":
                return await self._get_risk_history(params)
            else:
                return {"status": "error", "error": f"Unknown action: {action}"}
        except Exception as e:
            self.logger.exception(f"Risk agent execution error: {e}")
            return {"status": "error", "error": str(e), "action": action}

    async def _assess_risk(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced risk assessment with adaptive scoring and cross-validation"""
        try:
            scenario = params.get("scenario", "")
            factors = params.get("factors", [])
            context = params.get("context", {})

            # Identify risk factors in scenario
            identified_factors = self._identify_risk_factors(scenario, factors)

            # Calculate adaptive risk score
            risk_score = self._calculate_adaptive_risk_score(identified_factors, context)

            # Determine risk level with context consideration
            risk_level = self._determine_risk_level(risk_score, context)

            # Generate mitigation recommendations
            mitigation_required = risk_level in ["high", "critical"]
            mitigation_strategies = self._generate_mitigation_strategies(identified_factors, risk_level)

            # Cross-validate with ethics assessment if available
            ethics_feedback = {}
            if self.ethics_agent:
                ethics_result = await self.ethics_agent._review_ethics({
                    "action_description": scenario,
                    "context": context,
                    "stakeholder_impacts": []
                })
                if ethics_result.get("status") == "success":
                    ethics_feedback = {
                        "ethical_score": ethics_result.get("ethical_score"),
                        "correlation": self._calculate_ethics_correlation(risk_score, ethics_result.get("ethical_score", 0.5)),
                        "ethics_violations": len(ethics_result.get("violations", {}))
                    }

            # Enhance with simulation if available
            simulation_feedback = {}
            if self.simulation_agent:
                sim_result = await self.simulation_agent._verify_simulation({
                    "simulation_data": {
                        "action": scenario,
                        "context": context,
                        "risk_factors": list(identified_factors.keys())
                    }
                })
                if sim_result.get("status") == "success":
                    sim_data = sim_result.get("verification", {})
                    simulation_feedback = {
                        "simulation_risk": sim_data.get("risk_score", 0.5),
                        "confidence": sim_data.get("accuracy_score", 0.5)
                    }

            result = {
                "status": "success",
                "scenario": scenario,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "identified_factors": identified_factors,
                "mitigation_required": mitigation_required,
                "mitigation_strategies": mitigation_strategies,
                "ethics_feedback": ethics_feedback,
                "simulation_feedback": simulation_feedback,
                "context": context,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Store assessment in risk storage
            self.risk_storage.store_assessment(result)

            # Update risk history for adaptation
            self.risk_history.append({
                "score": risk_score,
                "level": risk_level,
                "factors": list(identified_factors.keys()),
                "accepted": not mitigation_required
            })

            # Emit event for real-time monitoring
            await self.emit_event("risk.assessment_completed", {
                "assessment_id": len(self.risk_storage.assessments),
                "risk_score": risk_score,
                "risk_level": risk_level,
                "factors_identified": len(identified_factors),
                "mitigation_required": mitigation_required,
                "ethics_correlation": ethics_feedback.get("correlation", 0)
            })

            return result

        except Exception as e:
            self.logger.exception(f"Risk assessment error: {e}")
            return {"status": "error", "error": str(e)}

    def _identify_risk_factors(self, scenario: str, additional_factors: List[str]) -> Dict[str, Any]:
        """Identify risk factors in scenario with severity analysis"""
        identified = {}

        # Check scenario text for risk indicators
        scenario_lower = scenario.lower()
        for factor_name, factor_info in self.risk_factors.items():
            indicators = factor_info["indicators"]
            matches = sum(1 for indicator in indicators if indicator in scenario_lower)

            if matches > 0:
                severity = factor_info["base_severity"] * (1 + (matches - 1) * 0.1)  # Increase with multiple matches
                identified[factor_name] = {
                    "severity": min(1.0, severity),
                    "category": factor_info["category"],
                    "matches": matches,
                    "description": f"Detected {matches} indicator(s) for {factor_name.replace('_', ' ')}"
                }

        # Add explicitly provided factors
        for factor in additional_factors:
            factor_key = factor.lower().replace(" ", "_")
            if factor_key in self.risk_factors and factor_key not in identified:
                factor_info = self.risk_factors[factor_key]
                identified[factor_key] = {
                    "severity": factor_info["base_severity"],
                    "category": factor_info["category"],
                    "matches": 1,
                    "description": f"Explicitly identified: {factor}"
                }

        return identified

    def _calculate_adaptive_risk_score(self, factors: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate adaptive risk score based on factors and historical patterns"""
        if not factors:
            return 0.1  # Minimal risk if no factors identified

        # Base score from factor severities and category weights
        total_weighted_severity = 0.0
        total_weight = 0.0

        for factor_name, factor_data in factors.items():
            severity = factor_data["severity"]
            category = factor_data["category"]
            category_weight = self.risk_categories[category]["weight"]

            # Apply adaptive factor weight if available
            factor_weight = self.factor_weights.get(factor_name, 1.0)

            weighted_severity = severity * category_weight * factor_weight
            total_weighted_severity += weighted_severity
            total_weight += category_weight

        base_score = total_weighted_severity / total_weight if total_weight > 0 else 0.5

        # Apply context adjustments
        context_multiplier = self._calculate_context_multiplier(context)
        adjusted_score = base_score * context_multiplier

        # Apply historical adjustment
        historical_adjustment = self._calculate_historical_adjustment(factors)
        final_score = adjusted_score * (1 + historical_adjustment)

        return max(0.0, min(1.0, final_score))

    def _calculate_context_multiplier(self, context: Dict[str, Any]) -> float:
        """Calculate context-based risk multiplier"""
        multiplier = 1.0

        # Environment adjustments
        if context.get("environment") == "production":
            multiplier *= 1.3
        elif context.get("environment") == "development":
            multiplier *= 0.7

        # Scale adjustments
        stakeholders = context.get("stakeholders_count", 1)
        if stakeholders > 1000:
            multiplier *= 1.4
        elif stakeholders > 100:
            multiplier *= 1.2
        elif stakeholders < 10:
            multiplier *= 0.8

        # Time sensitivity
        if context.get("urgency") == "high":
            multiplier *= 1.2
        elif context.get("time_available") == "limited":
            multiplier *= 1.1

        # Resource availability
        if context.get("resources_available") == "limited":
            multiplier *= 1.15

        return multiplier

    def _calculate_historical_adjustment(self, factors: Dict[str, Any]) -> float:
        """Calculate adjustment based on historical risk patterns"""
        if not self.risk_history:
            return 0.0

        # Find similar historical assessments
        factor_names = set(factors.keys())
        similar_assessments = []

        for hist in self.risk_history[-20:]:  # Last 20 assessments
            hist_factors = set(hist["factors"])
            overlap = len(factor_names.intersection(hist_factors))
            if overlap > 0:
                similarity = overlap / len(factor_names.union(hist_factors))
                if similarity > 0.3:  # 30% factor overlap threshold
                    similar_assessments.append(hist)

        if not similar_assessments:
            return 0.0

        # Calculate average historical score adjustment
        avg_hist_score = sum(h["score"] for h in similar_assessments) / len(similar_assessments)
        current_base = sum(f["severity"] for f in factors.values()) / len(factors)

        adjustment = (avg_hist_score - current_base) * 0.2  # 20% weight to historical patterns
        return max(-0.3, min(0.3, adjustment))  # Limit adjustment range

    def _determine_risk_level(self, risk_score: float, context: Dict[str, Any]) -> str:
        """Determine risk level with context consideration"""
        # Base thresholds
        if risk_score >= 0.8:
            level = "critical"
        elif risk_score >= 0.6:
            level = "high"
        elif risk_score >= 0.4:
            level = "medium"
        elif risk_score >= 0.2:
            level = "low"
        else:
            level = "minimal"

        # Context adjustments
        if context.get("tolerance") == "low" and level in ["low", "minimal"]:
            level = "medium"
        elif context.get("tolerance") == "high" and level == "medium":
            level = "low"

        return level

    def _generate_mitigation_strategies(self, factors: Dict[str, Any], risk_level: str) -> List[str]:
        """Generate mitigation strategies based on identified factors and risk level"""
        strategies = []

        # Factor-specific strategies
        factor_strategies = {
            "data_breach": ["Implement encryption", "Access controls", "Regular security audits"],
            "system_failure": ["Redundancy systems", "Monitoring and alerts", "Failover mechanisms"],
            "bias_amplification": ["Bias detection algorithms", "Diverse training data", "Human oversight"],
            "resource_exhaustion": ["Load balancing", "Resource monitoring", "Auto-scaling"],
            "privacy_violation": ["Privacy impact assessment", "Data minimization", "Consent management"],
            "financial_loss": ["Cost monitoring", "Budget controls", "Financial risk assessment"],
            "reputation_damage": ["Crisis communication plan", "Transparency measures", "Stakeholder engagement"]
        }

        for factor_name in factors.keys():
            if factor_name in factor_strategies:
                strategies.extend(factor_strategies[factor_name])

        # Risk level-specific strategies
        level_strategies = {
            "critical": ["Immediate executive review", "Full risk assessment", "Contingency planning"],
            "high": ["Senior management approval", "Enhanced monitoring", "Phased implementation"],
            "medium": ["Department head approval", "Additional testing", "Documentation review"],
            "low": ["Standard procedures", "Basic monitoring"],
            "minimal": ["Routine oversight"]
        }

        if risk_level in level_strategies:
            strategies.extend(level_strategies[risk_level])

        # Remove duplicates while preserving order
        seen = set()
        unique_strategies = []
        for strategy in strategies:
            if strategy not in seen:
                unique_strategies.append(strategy)
                seen.add(strategy)

        return unique_strategies[:8]  # Limit to top 8 strategies

    def _calculate_ethics_correlation(self, risk_score: float, ethics_score: float) -> float:
        """Calculate correlation between risk and ethics scores"""
        # Risk and ethics should be inversely correlated (high risk = low ethics)
        expected_ethics = 1 - risk_score
        diff = abs(ethics_score - expected_ethics)
        return max(0.0, 1.0 - diff * 2)  # Scale difference to correlation

    async def _monitor_risks(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced risk monitoring with trend analysis"""
        try:
            time_window = params.get("time_window", "24h")

            # Get risk patterns from storage
            patterns = self.risk_storage.get_risk_patterns()

            # Analyze current risk landscape
            active_risks = len([p for p in patterns.values() if p.get("recent_count", 0) > 0])
            high_severity = len([p for p in patterns.values() if p.get("avg_severity", 0) > 0.7])

            # Calculate risk trends
            trend_analysis = self._analyze_risk_trends(patterns)

            monitoring_results = {
                "active_risks": active_risks,
                "high_severity_risks": high_severity,
                "new_risks": trend_analysis.get("new_risks", 0),
                "mitigated_risks": trend_analysis.get("mitigated_risks", 0),
                "high_priority": self._identify_high_priority_risks(patterns),
                "status": "under_control" if active_risks < 5 else "requires_attention",
                "trend": trend_analysis.get("overall_trend", "stable"),
                "time_window": time_window
            }

            return {"status": "success", "monitoring": monitoring_results}

        except Exception as e:
            self.logger.exception(f"Risk monitoring error: {e}")
            return {"status": "error", "error": str(e)}

    def _analyze_risk_trends(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze risk trends from patterns"""
        trends = {
            "new_risks": 0,
            "mitigated_risks": 0,
            "increasing": [],
            "decreasing": [],
            "overall_trend": "stable"
        }

        for pattern_name, pattern_data in patterns.items():
            recent_count = pattern_data.get("recent_count", 0)
            total_count = pattern_data.get("count", 0)

            if total_count == recent_count and recent_count > 0:
                trends["new_risks"] += 1
            elif recent_count == 0 and total_count > 0:
                trends["mitigated_risks"] += 1

            # Simple trend analysis based on severity
            avg_severity = pattern_data.get("severity_sum", 0) / max(total_count, 1)
            if avg_severity > 0.7:
                trends["increasing"].append(pattern_name)
            elif avg_severity < 0.3:
                trends["decreasing"].append(pattern_name)

        # Determine overall trend
        if len(trends["increasing"]) > len(trends["decreasing"]):
            trends["overall_trend"] = "increasing"
        elif len(trends["decreasing"]) > len(trends["increasing"]):
            trends["overall_trend"] = "decreasing"

        return trends

    def _identify_high_priority_risks(self, patterns: Dict[str, Any]) -> List[str]:
        """Identify high priority risks based on patterns"""
        high_priority = []

        for pattern_name, pattern_data in patterns.items():
            avg_severity = pattern_data.get("severity_sum", 0) / max(pattern_data.get("count", 1), 1)
            recent_count = pattern_data.get("recent_count", 0)

            # High priority criteria
            if avg_severity > 0.8 or (avg_severity > 0.6 and recent_count > 2):
                risk_display = pattern_name.replace("_", " ").title()
                high_priority.append(f"{risk_display} (severity: {avg_severity:.2f})")

        return high_priority[:5]  # Top 5 high priority risks

    async def _mitigate_risks(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced risk mitigation with adaptive strategies"""
        try:
            risk_type = params.get("risk_type", "")
            context = params.get("context", {})

            # Get historical mitigation effectiveness
            mitigation_history = self._get_mitigation_history(risk_type)

            # Generate adaptive mitigation plan
            mitigation_plan = {
                "risk_type": risk_type,
                "strategies": self._generate_adaptive_mitigation(risk_type, context, mitigation_history),
                "priority": self._calculate_mitigation_priority(risk_type, context),
                "timeline": self._estimate_mitigation_timeline(risk_type, context),
                "resources_required": self._estimate_resources_required(risk_type),
                "success_probability": mitigation_history.get("avg_success", 0.7),
                "monitoring_plan": self._create_monitoring_plan(risk_type)
            }

            return {"status": "success", "mitigation_plan": mitigation_plan}

        except Exception as e:
            self.logger.exception(f"Risk mitigation error: {e}")
            return {"status": "error", "error": str(e)}

    def _generate_adaptive_mitigation(self, risk_type: str, context: Dict[str, Any],
                                    history: Dict[str, Any]) -> List[str]:
        """Generate adaptive mitigation strategies based on history and context"""
        base_strategies = self._generate_mitigation_strategies({risk_type: {"severity": 0.5}}, "medium")

        # Adapt based on historical success
        if history.get("avg_success", 0.5) < 0.6:
            # If historical mitigation wasn't successful, add more robust strategies
            base_strategies.extend([
                "Third-party expert consultation",
                "Comprehensive risk reassessment",
                "Extended testing and validation"
            ])

        # Context-specific adaptations
        if context.get("environment") == "production":
            base_strategies.insert(0, "Implement in staging environment first")
        if context.get("urgency") == "high":
            base_strategies.append("Parallel mitigation approaches")

        return list(set(base_strategies))  # Remove duplicates

    def _calculate_mitigation_priority(self, risk_type: str, context: Dict[str, Any]) -> str:
        """Calculate mitigation priority"""
        if risk_type in ["data_breach", "system_failure"] or context.get("urgency") == "critical":
            return "critical"
        elif context.get("stakeholders_count", 0) > 100 or context.get("environment") == "production":
            return "high"
        else:
            return "medium"

    def _estimate_mitigation_timeline(self, risk_type: str, context: Dict[str, Any]) -> str:
        """Estimate mitigation timeline"""
        if context.get("urgency") == "critical":
            return "immediate (hours)"
        elif risk_type in ["data_breach", "system_failure"]:
            return "urgent (1-2 days)"
        elif context.get("complexity") == "high":
            return "extended (1-2 weeks)"
        else:
            return "standard (3-5 days)"

    def _estimate_resources_required(self, risk_type: str) -> List[str]:
        """Estimate resources required for mitigation"""
        resource_map = {
            "data_breach": ["Security team", "Legal counsel", "IT infrastructure"],
            "system_failure": ["DevOps team", "Infrastructure resources", "Testing environment"],
            "bias_amplification": ["Data science team", "Ethics committee", "Training data"],
            "resource_exhaustion": ["Infrastructure team", "Monitoring tools", "Load balancers"],
            "privacy_violation": ["Privacy officer", "Legal team", "Compliance tools"],
            "financial_loss": ["Finance team", "Audit resources", "Cost monitoring tools"],
            "reputation_damage": ["Communications team", "PR resources", "Stakeholder management"]
        }

        return resource_map.get(risk_type, ["Cross-functional team", "Technical resources"])

    def _create_monitoring_plan(self, risk_type: str) -> Dict[str, Any]:
        """Create monitoring plan for mitigated risk"""
        return {
            "metrics": ["Risk recurrence", "Mitigation effectiveness", "Impact reduction"],
            "frequency": "daily" if risk_type in ["data_breach", "system_failure"] else "weekly",
            "duration": "30 days",
            "escalation_triggers": ["Risk reoccurrence", "Ineffective mitigation", "New related risks"]
        }

    def _get_mitigation_history(self, risk_type: str) -> Dict[str, Any]:
        """Get historical mitigation effectiveness for risk type"""
        # This would typically query a mitigation database
        # For now, return mock data
        return {
            "attempts": 5,
            "successful": 3,
            "avg_success": 0.6,
            "common_strategies": ["monitoring", "redundancy", "testing"]
        }

    async def _analyze_risk_trends(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze risk trends over time"""
        try:
            period = params.get("period", "30d")
            patterns = self.risk_storage.get_risk_patterns()

            trend_analysis = {
                "period": period,
                "patterns": {},
                "overall_trend": "stable",
                "recommendations": []
            }

            for pattern_name, pattern_data in patterns.items():
                trend = self._calculate_pattern_trend(pattern_data)
                trend_analysis["patterns"][pattern_name] = trend

            # Determine overall trend
            increasing = sum(1 for t in trend_analysis["patterns"].values() if t.get("direction") == "increasing")
            decreasing = sum(1 for t in trend_analysis["patterns"].values() if t.get("direction") == "decreasing")

            if increasing > decreasing * 1.5:
                trend_analysis["overall_trend"] = "worsening"
            elif decreasing > increasing * 1.5:
                trend_analysis["overall_trend"] = "improving"

            # Generate recommendations
            trend_analysis["recommendations"] = self._generate_trend_recommendations(trend_analysis)

            return {"status": "success", "trend_analysis": trend_analysis}

        except Exception as e:
            self.logger.exception(f"Risk trend analysis error: {e}")
            return {"status": "error", "error": str(e)}

    def _calculate_pattern_trend(self, pattern_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate trend for a specific risk pattern"""
        # Simple trend calculation based on recent vs historical data
        total_count = pattern_data.get("count", 0)
        recent_count = pattern_data.get("recent_count", 0)
        avg_severity = pattern_data.get("severity_sum", 0) / max(total_count, 1)

        if total_count < 2:
            return {"direction": "insufficient_data", "confidence": 0.0}

        recent_ratio = recent_count / total_count
        direction = "stable"

        if recent_ratio > 0.6:
            direction = "increasing"
        elif recent_ratio < 0.2:
            direction = "decreasing"

        confidence = min(1.0, total_count / 10)  # More data = higher confidence

        return {
            "direction": direction,
            "confidence": confidence,
            "recent_ratio": recent_ratio,
            "avg_severity": avg_severity
        }

    def _generate_trend_recommendations(self, trend_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on trend analysis"""
        recommendations = []

        if trend_analysis["overall_trend"] == "worsening":
            recommendations.extend([
                "Increase monitoring frequency",
                "Review risk mitigation strategies",
                "Consider additional preventive measures"
            ])
        elif trend_analysis["overall_trend"] == "improving":
            recommendations.extend([
                "Continue current mitigation approaches",
                "Document successful strategies",
                "Gradually reduce monitoring intensity"
            ])

        # Pattern-specific recommendations
        for pattern_name, trend in trend_analysis["patterns"].items():
            if trend.get("direction") == "increasing" and trend.get("confidence", 0) > 0.7:
                recommendations.append(f"Address increasing {pattern_name.replace('_', ' ')} risk")

        return recommendations[:5]

    async def _get_risk_history(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get risk assessment history"""
        try:
            limit = params.get("limit", 10)
            history = self.risk_storage.get_evaluation_history(limit)

            # Filter for risk assessments
            risk_history = [h for h in history if "risk_score" in h]

            return {
                "status": "success",
                "history": risk_history,
                "total_assessments": len(risk_history)
            }

        except Exception as e:
            self.logger.exception(f"Risk history retrieval error: {e}")
            return {"status": "error", "error": str(e)}

    def _update_factor_weights(self, patterns: Dict[str, Any]):
        """Update factor weights based on historical patterns"""
        for pattern_name, pattern_data in patterns.items():
            if "count" in pattern_data and pattern_data["count"] > 0:
                # Weight factors that appear frequently higher
                frequency_weight = min(2.0, pattern_data["count"] / 10)
                # Weight factors with high severity higher
                severity_weight = pattern_data.get("severity_sum", 0) / pattern_data["count"]

                self.factor_weights[pattern_name] = (frequency_weight + severity_weight) / 2

    async def shutdown(self) -> bool:
        """Shutdown the RiskAssessmentAgent gracefully"""
        try:
            self.logger.info(f"{self.name} shutting down")
            # Save any pending assessments
            self.risk_storage._save_data()
            return True
        except Exception as e:
            self.logger.exception(f"Failed to shutdown {self.name}: {e}")
            return False

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        patterns = self.risk_storage.get_risk_patterns()
        return {
            "total_patterns": len(patterns),
            "high_severity_patterns": len([p for p in patterns.values() if p.get("avg_severity", 0) > 0.7]),
            "active_risks": len([p for p in patterns.values() if p.get("recent_count", 0) > 0]),
            "categories": {cat: sum(1 for p in patterns.values() if p.get("category") == cat) for cat in self.risk_categories.keys()}
        }


__all__ = ["RiskAssessmentAgent"]
