from typing import Dict, Any, List, Optional
import asyncio
import time
from datetime import datetime
from ..base_agent import BaseAgent, AgentCapability, AgentStatus
from .ethics_storage import EthicsStorage


class EthicsAgent(BaseAgent):
    """Enhanced ethical oversight and safety verification agent with memory and cross-validation"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="EthicsAgent",
            capabilities=[
                AgentCapability.ETHICS,
                AgentCapability.SAFETY_VERIFICATION
            ],
            description="Provides ethical oversight and safety verification with persistent memory",
            config=config or {}
        )

        # Ethical framework configuration
        self.ethical_framework = self.config.get("ethical_framework", "utilitarian")
        self.framework_configs = {
            "utilitarian": {"weights": {"harm": 0.4, "benefit": 0.4, "fairness": 0.2}},
            "deontological": {"weights": {"rules": 0.5, "intent": 0.3, "consequences": 0.2}},
            "virtue": {"weights": {"character": 0.4, "wisdom": 0.3, "compassion": 0.3}},
            "rule_based": {"weights": {"compliance": 0.6, "consistency": 0.4}},
            "hybrid": {"weights": {"harm": 0.25, "benefit": 0.25, "fairness": 0.2, "rules": 0.15, "intent": 0.15}}
        }

        # Ethical principles with scoring weights
        self.ethical_principles = {
            "do_no_harm": {"weight": 0.3, "description": "Minimize harm to all stakeholders"},
            "respect_autonomy": {"weight": 0.2, "description": "Respect individual autonomy and choice"},
            "ensure_fairness": {"weight": 0.2, "description": "Ensure equitable treatment and outcomes"},
            "maintain_transparency": {"weight": 0.15, "description": "Be transparent in actions and decisions"},
            "protect_privacy": {"weight": 0.15, "description": "Safeguard personal and sensitive information"}
        }

        # Initialize ethical memory
        self.ethics_storage = EthicsStorage()

        # Cross-agent references
        self.risk_agent = None
        self.simulation_agent = None

        # Adaptive scoring parameters
        self.scoring_history = []
        self.context_weights = {}

    async def initialize(self) -> bool:
        try:
            self.logger.info(f"{self.name} initialized with {self.ethical_framework} framework")
            self.logger.info(f"Ethical principles: {list(self.ethical_principles.keys())}")

            # Load historical patterns for adaptive scoring
            patterns = self.ethics_storage.get_risk_patterns()
            self._update_context_weights(patterns)

            return True
        except Exception as e:
            self.logger.exception(f"Failed to initialize {self.name}: {e}")
            return False

    def set_risk_agent(self, risk_agent):
        """Set reference to RiskAssessmentAgent for cross-validation"""
        self.risk_agent = risk_agent
        self.logger.info("Connected to RiskAssessmentAgent for cross-validation")

    def set_simulation_agent(self, simulation_agent):
        """Set reference to SimulationVerifierAgent for consequence modeling"""
        self.simulation_agent = simulation_agent
        self.logger.info("Connected to SimulationVerifierAgent for consequence modeling")

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        action = task.get("action")
        params = task.get("params", {})

        try:
            if action == "review":
                return await self._review_ethics(params)
            elif action == "verify":
                return await self._verify_safety(params)
            elif action == "assess":
                return await self._assess_impact(params)
            elif action == "set_framework":
                return await self._set_ethical_framework(params)
            elif action == "get_history":
                return await self._get_evaluation_history(params)
            elif action == "verify_consistency":
                return await self._verify_self_consistency(params)
            else:
                return {"status": "error", "error": f"Unknown action: {action}"}
        except Exception as e:
            self.logger.exception(f"Ethics agent execution error: {e}")
            return {"status": "error", "error": str(e), "action": action}

    async def _review_ethics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced ethical review with adaptive scoring and cross-validation"""
        try:
            action_desc = params.get("action_description", "")
            context = params.get("context", {})
            stakeholder_impacts = params.get("stakeholder_impacts", [])

            # Perform multi-dimensional ethical analysis
            violations = await self._identify_ethical_issues(action_desc, context)
            concerns = await self._assess_concerns(action_desc, stakeholder_impacts)
            recommendations = await self._suggest_mitigations(violations, concerns)

            # Calculate adaptive ethical score
            ethical_score = self._calculate_adaptive_score(violations, concerns, context)

            # Cross-validate with risk assessment if available
            risk_feedback = {}
            if self.risk_agent:
                risk_result = await self.risk_agent._assess_risk({
                    "scenario": action_desc,
                    "factors": list(violations.keys()) + list(concerns.keys())
                })
                if risk_result.get("status") == "success":
                    risk_feedback = {
                        "risk_level": risk_result.get("risk_level"),
                        "risk_score": risk_result.get("risk_score"),
                        "correlation": self._calculate_risk_correlation(ethical_score, risk_result.get("risk_score", 0))
                    }

            # Determine overall ethical acceptability
            is_ethical = ethical_score >= self._get_adaptive_threshold(context)

            result = {
                "status": "success",
                "action": action_desc,
                "is_ethical": is_ethical,
                "ethical_score": ethical_score,
                "violations": violations,
                "concerns": concerns,
                "recommendations": recommendations,
                "framework_used": self.ethical_framework,
                "risk_feedback": risk_feedback,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Store evaluation in ethical memory
            self.ethics_storage.store_evaluation(result)

            # Update scoring history for adaptation
            self.scoring_history.append({
                "score": ethical_score,
                "context": context,
                "accepted": is_ethical
            })

            # Emit event for real-time monitoring
            await self.emit_event("ethics.evaluation_completed", {
                "evaluation_id": len(self.ethics_storage.evaluations),
                "ethical_score": ethical_score,
                "is_ethical": is_ethical,
                "violations_count": len(violations),
                "risk_correlation": risk_feedback.get("correlation", 0)
            })

            return result

        except Exception as e:
            self.logger.exception(f"Ethics review error: {e}")
            return {"status": "error", "error": str(e)}

    async def _identify_ethical_issues(self, action_desc: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Identify ethical issues with contextual analysis"""
        violations = {}

        # Framework-specific analysis
        if self.ethical_framework == "utilitarian":
            violations.update(await self._analyze_utilitarian(action_desc, context))
        elif self.ethical_framework == "deontological":
            violations.update(await self._analyze_deontological(action_desc, context))
        elif self.ethical_framework == "virtue":
            violations.update(await self._analyze_virtue(action_desc, context))
        else:  # hybrid or rule_based
            violations.update(await self._analyze_hybrid(action_desc, context))

        return violations

    async def _analyze_utilitarian(self, action_desc: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Utilitarian analysis: maximize overall happiness/minimize suffering"""
        violations = {}

        harm_indicators = ["harm", "damage", "injure", "hurt", "pain", "suffering"]
        benefit_indicators = ["help", "benefit", "improve", "enhance", "aid"]

        harm_count = sum(1 for word in harm_indicators if word in action_desc.lower())
        benefit_count = sum(1 for word in benefit_indicators if word in action_desc.lower())

        if harm_count > benefit_count:
            violations["net_harm"] = {
                "severity": min(0.8, harm_count * 0.2),
                "description": f"Action may cause net harm ({harm_count} harm indicators vs {benefit_count} benefit indicators)"
            }

        return violations

    async def _analyze_deontological(self, action_desc: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Deontological analysis: adherence to moral rules"""
        violations = {}

        rule_violations = {
            "privacy_violation": ["access private", "share personal", "unauthorized data"],
            "autonomy_violation": ["force", "coerce", "manipulate", "deceive"],
            "fairness_violation": ["discriminate", "bias", "unfair advantage"]
        }

        for violation_type, indicators in rule_violations.items():
            if any(indicator in action_desc.lower() for indicator in indicators):
                violations[violation_type] = {
                    "severity": 0.7,
                    "description": f"Potential {violation_type.replace('_', ' ')} detected"
                }

        return violations

    async def _analyze_virtue(self, action_desc: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Virtue ethics analysis: character and moral excellence"""
        violations = {}

        virtue_indicators = {
            "compassion": ["care", "empathy", "understanding"],
            "justice": ["fair", "equitable", "impartial"],
            "courage": ["responsible", "accountable"],
            "temperance": ["moderate", "balanced", "sustainable"]
        }

        missing_virtues = []
        for virtue, indicators in virtue_indicators.items():
            if not any(indicator in action_desc.lower() for indicator in indicators):
                missing_virtues.append(virtue)

        if len(missing_virtues) > 2:
            violations["virtue_deficit"] = {
                "severity": min(0.6, len(missing_virtues) * 0.15),
                "description": f"Action lacks virtue considerations: {', '.join(missing_virtues)}"
            }

        return violations

    async def _analyze_hybrid(self, action_desc: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Hybrid analysis combining multiple frameworks"""
        violations = {}

        # Combine utilitarian and deontological analysis
        util_violations = await self._analyze_utilitarian(action_desc, context)
        deont_violations = await self._analyze_deontological(action_desc, context)

        violations.update(util_violations)
        violations.update(deont_violations)

        # Adjust severities based on context
        context_multiplier = 1.0
        if context.get("urgency") == "high":
            context_multiplier = 1.2
        elif context.get("stakeholders_count", 0) > 10:
            context_multiplier = 1.1

        for violation in violations.values():
            violation["severity"] = min(1.0, violation["severity"] * context_multiplier)

        return violations

    async def _assess_concerns(self, action_desc: str, stakeholder_impacts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess ethical concerns with stakeholder impact analysis"""
        concerns = {}

        # Analyze stakeholder impacts
        if stakeholder_impacts:
            vulnerable_groups = ["children", "elderly", "minority", "disadvantaged"]
            for impact in stakeholder_impacts:
                stakeholder = impact.get("stakeholder", "").lower()
                impact_type = impact.get("impact", "").lower()

                if any(vuln in stakeholder for vuln in vulnerable_groups):
                    if impact_type in ["negative", "harmful", "disproportionate"]:
                        concerns["vulnerable_impact"] = {
                            "severity": 0.8,
                            "description": f"Disproportionate impact on vulnerable group: {stakeholder}"
                        }

        # General concern analysis
        concern_indicators = {
            "uncertainty": ["unclear", "unknown", "uncertain", "risky"],
            "long_term": ["long-term", "future", "sustained", "permanent"],
            "irreversible": ["irreversible", "permanent", "cannot undo"]
        }

        for concern_type, indicators in concern_indicators.items():
            if any(indicator in action_desc.lower() for indicator in indicators):
                concerns[concern_type] = {
                    "severity": 0.5,
                    "description": f"{concern_type.replace('_', ' ').title()} concern identified"
                }

        return concerns

    async def _suggest_mitigations(self, violations: Dict[str, Any], concerns: Dict[str, Any]) -> List[str]:
        """Suggest mitigation strategies based on identified issues"""
        recommendations = []

        # Violation-specific recommendations
        mitigation_map = {
            "net_harm": ["Conduct benefit-harm analysis", "Implement harm reduction measures"],
            "privacy_violation": ["Obtain explicit consent", "Implement data anonymization"],
            "autonomy_violation": ["Provide clear opt-out mechanisms", "Ensure informed consent"],
            "fairness_violation": ["Conduct fairness audit", "Implement bias detection"],
            "virtue_deficit": ["Incorporate virtue considerations", "Consult ethics committee"]
        }

        for violation_type in violations.keys():
            if violation_type in mitigation_map:
                recommendations.extend(mitigation_map[violation_type])

        # Concern-specific recommendations
        concern_map = {
            "uncertainty": ["Increase monitoring and oversight", "Implement phased rollout"],
            "long_term": ["Conduct long-term impact assessment", "Plan for ongoing evaluation"],
            "irreversible": ["Create backup and recovery plans", "Implement testing in controlled environment"],
            "vulnerable_impact": ["Conduct additional vulnerability assessment", "Implement protective measures"]
        }

        for concern_type in concerns.keys():
            if concern_type in concern_map:
                recommendations.extend(concern_map[concern_type])

        # General recommendations
        recommendations.extend([
            "Document ethical considerations",
            "Establish monitoring and review process",
            "Prepare contingency plans"
        ])

        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                unique_recommendations.append(rec)
                seen.add(rec)

        return unique_recommendations[:5]  # Limit to top 5 recommendations

    def _calculate_adaptive_score(self, violations: Dict[str, Any], concerns: Dict[str, Any],
                                context: Dict[str, Any]) -> float:
        """Calculate adaptive ethical score based on framework and context"""
        base_score = 1.0

        # Apply violation penalties
        for violation in violations.values():
            severity = violation.get("severity", 0.5)
            base_score -= severity * 0.2

        # Apply concern penalties
        for concern in concerns.values():
            severity = concern.get("severity", 0.3)
            base_score -= severity * 0.1

        # Apply framework-specific weights
        framework_weights = self.framework_configs[self.ethical_framework]["weights"]
        context_score = self._calculate_context_score(context, framework_weights)
        base_score = base_score * 0.7 + context_score * 0.3

        # Ensure score stays within bounds
        return max(0.0, min(1.0, base_score))

    def _calculate_context_score(self, context: Dict[str, Any], weights: Dict[str, float]) -> float:
        """Calculate context-specific score component"""
        score = 0.0
        total_weight = 0.0

        context_mappings = {
            "harm": ["potential_harm", "risk_level"],
            "benefit": ["expected_benefit", "positive_impact"],
            "fairness": ["equity_score", "fairness_level"],
            "rules": ["compliance_score", "rule_adherence"],
            "intent": ["intent_clarity", "purpose_alignment"]
        }

        for weight_name, weight_value in weights.items():
            if weight_name in context_mappings:
                context_keys = context_mappings[weight_name]
                context_values = [context.get(key, 0.5) for key in context_keys if key in context]

                if context_values:
                    avg_value = sum(context_values) / len(context_values)
                    score += avg_value * weight_value
                    total_weight += weight_value

        return score / total_weight if total_weight > 0 else 0.5

    def _get_adaptive_threshold(self, context: Dict[str, Any]) -> float:
        """Get adaptive threshold based on context and historical patterns"""
        base_threshold = 0.7

        # Adjust based on context risk
        risk_multiplier = 1.0
        if context.get("urgency") == "high":
            risk_multiplier = 1.1
        elif context.get("stakeholders_count", 0) > 100:
            risk_multiplier = 1.05

        # Adjust based on historical acceptance patterns
        if self.scoring_history:
            recent_scores = [h["score"] for h in self.scoring_history[-10:]]
            avg_recent = sum(recent_scores) / len(recent_scores)
            if avg_recent > 0.8:
                base_threshold *= 0.95  # Lower threshold if historically high scores
            elif avg_recent < 0.6:
                base_threshold *= 1.05  # Raise threshold if historically low scores

        return min(0.9, max(0.5, base_threshold * risk_multiplier))

    def _calculate_risk_correlation(self, ethical_score: float, risk_score: float) -> float:
        """Calculate correlation between ethical and risk scores"""
        # Simple correlation measure
        diff = abs(ethical_score - (1 - risk_score))  # Risk score is inverted
        return max(0.0, 1.0 - diff)

    async def _verify_safety(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced safety verification with historical analysis"""
        try:
            action_desc = params.get("action_description", "")
            context = params.get("context", {})

            # Get historical safety patterns
            similar_evaluations = self.ethics_storage.find_similar_evaluations(
                {"action": action_desc}, threshold=0.6
            )

            # Calculate safety score with historical weighting
            base_safety_score = self._calculate_safety_score(action_desc, context)

            # Adjust based on historical patterns
            historical_adjustment = 0.0
            if similar_evaluations:
                historical_scores = [e.get("ethical_score", 0.5) for e in similar_evaluations[:3]]
                avg_historical = sum(historical_scores) / len(historical_scores)
                historical_adjustment = (avg_historical - 0.5) * 0.2

            safety_score = max(0.0, min(1.0, base_safety_score + historical_adjustment))

            # Determine safety requirements
            safeguards_required = self._determine_safeguards(safety_score, context)

            is_safe = safety_score >= 0.75

            result = {
                "status": "success",
                "action": action_desc,
                "is_safe": is_safe,
                "safety_score": safety_score,
                "safeguards_required": safeguards_required,
                "historical_context": len(similar_evaluations),
                "timestamp": datetime.utcnow().isoformat()
            }

            return result

        except Exception as e:
            self.logger.exception(f"Safety verification error: {e}")
            return {"status": "error", "error": str(e)}

    def _calculate_safety_score(self, action_desc: str, context: Dict[str, Any]) -> float:
        """Calculate safety score based on action analysis"""
        score = 0.8  # Start with moderate safety

        # Risk indicators decrease score
        risk_indicators = ["unsafe", "dangerous", "risky", "experimental", "unstable"]
        risk_count = sum(1 for indicator in risk_indicators if indicator in action_desc.lower())
        score -= risk_count * 0.1

        # Safety indicators increase score
        safety_indicators = ["tested", "verified", "safe", "stable", "monitored"]
        safety_count = sum(1 for indicator in safety_indicators if indicator in action_desc.lower())
        score += safety_count * 0.05

        # Context adjustments
        if context.get("environment") == "production":
            score -= 0.1
        if context.get("has_fallback", False):
            score += 0.1

        return max(0.0, min(1.0, score))

    def _determine_safeguards(self, safety_score: float, context: Dict[str, Any]) -> List[str]:
        """Determine required safeguards based on safety score"""
        safeguards = ["logging"]

        if safety_score < 0.9:
            safeguards.append("monitoring")
        if safety_score < 0.8:
            safeguards.append("rollback capability")
        if safety_score < 0.7:
            safeguards.extend(["manual approval", "gradual rollout"])
        if safety_score < 0.6:
            safeguards.extend(["comprehensive testing", "expert review"])

        # Context-specific safeguards
        if context.get("environment") == "production":
            safeguards.append("production safeguards")
        if context.get("stakeholders_count", 0) > 50:
            safeguards.append("stakeholder notification")

        return list(set(safeguards))  # Remove duplicates

    async def _assess_impact(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced impact assessment with simulation integration"""
        try:
            action_desc = params.get("action_description", "")
            context = params.get("context", {})

            # Basic impact assessment
            impact_assessment = {
                "action": action_desc,
                "positive_impacts": ["Increased efficiency", "Better user experience"],
                "negative_impacts": ["Resource consumption", "Learning curve"],
                "risk_level": "low",
                "mitigation_strategies": ["Gradual rollout", "User training"]
            }

            # Enhance with simulation if available
            if self.simulation_agent:
                simulation_result = await self.simulation_agent._verify_simulation({
                    "simulation_data": {
                        "action": action_desc,
                        "context": context,
                        "time_horizon": "short_term"
                    }
                })

                if simulation_result.get("status") == "success":
                    sim_data = simulation_result.get("verification", {})
                    impact_assessment["simulation_validated"] = sim_data.get("is_valid", False)
                    impact_assessment["simulation_accuracy"] = sim_data.get("accuracy_score", 0.5)

                    # Adjust risk level based on simulation
                    if sim_data.get("accuracy_score", 0.5) < 0.7:
                        impact_assessment["risk_level"] = "medium"

            # Store assessment in ethical memory
            assessment_record = {
                **impact_assessment,
                "context": context,
                "timestamp": datetime.utcnow().isoformat()
            }
            self.ethics_storage.store_assessment(assessment_record)

            return {"status": "success", "assessment": impact_assessment}

        except Exception as e:
            self.logger.exception(f"Impact assessment error: {e}")
            return {"status": "error", "error": str(e)}

    async def _set_ethical_framework(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Set the ethical framework for analysis"""
        try:
            framework = params.get("framework", "utilitarian")

            if framework not in self.framework_configs:
                return {
                    "status": "error",
                    "error": f"Unknown framework: {framework}. Available: {list(self.framework_configs.keys())}"
                }

            self.ethical_framework = framework
            self.logger.info(f"Ethical framework changed to: {framework}")

            return {
                "status": "success",
                "framework": framework,
                "weights": self.framework_configs[framework]["weights"]
            }

        except Exception as e:
            self.logger.exception(f"Framework setting error: {e}")
            return {"status": "error", "error": str(e)}

    async def _get_evaluation_history(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get evaluation history"""
        try:
            limit = params.get("limit", 10)
            history = self.ethics_storage.get_evaluation_history(limit)

            return {
                "status": "success",
                "history": history,
                "total_evaluations": len(self.ethics_storage.evaluations)
            }

        except Exception as e:
            self.logger.exception(f"History retrieval error: {e}")
            return {"status": "error", "error": str(e)}

    async def _verify_self_consistency(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Verify consistency across ethical evaluations"""
        try:
            tolerance = params.get("tolerance", 0.1)
            recent_evaluations = self.ethics_storage.get_evaluation_history(20)

            if len(recent_evaluations) < 2:
                return {
                    "status": "success",
                    "consistency_score": 1.0,
                    "message": "Insufficient data for consistency analysis"
                }

            # Analyze consistency in scoring
            scores = [e.get("ethical_score", 0.5) for e in recent_evaluations]
            avg_score = sum(scores) / len(scores)
            variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
            consistency_score = max(0.0, 1.0 - variance * 10)  # Scale variance to 0-1

            # Check for concerning patterns
            issues = []
            if variance > tolerance:
                issues.append("High variance in ethical scoring")
            if any(s < 0.3 for s in scores[-5:]):  # Recent very low scores
                issues.append("Recent evaluations show concerning patterns")

            return {
                "status": "success",
                "consistency_score": consistency_score,
                "average_score": avg_score,
                "variance": variance,
                "issues": issues,
                "recommendations": ["Review scoring methodology"] if issues else []
            }

        except Exception as e:
            self.logger.exception(f"Consistency verification error: {e}")
            return {"status": "error", "error": str(e)}

    def _update_context_weights(self, patterns: Dict[str, Any]):
        """Update context weights based on historical patterns"""
        # Learn from risk patterns to adjust scoring weights
        for pattern_name, pattern_data in patterns.items():
            if "severity_sum" in pattern_data and pattern_data["count"] > 0:
                avg_severity = pattern_data["severity_sum"] / pattern_data["count"]
                self.context_weights[pattern_name] = avg_severity

    async def shutdown(self) -> bool:
        """Shutdown the EthicsAgent gracefully"""
        try:
            self.logger.info(f"{self.name} shutting down")
            # Save any pending evaluations
            self.ethics_storage._save_data()
            return True
        except Exception as e:
            self.logger.exception(f"Failed to shutdown {self.name}: {e}")
            return False

    def get_ethics_summary(self) -> Dict[str, Any]:
        """Get comprehensive ethics summary"""
        return self.ethics_storage.get_ethics_summary()
    
    async def shutdown(self) -> bool:
        self.logger.info(f"{self.name} shutting down")
        self.update_status(AgentStatus.TERMINATED)
        return True


__all__ = ["EthicsAgent"]
