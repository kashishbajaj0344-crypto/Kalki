from typing import Dict, Any, List, Optional
import asyncio
import time
from datetime import datetime
from ..base_agent import BaseAgent, AgentCapability, AgentStatus
from .ethics_storage import EthicsStorage


class SimulationVerifierAgent(BaseAgent):
    """Enhanced simulation verification agent with parallel processing and meta-evaluation"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="SimulationVerifierAgent",
            capabilities=[AgentCapability.SAFETY_VERIFICATION],
            description="Verifies simulations with parallel processing and meta-evaluation",
            dependencies=["EthicsAgent"],
            config=config or {}
        )

        # Simulation types with verification criteria
        self.simulation_types = {
            "safety_critical": {
                "verification_depth": "comprehensive",
                "parallel_scenarios": 10,
                "accuracy_threshold": 0.95,
                "edge_case_coverage": 0.9
            },
            "ethical_dilemma": {
                "verification_depth": "deep",
                "parallel_scenarios": 8,
                "accuracy_threshold": 0.90,
                "edge_case_coverage": 0.8
            },
            "performance_test": {
                "verification_depth": "standard",
                "parallel_scenarios": 5,
                "accuracy_threshold": 0.85,
                "edge_case_coverage": 0.6
            },
            "exploratory": {
                "verification_depth": "basic",
                "parallel_scenarios": 3,
                "accuracy_threshold": 0.75,
                "edge_case_coverage": 0.4
            }
        }

        # Verification criteria
        self.verification_criteria = {
            "statistical_validity": {"weight": 0.25, "description": "Statistical soundness of simulation"},
            "edge_case_coverage": {"weight": 0.20, "description": "Coverage of edge cases and outliers"},
            "model_accuracy": {"weight": 0.20, "description": "Accuracy of underlying models"},
            "consequence_modeling": {"weight": 0.15, "description": "Proper modeling of consequences"},
            "uncertainty_quantification": {"weight": 0.10, "description": "Quantification of uncertainties"},
            "validation_against_real": {"weight": 0.10, "description": "Validation against real-world data"}
        }

        # Initialize simulation storage
        self.simulation_storage = EthicsStorage()

        # Cross-agent references
        self.ethics_agent = None
        self.risk_agent = None

        # Parallel processing configuration
        self.max_concurrent_simulations = self.config.get("max_concurrent", 5)
        self.simulation_timeout = self.config.get("timeout", 300)  # 5 minutes

        # Meta-evaluation parameters
        self.simulation_history = []
        self.verification_patterns = {}

    async def initialize(self) -> bool:
        try:
            self.logger.info(f"{self.name} initialized with {len(self.simulation_types)} simulation types")
            self.logger.info(f"Max concurrent simulations: {self.max_concurrent_simulations}")

            # Load historical simulation patterns
            patterns = self.simulation_storage.get_risk_patterns()
            self._update_verification_patterns(patterns)

            return True
        except Exception as e:
            self.logger.exception(f"Failed to initialize {self.name}: {e}")
            return False

    def set_ethics_agent(self, ethics_agent):
        """Set reference to EthicsAgent for consequence evaluation"""
        self.ethics_agent = ethics_agent
        self.logger.info("Connected to EthicsAgent for consequence evaluation")

    def set_risk_agent(self, risk_agent):
        """Set reference to RiskAssessmentAgent for risk modeling"""
        self.risk_agent = risk_agent
        self.logger.info("Connected to RiskAssessmentAgent for risk modeling")

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        action = task.get("action")
        params = task.get("params", {})

        try:
            if action == "verify_simulation":
                return await self._verify_simulation(params)
            elif action == "verify_experiment":
                return await self._verify_experiment(params)
            elif action == "run_parallel_simulations":
                return await self._run_parallel_simulations(params)
            elif action == "meta_evaluate":
                return await self._meta_evaluate_simulations(params)
            elif action == "get_simulation_history":
                return await self._get_simulation_history(params)
            else:
                return {"status": "error", "error": f"Unknown action: {action}"}
        except Exception as e:
            self.logger.exception(f"Simulation agent execution error: {e}")
            return {"status": "error", "error": str(e), "action": action}

    async def _verify_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced simulation verification with comprehensive analysis"""
        try:
            simulation_data = params.get("simulation_data", {})
            simulation_type = self._determine_simulation_type(simulation_data)

            # Run parallel verification checks
            verification_tasks = [
                self._check_statistical_validity(simulation_data),
                self._check_edge_case_coverage(simulation_data),
                self._check_model_accuracy(simulation_data),
                self._check_consequence_modeling(simulation_data),
                self._check_uncertainty_quantification(simulation_data),
                self._check_real_world_validation(simulation_data)
            ]

            # Execute verification checks concurrently
            verification_results = await asyncio.gather(*verification_tasks, return_exceptions=True)

            # Process results and handle exceptions
            processed_results = {}
            for i, result in enumerate(verification_results):
                criterion_name = list(self.verification_criteria.keys())[i]
                if isinstance(result, Exception):
                    processed_results[criterion_name] = {"score": 0.0, "error": str(result)}
                else:
                    processed_results[criterion_name] = result

            # Calculate overall verification score
            overall_score = self._calculate_overall_verification_score(processed_results)

            # Determine simulation validity
            type_config = self.simulation_types[simulation_type]
            is_valid = overall_score >= type_config["accuracy_threshold"]

            # Generate issues and warnings
            issues_found, warnings = self._analyze_verification_issues(processed_results, type_config)

            # Cross-validate with ethics and risk agents
            cross_validation = await self._perform_cross_validation(simulation_data)

            # Determine approval status
            approved_for_use = self._determine_approval_status(overall_score, issues_found, cross_validation)

            verification_result = {
                "simulation_type": simulation_type,
                "is_valid": is_valid,
                "accuracy_score": overall_score,
                "verification_details": processed_results,
                "issues_found": issues_found,
                "warnings": warnings,
                "approved_for_use": approved_for_use,
                "cross_validation": cross_validation,
                "recommendations": self._generate_verification_recommendations(issues_found, simulation_type),
                "timestamp": datetime.utcnow().isoformat()
            }

            # Store simulation result
            self.simulation_storage.store_simulation(verification_result)

            # Update simulation history
            self.simulation_history.append({
                "type": simulation_type,
                "score": overall_score,
                "valid": is_valid,
                "issues": len(issues_found)
            })

            # Emit event for real-time monitoring
            await self.emit_event("simulation.verification_completed", {
                "simulation_id": len(self.simulation_storage.simulations),
                "simulation_type": simulation_type,
                "accuracy_score": overall_score,
                "is_valid": is_valid,
                "issues_found": len(issues_found),
                "approved_for_use": approved_for_use
            })

            return {"status": "success", "verification": verification_result}

        except Exception as e:
            self.logger.exception(f"Simulation verification error: {e}")
            return {"status": "error", "error": str(e)}

    def _determine_simulation_type(self, simulation_data: Dict[str, Any]) -> str:
        """Determine the type of simulation based on its characteristics"""
        action = simulation_data.get("action", "").lower()
        context = simulation_data.get("context", {})

        # Classification logic
        if any(word in action for word in ["safety", "critical", "emergency", "failure"]):
            return "safety_critical"
        elif any(word in action for word in ["ethical", "moral", "dilemma", "choice"]):
            return "ethical_dilemma"
        elif any(word in action for word in ["performance", "benchmark", "test", "measure"]):
            return "performance_test"
        elif context.get("exploratory", False):
            return "exploratory"
        else:
            return "performance_test"  # Default

    async def _check_statistical_validity(self, simulation_data: Dict[str, Any]) -> Dict[str, float]:
        """Check statistical validity of simulation"""
        # Simulate statistical analysis
        await asyncio.sleep(0.1)  # Simulate processing time

        # Basic checks (would be more sophisticated in real implementation)
        has_sample_size = simulation_data.get("sample_size", 0) > 0
        has_confidence_intervals = simulation_data.get("confidence_intervals", False)
        has_statistical_tests = simulation_data.get("statistical_tests", False)

        score = 0.0
        if has_sample_size:
            score += 0.4
        if has_confidence_intervals:
            score += 0.3
        if has_statistical_tests:
            score += 0.3

        return {"score": min(1.0, score), "details": "Statistical validity assessment"}

    async def _check_edge_case_coverage(self, simulation_data: Dict[str, Any]) -> Dict[str, float]:
        """Check coverage of edge cases"""
        await asyncio.sleep(0.1)

        edge_cases = simulation_data.get("edge_cases", [])
        total_scenarios = simulation_data.get("total_scenarios", 1)

        if total_scenarios == 0:
            return {"score": 0.0, "details": "No scenarios defined"}

        coverage_ratio = len(edge_cases) / total_scenarios
        score = min(1.0, coverage_ratio * 2)  # Scale to ensure adequate coverage

        return {"score": score, "details": f"Edge case coverage: {coverage_ratio:.2%}"}

    async def _check_model_accuracy(self, simulation_data: Dict[str, Any]) -> Dict[str, float]:
        """Check accuracy of underlying models"""
        await asyncio.sleep(0.1)

        model_accuracy = simulation_data.get("model_accuracy", 0.8)
        validation_score = simulation_data.get("validation_score", 0.7)

        # Combine model accuracy with validation score
        combined_score = (model_accuracy + validation_score) / 2

        return {"score": combined_score, "details": f"Model accuracy: {model_accuracy:.2%}"}

    async def _check_consequence_modeling(self, simulation_data: Dict[str, Any]) -> Dict[str, float]:
        """Check proper modeling of consequences"""
        await asyncio.sleep(0.1)

        consequences = simulation_data.get("consequences", [])
        has_short_term = any(c.get("timeframe") == "short" for c in consequences)
        has_long_term = any(c.get("timeframe") == "long" for c in consequences)
        has_secondary = any(c.get("type") == "secondary" for c in consequences)

        score = 0.0
        if has_short_term:
            score += 0.4
        if has_long_term:
            score += 0.4
        if has_secondary:
            score += 0.2

        return {"score": score, "details": "Consequence modeling assessment"}

    async def _check_uncertainty_quantification(self, simulation_data: Dict[str, Any]) -> Dict[str, float]:
        """Check quantification of uncertainties"""
        await asyncio.sleep(0.1)

        has_uncertainty_bounds = simulation_data.get("uncertainty_bounds", False)
        has_sensitivity_analysis = simulation_data.get("sensitivity_analysis", False)
        uncertainty_range = simulation_data.get("uncertainty_range", 0.5)

        score = uncertainty_range  # Base score from uncertainty range
        if has_uncertainty_bounds:
            score += 0.2
        if has_sensitivity_analysis:
            score += 0.2

        return {"score": min(1.0, score), "details": "Uncertainty quantification"}

    async def _check_real_world_validation(self, simulation_data: Dict[str, Any]) -> Dict[str, float]:
        """Check validation against real-world data"""
        await asyncio.sleep(0.1)

        has_historical_data = simulation_data.get("historical_validation", False)
        has_expert_review = simulation_data.get("expert_validation", False)
        validation_coverage = simulation_data.get("validation_coverage", 0.0)

        score = validation_coverage
        if has_historical_data:
            score += 0.3
        if has_expert_review:
            score += 0.2

        return {"score": min(1.0, score), "details": "Real-world validation"}

    def _calculate_overall_verification_score(self, verification_results: Dict[str, Any]) -> float:
        """Calculate overall verification score from individual criteria"""
        total_weighted_score = 0.0
        total_weight = 0.0

        for criterion_name, result in verification_results.items():
            if criterion_name in self.verification_criteria:
                weight = self.verification_criteria[criterion_name]["weight"]
                score = result.get("score", 0.0)
                total_weighted_score += score * weight
                total_weight += weight

        return total_weighted_score / total_weight if total_weight > 0 else 0.0

    def _analyze_verification_issues(self, results: Dict[str, Any], type_config: Dict[str, Any]) -> tuple:
        """Analyze verification results for issues and warnings"""
        issues_found = []
        warnings = []

        threshold = type_config["accuracy_threshold"]

        for criterion_name, result in results.items():
            score = result.get("score", 0.0)

            if score < threshold * 0.7:  # Critical failure
                issues_found.append(f"Critical: {criterion_name} score too low ({score:.2f})")
            elif score < threshold:  # Warning level
                warnings.append(f"Warning: {criterion_name} below threshold ({score:.2f})")

        return issues_found, warnings

    async def _perform_cross_validation(self, simulation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform cross-validation with ethics and risk agents"""
        cross_validation = {}

        if self.ethics_agent:
            try:
                ethics_result = await self.ethics_agent._review_ethics({
                    "action_description": simulation_data.get("action", ""),
                    "context": simulation_data.get("context", {})
                })
                if ethics_result.get("status") == "success":
                    cross_validation["ethics"] = {
                        "score": ethics_result.get("ethical_score", 0.5),
                        "concerns": len(ethics_result.get("violations", []))
                    }
            except Exception as e:
                cross_validation["ethics"] = {"error": str(e)}

        if self.risk_agent:
            try:
                risk_result = await self.risk_agent._assess_risk({
                    "scenario": simulation_data.get("action", ""),
                    "factors": simulation_data.get("risk_factors", [])
                })
                if risk_result.get("status") == "success":
                    cross_validation["risk"] = {
                        "level": risk_result.get("risk_level"),
                        "score": risk_result.get("risk_score", 0.5)
                    }
            except Exception as e:
                cross_validation["risk"] = {"error": str(e)}

        return cross_validation

    def _determine_approval_status(self, score: float, issues: List[str], cross_validation: Dict[str, Any]) -> bool:
        """Determine if simulation is approved for use"""
        # Must pass basic score threshold
        if score < 0.7:
            return False

        # Must not have critical issues
        if any("Critical:" in issue for issue in issues):
            return False

        # Check cross-validation results
        ethics_score = cross_validation.get("ethics", {}).get("score", 0.5)
        risk_level = cross_validation.get("risk", {}).get("level", "medium")

        if ethics_score < 0.6 or risk_level in ["high", "critical"]:
            return False

        return True

    def _generate_verification_recommendations(self, issues: List[str], simulation_type: str) -> List[str]:
        """Generate recommendations based on verification issues"""
        recommendations = []

        if any("statistical" in issue.lower() for issue in issues):
            recommendations.append("Improve statistical methodology and sample sizes")

        if any("edge" in issue.lower() for issue in issues):
            recommendations.append("Increase edge case coverage in simulation design")

        if any("model" in issue.lower() for issue in issues):
            recommendations.append("Validate and calibrate underlying models")

        if any("consequence" in issue.lower() for issue in issues):
            recommendations.append("Enhance consequence modeling, especially long-term effects")

        if any("uncertainty" in issue.lower() for issue in issues):
            recommendations.append("Implement proper uncertainty quantification")

        if any("validation" in issue.lower() for issue in issues):
            recommendations.append("Add real-world validation and expert review")

        # Type-specific recommendations
        if simulation_type == "safety_critical":
            recommendations.append("Implement additional safety verification layers")
        elif simulation_type == "ethical_dilemma":
            recommendations.append("Include ethical review board consultation")

        return recommendations[:5]  # Limit to top 5

    async def _verify_experiment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced experiment verification with comprehensive safety checks"""
        try:
            experiment_desc = params.get("experiment_description", "")
            experiment_config = params.get("config", {})

            # Run comprehensive safety checks
            safety_checks = await self._run_experiment_safety_checks(experiment_config)

            # Assess containment and rollback
            containment_assessment = self._assess_experiment_containment(experiment_config)
            rollback_assessment = self._assess_rollback_capability(experiment_config)

            # Determine overall safety
            is_safe = (safety_checks["overall_score"] >= 0.85 and
                      containment_assessment["adequate"] and
                      rollback_assessment["available"])

            # Generate conditions and monitoring requirements
            conditions = self._generate_experiment_conditions(safety_checks, containment_assessment)
            monitoring_plan = self._create_experiment_monitoring_plan(experiment_config)

            verification_result = {
                "is_safe": is_safe,
                "safety_score": safety_checks["overall_score"],
                "containment_adequate": containment_assessment["adequate"],
                "rollback_available": rollback_assessment["available"],
                "monitoring_enabled": True,
                "approval_status": "approved" if is_safe else "rejected",
                "conditions": conditions,
                "monitoring_plan": monitoring_plan,
                "safety_checks": safety_checks,
                "recommendations": self._generate_experiment_recommendations(safety_checks),
                "timestamp": datetime.utcnow().isoformat()
            }

            return {"status": "success", "experiment": experiment_desc, "verification": verification_result}

        except Exception as e:
            self.logger.exception(f"Experiment verification error: {e}")
            return {"status": "error", "error": str(e)}

    async def _run_experiment_safety_checks(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive safety checks for experiment"""
        checks = {
            "isolation_level": self._check_isolation_level(config),
            "resource_limits": self._check_resource_limits(config),
            "data_safety": self._check_data_safety(config),
            "failure_modes": self._check_failure_modes(config),
            "monitoring_coverage": self._check_monitoring_coverage(config)
        }

        overall_score = sum(check["score"] for check in checks.values()) / len(checks)

        return {
            "overall_score": overall_score,
            "checks": checks,
            "passed": sum(1 for check in checks.values() if check["score"] >= 0.8),
            "total": len(checks)
        }

    def _check_isolation_level(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Check experiment isolation level"""
        isolation = config.get("isolation", "none")
        score_map = {"full": 1.0, "sandbox": 0.9, "container": 0.8, "partial": 0.6, "none": 0.1}
        return {"score": score_map.get(isolation, 0.1), "level": isolation}

    def _check_resource_limits(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Check resource limits and constraints"""
        has_cpu_limit = config.get("cpu_limit", False)
        has_memory_limit = config.get("memory_limit", False)
        has_timeout = config.get("timeout", False)

        score = 0.0
        if has_cpu_limit:
            score += 0.4
        if has_memory_limit:
            score += 0.4
        if has_timeout:
            score += 0.2

        return {"score": score, "limits": [has_cpu_limit, has_memory_limit, has_timeout]}

    def _check_data_safety(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Check data safety measures"""
        has_backup = config.get("data_backup", False)
        has_encryption = config.get("data_encryption", False)
        has_access_control = config.get("access_control", False)

        score = 0.0
        if has_backup:
            score += 0.4
        if has_encryption:
            score += 0.3
        if has_access_control:
            score += 0.3

        return {"score": score, "measures": [has_backup, has_encryption, has_access_control]}

    def _check_failure_modes(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Check failure mode analysis"""
        failure_modes = config.get("failure_modes", [])
        has_contingency = config.get("contingency_plan", False)

        score = min(1.0, len(failure_modes) / 5)  # At least 5 failure modes identified
        if has_contingency:
            score += 0.2

        return {"score": score, "modes_identified": len(failure_modes), "contingency": has_contingency}

    def _check_monitoring_coverage(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Check monitoring coverage"""
        monitoring_metrics = config.get("monitoring_metrics", [])
        has_alerts = config.get("alerts_enabled", False)

        score = min(1.0, len(monitoring_metrics) / 10)  # At least 10 metrics
        if has_alerts:
            score += 0.2

        return {"score": score, "metrics": len(monitoring_metrics), "alerts": has_alerts}

    def _assess_experiment_containment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Assess experiment containment adequacy"""
        containment_measures = config.get("containment", [])
        required_measures = ["isolation", "resource_limits", "monitoring", "kill_switch"]

        adequate = all(measure in containment_measures for measure in required_measures)
        coverage = len(set(containment_measures).intersection(required_measures)) / len(required_measures)

        return {
            "adequate": adequate,
            "coverage": coverage,
            "measures": containment_measures,
            "missing": [m for m in required_measures if m not in containment_measures]
        }

    def _assess_rollback_capability(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Assess rollback capability"""
        has_rollback = config.get("rollback_enabled", False)
        rollback_tests = config.get("rollback_tested", False)
        backup_available = config.get("backup_available", False)

        available = has_rollback and rollback_tests and backup_available

        return {
            "available": available,
            "tested": rollback_tests,
            "backup": backup_available,
            "automated": config.get("rollback_automated", False)
        }

    def _generate_experiment_conditions(self, safety_checks: Dict[str, Any],
                                      containment: Dict[str, Any]) -> List[str]:
        """Generate conditions for experiment execution"""
        conditions = ["Must run in controlled environment"]

        if safety_checks["overall_score"] < 0.9:
            conditions.append("Requires additional safety monitoring")

        if not containment["adequate"]:
            conditions.extend([
                f"Implement missing containment: {', '.join(containment['missing'])}",
                "Obtain additional safety approvals"
            ])

        conditions.extend([
            "Maintain continuous monitoring throughout execution",
            "Have rollback procedures ready",
            "Document all observations and anomalies"
        ])

        return conditions

    def _create_experiment_monitoring_plan(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create monitoring plan for experiment"""
        return {
            "metrics": config.get("monitoring_metrics", ["cpu", "memory", "errors"]),
            "frequency": "real-time",
            "alerts": ["resource_threshold", "error_detection", "anomaly_detection"],
            "escalation": ["immediate_notification", "automatic_shutdown_if_critical"],
            "logging": "comprehensive"
        }

    def _generate_experiment_recommendations(self, safety_checks: Dict[str, Any]) -> List[str]:
        """Generate recommendations for experiment safety improvements"""
        recommendations = []

        for check_name, check_result in safety_checks["checks"].items():
            if check_result["score"] < 0.8:
                if check_name == "isolation_level":
                    recommendations.append("Improve experiment isolation (use full sandboxing)")
                elif check_name == "resource_limits":
                    recommendations.append("Implement comprehensive resource limits")
                elif check_name == "data_safety":
                    recommendations.append("Enhance data safety measures (backup, encryption, access control)")
                elif check_name == "failure_modes":
                    recommendations.append("Complete failure mode analysis and contingency planning")
                elif check_name == "monitoring_coverage":
                    recommendations.append("Expand monitoring coverage and enable alerts")

        return recommendations

    async def _run_parallel_simulations(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run multiple simulations in parallel"""
        try:
            simulation_configs = params.get("simulations", [])
            max_concurrent = min(len(simulation_configs), self.max_concurrent_simulations)

            # Create verification tasks
            verification_tasks = []
            for config in simulation_configs:
                task = self._verify_simulation({"simulation_data": config})
                verification_tasks.append(task)

            # Run simulations with concurrency control
            semaphore = asyncio.Semaphore(max_concurrent)
            async def limited_task(task):
                async with semaphore:
                    return await task

            # Execute with timeout
            limited_tasks = [limited_task(task) for task in verification_tasks]
            results = await asyncio.gather(*limited_tasks, return_exceptions=True)

            # Process results
            successful = 0
            failed = 0
            processed_results = []

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        "simulation_id": i,
                        "status": "error",
                        "error": str(result)
                    })
                    failed += 1
                else:
                    processed_results.append({
                        "simulation_id": i,
                        "status": "success",
                        "result": result
                    })
                    successful += 1

            return {
                "status": "success",
                "total_simulations": len(simulation_configs),
                "successful": successful,
                "failed": failed,
                "results": processed_results,
                "execution_time": time.time()
            }

        except Exception as e:
            self.logger.exception(f"Parallel simulation error: {e}")
            return {"status": "error", "error": str(e)}

    async def _meta_evaluate_simulations(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform meta-evaluation of simulation verification consistency"""
        try:
            recent_simulations = self.simulation_storage.get_simulation_history(20)

            if len(recent_simulations) < 3:
                return {
                    "status": "success",
                    "consistency_score": 1.0,
                    "message": "Insufficient simulation history for meta-evaluation"
                }

            # Analyze consistency across verifications
            scores = [s.get("accuracy_score", 0.5) for s in recent_simulations]
            avg_score = sum(scores) / len(scores)
            variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
            consistency_score = max(0.0, 1.0 - variance * 5)  # Scale variance

            # Check for systematic biases
            biases = self._detect_systematic_biases(recent_simulations)

            # Generate meta-evaluation insights
            insights = self._generate_meta_insights(recent_simulations, consistency_score)

            return {
                "status": "success",
                "consistency_score": consistency_score,
                "average_score": avg_score,
                "variance": variance,
                "biases_detected": biases,
                "insights": insights,
                "recommendations": self._generate_meta_recommendations(biases, consistency_score)
            }

        except Exception as e:
            self.logger.exception(f"Meta-evaluation error: {e}")
            return {"status": "error", "error": str(e)}

    def _detect_systematic_biases(self, simulations: List[Dict[str, Any]]) -> List[str]:
        """Detect systematic biases in simulation verification"""
        biases = []

        # Check for simulation type bias
        type_scores = {}
        for sim in simulations:
            sim_type = sim.get("simulation_type", "unknown")
            score = sim.get("accuracy_score", 0.5)
            if sim_type not in type_scores:
                type_scores[sim_type] = []
            type_scores[sim_type].append(score)

        for sim_type, scores in type_scores.items():
            if len(scores) >= 3:
                avg_score = sum(scores) / len(scores)
                if sim_type == "safety_critical" and avg_score < 0.9:
                    biases.append(f"Consistently low scores for safety-critical simulations")
                elif sim_type == "exploratory" and avg_score > 0.8:
                    biases.append(f"Consistently high scores for exploratory simulations")

        return biases

    def _generate_meta_insights(self, simulations: List[Dict[str, Any]], consistency_score: float) -> List[str]:
        """Generate insights from meta-evaluation"""
        insights = []

        if consistency_score > 0.8:
            insights.append("High consistency in simulation verification")
        elif consistency_score < 0.6:
            insights.append("Significant inconsistency in verification scores")

        # Analyze trends
        recent_scores = [s.get("accuracy_score", 0.5) for s in simulations[-5:]]
        older_scores = [s.get("accuracy_score", 0.5) for s in simulations[:-5]]

        if recent_scores and older_scores:
            recent_avg = sum(recent_scores) / len(recent_scores)
            older_avg = sum(older_scores) / len(older_scores)

            if recent_avg > older_avg + 0.1:
                insights.append("Verification scores improving over time")
            elif recent_avg < older_avg - 0.1:
                insights.append("Verification scores declining over time")

        return insights

    def _generate_meta_recommendations(self, biases: List[str], consistency_score: float) -> List[str]:
        """Generate recommendations based on meta-evaluation"""
        recommendations = []

        if consistency_score < 0.7:
            recommendations.append("Review verification criteria for consistency")
            recommendations.append("Implement inter-rater reliability checks")

        if biases:
            recommendations.append("Address detected systematic biases in scoring")
            recommendations.append("Review verification guidelines for different simulation types")

        if consistency_score > 0.9:
            recommendations.append("Current verification process is highly consistent")

        return recommendations

    async def _get_simulation_history(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get simulation verification history"""
        try:
            limit = params.get("limit", 10)
            history = self.simulation_storage.get_simulation_history(limit)

            return {
                "status": "success",
                "history": history,
                "total_simulations": len(self.simulation_storage.simulations)
            }

        except Exception as e:
            self.logger.exception(f"Simulation history retrieval error: {e}")
            return {"status": "error", "error": str(e)}

    def _update_verification_patterns(self, patterns: Dict[str, Any]):
        """Update verification patterns based on historical data"""
        for pattern_name, pattern_data in patterns.items():
            if "count" in pattern_data and pattern_data["count"] > 0:
                # Learn from historical verification patterns
                success_rate = pattern_data.get("success_rate", 0.5)
                self.verification_patterns[pattern_name] = {
                    "expected_accuracy": success_rate,
                    "historical_weight": min(1.0, pattern_data["count"] / 20)
                }

    async def shutdown(self) -> bool:
        """Shutdown the SimulationVerifierAgent gracefully"""
        try:
            self.logger.info(f"{self.name} shutting down")
            # Save any pending simulations
            self.simulation_storage._save_data()
            return True
        except Exception as e:
            self.logger.exception(f"Failed to shutdown {self.name}: {e}")
            return False

    def get_simulation_summary(self) -> Dict[str, Any]:
        """Get comprehensive simulation summary"""
        simulations = self.simulation_storage.get_simulation_history(100)
        return {
            "total_simulations": len(simulations),
            "average_accuracy": sum(s.get("accuracy_score", 0) for s in simulations) / max(len(simulations), 1),
            "types_breakdown": {},
            "recent_performance": self._calculate_recent_performance(simulations)
        }

    def _calculate_recent_performance(self, simulations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate recent simulation performance"""
        if len(simulations) < 5:
            return {"insufficient_data": True}

        recent = simulations[-5:]
        return {
            "average_score": sum(s.get("accuracy_score", 0) for s in recent) / len(recent),
            "consistency": self._calculate_score_consistency([s.get("accuracy_score", 0) for s in recent]),
            "trend": "improving" if recent[-1].get("accuracy_score", 0) > recent[0].get("accuracy_score", 0) else "stable"
        }

    def _calculate_score_consistency(self, scores: List[float]) -> float:
        """Calculate consistency of scores"""
        if len(scores) < 2:
            return 1.0

        avg_score = sum(scores) / len(scores)
        variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
        return max(0.0, 1.0 - variance * 10)


__all__ = ["SimulationVerifierAgent"]
