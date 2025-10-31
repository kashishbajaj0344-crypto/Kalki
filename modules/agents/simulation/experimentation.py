"""
Experimentation Agent (Phase 9)
==============================

Implements experimental design, hypothesis testing, and controlled experimentation.
Uses statistical experimental design, A/B testing, and sandbox environments.
"""

import asyncio
import time
import random
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
from collections import defaultdict
from scipy import stats
import math
import hashlib

from modules.logging_config import get_logger
from ..base_agent import BaseAgent, AgentCapability, AgentStatus
from ..knowledge.rollback_manager import RollbackManager

logger = get_logger("Kalki.Experimentation")


@dataclass
class Experiment:
    """Experimental design definition"""
    experiment_id: str
    name: str
    description: str
    hypothesis: str
    variables: Dict[str, Any]
    control_group: Dict[str, Any]
    treatment_groups: List[Dict[str, Any]]
    sample_size: int
    duration: int
    metrics: List[str]
    statistical_tests: List[str]


@dataclass
class ExperimentResult:
    """Experiment execution result"""
    experiment_id: str
    run_id: str
    start_time: float
    end_time: float
    control_results: Dict[str, List[float]]
    treatment_results: Dict[str, List[Dict[str, float]]]
    statistical_analysis: Dict[str, Any]
    conclusion: str
    confidence_level: float


@dataclass
class SandboxEnvironment:
    """Isolated experimentation environment"""
    sandbox_id: str
    name: str
    description: str
    resources: Dict[str, Any]
    isolation_level: str  # "full", "partial", "minimal"
    time_limit: int
    memory_limit: int
    network_access: bool
    file_system_access: bool


class ExperimentationAgent(BaseAgent):
    """
    Advanced experimentation agent for hypothesis testing and controlled experiments.
    Implements statistical experimental design, A/B testing, and sandbox environments.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="ExperimentationAgent",
            capabilities=[
                AgentCapability.EXPERIMENTATION,
                AgentCapability.ANALYTICS,
                AgentCapability.SIMULATION
            ],
            description="Statistical experimental design and controlled experimentation",
            config=config or {}
        )

        # Experiment parameters
        self.max_experiment_time = self.config.get('max_experiment_time', 3600)
        self.default_sample_size = self.config.get('default_sample_size', 100)
        self.default_confidence_level = self.config.get('default_confidence_level', 0.95)
        self.max_parallel_experiments = self.config.get('max_parallel_experiments', 3)

        # Experiment state
        self.active_experiments = {}
        self.experiment_history = defaultdict(list)
        self.experiments = {}
        self.sandboxes = {}

        # Statistical tools
        self.statistical_tests = {
            "t_test": self._perform_t_test,
            "anova": self._perform_anova,
            "chi_square": self._perform_chi_square,
            "mann_whitney": self._perform_mann_whitney,
            "correlation": self._perform_correlation_analysis
        }

        # Performance tracking
        self.experiment_stats = {
            "total_experiments": 0,
            "successful_experiments": 0,
            "failed_experiments": 0,
            "average_duration": 0,
            "total_computation_time": 0
        }

        # Persistence and checkpointing
        self.rollback_manager = RollbackManager(self.config.get('rollback_config', {}))
        self.checkpoint_interval = self.config.get('checkpoint_interval', 300)  # 5 minutes

        # Safety and compliance (simplified for experimentation)
        self.safety_enabled = self.config.get('safety_enabled', True)
        self.prohibited_keywords = [
            "virus", "pathogen", "toxin", "bioweapon", "pandemic",
            "explosive", "toxic", "carcinogen", "radioactive"
        ]

    async def initialize(self) -> bool:
        """Initialize experimentation environment"""
        try:
            logger.info("ExperimentationAgent initializing statistical experimentation framework")

            # Initialize rollback manager
            rollback_initialized = await self.rollback_manager.initialize()
            if not rollback_initialized:
                logger.warning("Failed to initialize rollback manager, proceeding without persistence")

            # Initialize default experiments
            await self._initialize_default_experiments()

            # Initialize sandbox environments
            await self._initialize_sandboxes()

            # Warm up statistical engines
            await self._warm_up_statistical_engines()

            logger.info("ExperimentationAgent initialized with A/B testing and hypothesis validation")
            return True

        except Exception as e:
            logger.exception(f"Failed to initialize ExperimentationAgent: {e}")
            return False

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute experimentation operations"""
        action = task.get("action")
        params = task.get("params", {})

        if action == "design_experiment":
            return await self._design_experiment(params)
        elif action == "run_experiment":
            return await self._run_experiment(params)
        elif action == "analyze_results":
            return await self._analyze_experiment_results(params)
        elif action == "create_sandbox":
            return await self._create_sandbox_environment(params)
        elif action == "run_sandbox_experiment":
            return await self._run_sandbox_experiment(params)
        elif action == "validate_hypothesis":
            return await self._validate_hypothesis(params)
        else:
            return {"status": "error", "error": f"Unknown action: {action}"}

    async def _design_experiment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Design a statistical experiment"""
        try:
            experiment_type = params.get("type", "ab_test")
            hypothesis = params.get("hypothesis", "")
            variables = params.get("variables", {})
            sample_size = params.get("sample_size", self.default_sample_size)

            if experiment_type == "ab_test":
                design = await self._design_ab_test(hypothesis, variables, sample_size)
            elif experiment_type == "factorial":
                design = await self._design_factorial_experiment(hypothesis, variables, sample_size)
            elif experiment_type == "multivariate":
                design = await self._design_multivariate_experiment(hypothesis, variables, sample_size)
            else:
                return {"status": "error", "error": f"Unknown experiment type: {experiment_type}"}

            # Create experiment
            experiment_id = f"exp_{len(self.experiments)}_{int(time.time())}"
            experiment = Experiment(
                experiment_id=experiment_id,
                name=params.get("name", f"Experiment {experiment_id}"),
                description=params.get("description", ""),
                hypothesis=hypothesis,
                variables=variables,
                control_group=design["control_group"],
                treatment_groups=design["treatment_groups"],
                sample_size=sample_size,
                duration=params.get("duration", 60),
                metrics=params.get("metrics", ["outcome"]),
                statistical_tests=params.get("statistical_tests", ["t_test"])
            )

            self.experiments[experiment_id] = experiment

            return {
                "status": "success",
                "experiment_id": experiment_id,
                "design": design,
                "power_analysis": await self._calculate_statistical_power(experiment)
            }

        except Exception as e:
            logger.exception(f"Experiment design error: {e}")
            return {"status": "error", "error": str(e)}

    async def _run_experiment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a designed experiment"""
        try:
            experiment_id = params.get("experiment_id")
            run_id = params.get("run_id", f"run_{int(time.time())}")

            if experiment_id not in self.experiments:
                return {"status": "error", "error": f"Experiment {experiment_id} not found"}

            experiment = self.experiments[experiment_id]

            # Safety check for experimentation
            if self.safety_enabled:
                experiment_text = f"{experiment.name} {experiment.description} {experiment.hypothesis}".lower()
                safety_violations = []
                
                for keyword in self.prohibited_keywords:
                    if keyword.lower() in experiment_text:
                        safety_violations.append(f"Prohibited keyword '{keyword}' detected")
                
                if safety_violations:
                    await self.emit_event("experiment.safety_blocked", {
                        "experiment_id": experiment_id,
                        "run_id": run_id,
                        "violations": safety_violations
                    })
                    return {
                        "status": "blocked",
                        "error": "Experiment contains prohibited content",
                        "violations": safety_violations
                    }

            # Check if experiment is already running
            if experiment_id in self.active_experiments:
                return {"status": "error", "error": f"Experiment {experiment_id} is already running"}

            # Create pre-experiment checkpoint
            experiment_state = {
                "experiment_id": experiment_id,
                "run_id": run_id,
                "experiment_name": experiment.name,
                "hypothesis": experiment.hypothesis,
                "sample_size": experiment.sample_size,
                "start_time": time.time(),
                "active_experiments": dict(self.active_experiments),
                "experiment_stats": dict(self.experiment_stats)
            }
            
            checkpoint_id = await self.rollback_manager.create_checkpoint(
                name=f"pre_experiment_{experiment_id}",
                state=experiment_state,
                tags=["experiment", "checkpoint"],
                metadata={"experiment_name": experiment.name, "hypothesis": experiment.hypothesis},
                source_agent=self.name,
                description=f"Pre-experiment checkpoint for {experiment_id}"
            )

            # Emit experiment start event
            await self.emit_event("experiment.started", {
                "experiment_id": experiment_id,
                "run_id": run_id,
                "experiment_name": experiment.name,
                "hypothesis": experiment.hypothesis,
                "sample_size": experiment.sample_size,
                "num_treatments": len(experiment.treatment_groups),
                "metrics": experiment.metrics,
                "checkpoint_id": checkpoint_id
            })

            # Start experiment
            self.active_experiments[experiment_id] = run_id

            try:
                # Execute experiment
                result = await self._execute_experiment(experiment, run_id)

                # Store result
                self.experiment_history[experiment_id].append(result)
                self._update_experiment_stats(True, result.end_time - result.start_time)

                # Create completion checkpoint
                completion_state = {
                    "experiment_id": experiment_id,
                    "run_id": run_id,
                    "completed_at": time.time(),
                    "experiment_result": {
                        "start_time": result.start_time,
                        "end_time": result.end_time,
                        "conclusion": result.conclusion,
                        "confidence_level": result.confidence_level,
                        "statistical_analysis": result.statistical_analysis
                    },
                    "experiment_stats": dict(self.experiment_stats)
                }
                
                completion_checkpoint_id = await self.rollback_manager.create_checkpoint(
                    name=f"completed_experiment_{experiment_id}",
                    state=completion_state,
                    tags=["experiment", "completed"],
                    metadata={"experiment_name": experiment.name, "conclusion": result.conclusion, "success": True},
                    source_agent=self.name,
                    description=f"Completion checkpoint for successful experiment {experiment_id}"
                )

                # Emit experiment completion event
                await self.emit_event("experiment.completed", {
                    "experiment_id": experiment_id,
                    "run_id": run_id,
                    "duration": result.end_time - result.start_time,
                    "conclusion": result.conclusion,
                    "confidence_level": result.confidence_level,
                    "statistical_significance": result.statistical_analysis.get("significant", False),
                    "checkpoint_id": completion_checkpoint_id
                })

                return {
                    "status": "success",
                    "experiment_id": experiment_id,
                    "run_id": run_id,
                    "result": result
                }

            finally:
                # Clean up
                if experiment_id in self.active_experiments:
                    del self.active_experiments[experiment_id]

        except Exception as e:
            logger.exception(f"Experiment execution error: {e}")
            self._update_experiment_stats(False, 0)
            
            # Create error checkpoint for recovery
            error_state = {
                "experiment_id": params.get("experiment_id", "unknown"),
                "run_id": params.get("run_id", "unknown"),
                "error_time": time.time(),
                "error": str(e),
                "error_type": type(e).__name__,
                "active_experiments": dict(self.active_experiments),
                "experiment_stats": dict(self.experiment_stats)
            }
            
            await self.rollback_manager.create_checkpoint(
                name=f"error_experiment_{params.get('experiment_id', 'unknown')}",
                state=error_state,
                tags=["experiment", "error"],
                metadata={"error_type": type(e).__name__, "experiment_id": params.get("experiment_id", "unknown")},
                source_agent=self.name,
                description=f"Error checkpoint for failed experiment"
            )
            
            # Emit experiment error event
            await self.emit_event("experiment.error", {
                "experiment_id": params.get("experiment_id", "unknown"),
                "run_id": params.get("run_id", "unknown"),
                "error": str(e),
                "error_type": type(e).__name__
            })
            return {"status": "error", "error": str(e)}

    async def _execute_experiment(self, experiment: Experiment, run_id: str) -> ExperimentResult:
        """Execute the experiment"""
        start_time = time.time()

        # Generate control group data
        control_results = {}
        for metric in experiment.metrics:
            control_results[metric] = await self._generate_sample_data(
                experiment.control_group, metric, experiment.sample_size
            )

        # Generate treatment group data
        treatment_results = {}
        for i, treatment in enumerate(experiment.treatment_groups):
            treatment_results[f"treatment_{i}"] = {}
            for metric in experiment.metrics:
                treatment_results[f"treatment_{i}"][metric] = await self._generate_sample_data(
                    treatment, metric, experiment.sample_size
                )

        end_time = time.time()

        # Perform statistical analysis
        statistical_analysis = await self._perform_statistical_analysis(
            control_results, treatment_results, experiment.statistical_tests
        )

        # Draw conclusion
        conclusion = self._draw_experiment_conclusion(statistical_analysis, experiment.hypothesis)

        return ExperimentResult(
            experiment_id=experiment.experiment_id,
            run_id=run_id,
            start_time=start_time,
            end_time=end_time,
            control_results=control_results,
            treatment_results=treatment_results,
            statistical_analysis=statistical_analysis,
            conclusion=conclusion,
            confidence_level=self.default_confidence_level
        )

    async def _generate_sample_data(self, group_config: Dict[str, Any], metric: str,
                                  sample_size: int) -> List[float]:
        """Generate sample data for a group"""
        # Extract distribution parameters
        distribution = group_config.get(f"{metric}_distribution", "normal")
        params = group_config.get(f"{metric}_params", {})

        data = []
        for _ in range(sample_size):
            if distribution == "normal":
                value = np.random.normal(
                    params.get("mean", 0),
                    params.get("std", 1)
                )
            elif distribution == "uniform":
                value = np.random.uniform(
                    params.get("min", 0),
                    params.get("max", 1)
                )
            elif distribution == "exponential":
                value = np.random.exponential(params.get("rate", 1))
            elif distribution == "beta":
                value = np.random.beta(
                    params.get("alpha", 2),
                    params.get("beta", 2)
                )
            else:
                value = np.random.normal(0, 1)  # Default

            data.append(value)

        return data

    async def _perform_statistical_analysis(self, control_results: Dict[str, List[float]],
                                         treatment_results: Dict[str, Dict[str, List[float]]],
                                         tests: List[str]) -> Dict[str, Any]:
        """Perform statistical analysis on experiment results"""
        analysis = {}

        for test_name in tests:
            if test_name in self.statistical_tests:
                test_func = self.statistical_tests[test_name]
                try:
                    analysis[test_name] = await test_func(control_results, treatment_results)
                except Exception as e:
                    logger.exception(f"Statistical test {test_name} failed: {e}")
                    analysis[test_name] = {"error": str(e)}

        return analysis

    async def _perform_t_test(self, control_results: Dict[str, List[float]],
                            treatment_results: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
        """Perform t-test analysis"""
        results = {}

        for metric in control_results.keys():
            control_data = control_results[metric]
            results[metric] = {}

            for treatment_name, treatment_data in treatment_results.items():
                if metric in treatment_data:
                    treatment_values = treatment_data[metric]

                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(control_data, treatment_values)

                    results[metric][treatment_name] = {
                        "t_statistic": t_stat,
                        "p_value": p_value,
                        "significant": p_value < (1 - self.default_confidence_level),
                        "effect_size": abs(np.mean(treatment_values) - np.mean(control_data)) / np.std(control_data)
                    }

        return results

    async def _perform_anova(self, control_results: Dict[str, List[float]],
                           treatment_results: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
        """Perform ANOVA analysis"""
        results = {}

        for metric in control_results.keys():
            all_data = [control_results[metric]]
            group_labels = ["control"]

            for treatment_name, treatment_data in treatment_results.items():
                if metric in treatment_data:
                    all_data.append(treatment_data[metric])
                    group_labels.append(treatment_name)

            try:
                f_stat, p_value = stats.f_oneway(*all_data)

                results[metric] = {
                    "f_statistic": f_stat,
                    "p_value": p_value,
                    "significant": p_value < (1 - self.default_confidence_level),
                    "groups_tested": group_labels
                }
            except Exception as e:
                results[metric] = {"error": str(e)}

        return results

    async def _perform_chi_square(self, control_results: Dict[str, List[float]],
                                treatment_results: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
        """Perform chi-square test for categorical data"""
        # Simplified implementation - assumes binned data
        results = {}

        for metric in control_results.keys():
            # Create contingency table from binned data
            try:
                control_binned = np.histogram(control_results[metric], bins=5)[0]

                for treatment_name, treatment_data in treatment_results.items():
                    if metric in treatment_data:
                        treatment_binned = np.histogram(treatment_data[metric], bins=5)[0]

                        # Create contingency table
                        contingency_table = np.array([control_binned, treatment_binned])

                        chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)

                        results[f"{metric}_{treatment_name}"] = {
                            "chi2_statistic": chi2_stat,
                            "p_value": p_value,
                            "degrees_of_freedom": dof,
                            "significant": p_value < (1 - self.default_confidence_level)
                        }
            except Exception as e:
                results[metric] = {"error": str(e)}

        return results

    async def _perform_mann_whitney(self, control_results: Dict[str, List[float]],
                                  treatment_results: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
        """Perform Mann-Whitney U test (non-parametric)"""
        results = {}

        for metric in control_results.keys():
            control_data = control_results[metric]
            results[metric] = {}

            for treatment_name, treatment_data in treatment_results.items():
                if metric in treatment_data:
                    treatment_values = treatment_data[metric]

                    # Perform Mann-Whitney U test
                    u_stat, p_value = stats.mannwhitneyu(control_data, treatment_values, alternative='two-sided')

                    results[metric][treatment_name] = {
                        "u_statistic": u_stat,
                        "p_value": p_value,
                        "significant": p_value < (1 - self.default_confidence_level)
                    }

        return results

    async def _perform_correlation_analysis(self, control_results: Dict[str, List[float]],
                                          treatment_results: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
        """Perform correlation analysis"""
        results = {}

        # Analyze correlations within control group
        if len(control_results) > 1:
            control_corr = np.corrcoef(list(control_results.values()))
            results["control_correlations"] = control_corr.tolist()

        # Analyze correlations within treatment groups
        for treatment_name, treatment_data in treatment_results.items():
            if len(treatment_data) > 1:
                treatment_corr = np.corrcoef(list(treatment_data.values()))
                results[f"{treatment_name}_correlations"] = treatment_corr.tolist()

        return results

    def _draw_experiment_conclusion(self, statistical_analysis: Dict[str, Any],
                                  hypothesis: str) -> str:
        """Draw conclusion from statistical analysis"""
        significant_findings = []

        for test_name, test_results in statistical_analysis.items():
            if isinstance(test_results, dict):
                for metric, metric_results in test_results.items():
                    if isinstance(metric_results, dict):
                        for group, group_results in metric_results.items():
                            if isinstance(group_results, dict) and group_results.get("significant", False):
                                significant_findings.append(f"{test_name} on {metric} for {group}")

        if significant_findings:
            return f"Hypothesis supported by: {', '.join(significant_findings)}. Original hypothesis: {hypothesis}"
        else:
            return f"No significant effects detected. Hypothesis not supported: {hypothesis}"

    async def _design_ab_test(self, hypothesis: str, variables: Dict[str, Any],
                           sample_size: int) -> Dict[str, Any]:
        """Design an A/B test experiment"""
        # Extract key variable
        key_variable = list(variables.keys())[0] if variables else "treatment"

        control_group = {
            f"{key_variable}_distribution": "normal",
            f"{key_variable}_params": {"mean": 0, "std": 1}
        }

        treatment_groups = [{
            f"{key_variable}_distribution": "normal",
            f"{key_variable}_params": {"mean": 0.5, "std": 1}  # Expected effect
        }]

        return {
            "type": "ab_test",
            "control_group": control_group,
            "treatment_groups": treatment_groups,
            "expected_effect_size": 0.5,
            "power": await self._calculate_ab_test_power(sample_size, 0.5)
        }

    async def _design_factorial_experiment(self, hypothesis: str, variables: Dict[str, Any],
                                         sample_size: int) -> Dict[str, Any]:
        """Design a factorial experiment"""
        # Create combinations of variables
        var_names = list(variables.keys())
        var_values = [variables[name].get("levels", [0, 1]) for name in var_names]

        from itertools import product
        combinations = list(product(*var_values))

        control_group = {f"{name}_value": 0 for name in var_names}

        treatment_groups = []
        for combo in combinations[1:]:  # Skip all-zero combination (control)
            treatment = {f"{name}_value": value for name, value in zip(var_names, combo)}
            treatment_groups.append(treatment)

        return {
            "type": "factorial",
            "variables": var_names,
            "combinations": len(combinations),
            "control_group": control_group,
            "treatment_groups": treatment_groups
        }

    async def _design_multivariate_experiment(self, hypothesis: str, variables: Dict[str, Any],
                                           sample_size: int) -> Dict[str, Any]:
        """Design a multivariate experiment"""
        # Use experimental design principles
        control_group = {}
        treatment_groups = []

        for var_name, var_config in variables.items():
            levels = var_config.get("levels", 3)
            for level in range(1, levels):
                treatment = {f"{var_name}_level": level}
                treatment_groups.append(treatment)

        return {
            "type": "multivariate",
            "variables": list(variables.keys()),
            "control_group": control_group,
            "treatment_groups": treatment_groups
        }

    async def _calculate_statistical_power(self, experiment: Experiment) -> Dict[str, Any]:
        """Calculate statistical power for the experiment"""
        # Simplified power calculation
        effect_size = 0.5  # Medium effect
        alpha = 1 - self.default_confidence_level
        n = experiment.sample_size

        # For t-test power calculation
        power = self._calculate_t_test_power(effect_size, alpha, n)

        return {
            "effect_size": effect_size,
            "alpha": alpha,
            "sample_size": n,
            "power": power,
            "adequate_power": power > 0.8
        }

    def _calculate_t_test_power(self, effect_size: float, alpha: float, n: int) -> float:
        """Calculate power for t-test"""
        # Simplified calculation using normal approximation
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = effect_size * np.sqrt(n/2) - z_alpha

        power = 1 - stats.norm.cdf(z_beta)
        return power

    async def _calculate_ab_test_power(self, sample_size: int, effect_size: float) -> float:
        """Calculate power for A/B test"""
        return self._calculate_t_test_power(effect_size, 0.05, sample_size)

    async def _analyze_experiment_results(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze experiment results"""
        try:
            experiment_id = params.get("experiment_id")

            if experiment_id not in self.experiment_history:
                return {"status": "error", "error": f"No results found for experiment {experiment_id}"}

            results = self.experiment_history[experiment_id]

            # Aggregate analysis
            analysis = {
                "total_runs": len(results),
                "success_rate": len([r for r in results if "error" not in r.conclusion]) / len(results),
                "average_duration": np.mean([r.end_time - r.start_time for r in results]),
                "conclusions": [r.conclusion for r in results]
            }

            return {
                "status": "success",
                "experiment_id": experiment_id,
                "analysis": analysis
            }

        except Exception as e:
            logger.exception(f"Results analysis error: {e}")
            return {"status": "error", "error": str(e)}

    async def _create_sandbox_environment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a sandbox environment for safe experimentation"""
        try:
            sandbox_id = f"sandbox_{len(self.sandboxes)}_{int(time.time())}"

            sandbox = SandboxEnvironment(
                sandbox_id=sandbox_id,
                name=params.get("name", f"Sandbox {sandbox_id}"),
                description=params.get("description", "Isolated experimentation environment"),
                resources=params.get("resources", {}),
                isolation_level=params.get("isolation_level", "partial"),
                time_limit=params.get("time_limit", 300),
                memory_limit=params.get("memory_limit", 100 * 1024 * 1024),  # 100MB
                network_access=params.get("network_access", False),
                file_system_access=params.get("file_system_access", True)
            )

            self.sandboxes[sandbox_id] = sandbox

            return {
                "status": "success",
                "sandbox_id": sandbox_id,
                "message": f"Sandbox {sandbox_id} created with {sandbox.isolation_level} isolation"
            }

        except Exception as e:
            logger.exception(f"Sandbox creation error: {e}")
            return {"status": "error", "error": str(e)}

    async def _run_sandbox_experiment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run experiment in sandbox environment"""
        try:
            sandbox_id = params.get("sandbox_id")
            experiment_code = params.get("experiment_code", "")

            if sandbox_id not in self.sandboxes:
                return {"status": "error", "error": f"Sandbox {sandbox_id} not found"}

            sandbox = self.sandboxes[sandbox_id]

            # Emit sandbox experiment start event
            await self.emit_event("sandbox_experiment.started", {
                "sandbox_id": sandbox_id,
                "sandbox_name": sandbox.name,
                "isolation_level": sandbox.isolation_level,
                "time_limit": sandbox.time_limit,
                "code_length": len(experiment_code)
            })

            # Execute experiment in sandbox
            result = await self._execute_sandbox_experiment(sandbox, experiment_code)

            # Emit sandbox experiment completion event
            await self.emit_event("sandbox_experiment.completed", {
                "sandbox_id": sandbox_id,
                "execution_time": result.get("execution_time", 0),
                "success": result.get("success", False),
                "variables_created": len(result.get("local_variables", {}))
            })

            return {
                "status": "success",
                "sandbox_id": sandbox_id,
                "result": result
            }

        except Exception as e:
            logger.exception(f"Sandbox experiment error: {e}")
            # Emit sandbox experiment error event
            await self.emit_event("sandbox_experiment.error", {
                "sandbox_id": params.get("sandbox_id", "unknown"),
                "error": str(e),
                "error_type": type(e).__name__
            })
            return {"status": "error", "error": str(e)}

    async def _execute_sandbox_experiment(self, sandbox: SandboxEnvironment,
                                       experiment_code: str) -> Dict[str, Any]:
        """Execute experiment code in sandbox"""
        # Simplified sandbox execution - in real implementation would use containers/VMs
        try:
            # Create isolated execution environment
            local_vars = {}

            # Execute code with time and memory limits (simplified)
            start_time = time.time()

            # Basic security check - reject dangerous operations
            dangerous_patterns = ["import os", "import subprocess", "open(", "__import__"]
            for pattern in dangerous_patterns:
                if pattern in experiment_code:
                    return {"error": f"Dangerous operation detected: {pattern}"}

            # Execute the code
            exec(experiment_code, {"__builtins__": {}}, local_vars)

            execution_time = time.time() - start_time

            return {
                "execution_time": execution_time,
                "local_variables": {k: str(v) for k, v in local_vars.items()},
                "success": True
            }

        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }

    async def _validate_hypothesis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a hypothesis using statistical testing"""
        try:
            hypothesis = params.get("hypothesis", "")
            data = params.get("data", {})
            test_type = params.get("test_type", "t_test")

            if not data:
                return {"status": "error", "error": "No data provided for hypothesis testing"}

            # Perform the specified test
            if test_type in self.statistical_tests:
                test_func = self.statistical_tests[test_type]

                # Prepare data in expected format
                control_results = {"metric": data.get("control", [])}
                treatment_results = {"treatment_0": {"metric": data.get("treatment", [])}}

                analysis = await test_func(control_results, treatment_results)

                # Determine if hypothesis is supported
                supported = False
                if "metric" in analysis and "treatment_0" in analysis["metric"]:
                    result = analysis["metric"]["treatment_0"]
                    supported = result.get("significant", False)

                return {
                    "status": "success",
                    "hypothesis": hypothesis,
                    "test_type": test_type,
                    "analysis": analysis,
                    "supported": supported,
                    "confidence_level": self.default_confidence_level
                }
            else:
                return {"status": "error", "error": f"Unknown test type: {test_type}"}

        except Exception as e:
            logger.exception(f"Hypothesis validation error: {e}")
            return {"status": "error", "error": str(e)}

    async def _initialize_default_experiments(self):
        """Initialize default experiments"""
        # A/B test for conversion optimization
        ab_experiment = Experiment(
            experiment_id="ab_conversion_test",
            name="A/B Conversion Test",
            description="Test different UI designs on conversion rates",
            hypothesis="New design increases conversion rate by 10%",
            variables={"design_version": {"levels": ["A", "B"]}},
            control_group={"conversion_distribution": "beta", "conversion_params": {"alpha": 2, "beta": 8}},
            treatment_groups=[{"conversion_distribution": "beta", "conversion_params": {"alpha": 2.2, "beta": 7.8}}],
            sample_size=1000,
            duration=30,
            metrics=["conversion_rate"],
            statistical_tests=["t_test", "mann_whitney"]
        )

        # Multivariate feature test
        mv_experiment = Experiment(
            experiment_id="feature_interaction_test",
            name="Feature Interaction Test",
            description="Test interactions between multiple features",
            hypothesis="Feature combination A+B+C increases engagement",
            variables={
                "feature_A": {"levels": [0, 1]},
                "feature_B": {"levels": [0, 1]},
                "feature_C": {"levels": [0, 1]}
            },
            control_group={"engagement_value": 0},
            treatment_groups=[
                {"feature_A_value": 1, "engagement_value": 0.1},
                {"feature_B_value": 1, "engagement_value": 0.15},
                {"feature_C_value": 1, "engagement_value": 0.12},
                {"feature_A_value": 1, "feature_B_value": 1, "engagement_value": 0.3},
                {"feature_A_value": 1, "feature_C_value": 1, "engagement_value": 0.25},
                {"feature_B_value": 1, "feature_C_value": 1, "engagement_value": 0.32},
                {"feature_A_value": 1, "feature_B_value": 1, "feature_C_value": 1, "engagement_value": 0.5}
            ],
            sample_size=500,
            duration=45,
            metrics=["engagement"],
            statistical_tests=["anova", "t_test"]
        )

        self.experiments.update({
            "ab_conversion_test": ab_experiment,
            "feature_interaction_test": mv_experiment
        })

    async def _initialize_sandboxes(self):
        """Initialize default sandbox environments"""
        # Full isolation sandbox
        full_sandbox = SandboxEnvironment(
            sandbox_id="full_isolation",
            name="Full Isolation Sandbox",
            description="Completely isolated environment for high-risk experiments",
            resources={"cpu": 1, "memory": "256MB"},
            isolation_level="full",
            time_limit=60,
            memory_limit=256 * 1024 * 1024,
            network_access=False,
            file_system_access=False
        )

        # Partial isolation sandbox
        partial_sandbox = SandboxEnvironment(
            sandbox_id="partial_isolation",
            name="Partial Isolation Sandbox",
            description="Partially isolated environment for medium-risk experiments",
            resources={"cpu": 2, "memory": "512MB"},
            isolation_level="partial",
            time_limit=300,
            memory_limit=512 * 1024 * 1024,
            network_access=False,
            file_system_access=True
        )

        self.sandboxes.update({
            "full_isolation": full_sandbox,
            "partial_isolation": partial_sandbox
        })

    async def _warm_up_statistical_engines(self):
        """Warm up statistical computation engines"""
        try:
            # Run a quick statistical test
            test_data = np.random.normal(0, 1, 100)
            stats.ttest_1samp(test_data, 0)
            logger.info("Statistical engines warmed up")

        except Exception as e:
            logger.exception(f"Statistical engine warmup failed: {e}")

    def _update_experiment_stats(self, success: bool, duration: float):
        """Update experiment statistics"""
        self.experiment_stats["total_experiments"] += 1

        if success:
            self.experiment_stats["successful_experiments"] += 1
        else:
            self.experiment_stats["failed_experiments"] += 1

        self.experiment_stats["total_computation_time"] += duration

        if self.experiment_stats["total_experiments"] > 0:
            self.experiment_stats["average_duration"] = \
                self.experiment_stats["total_computation_time"] / self.experiment_stats["total_experiments"]

    async def shutdown(self) -> bool:
        """Shutdown the experimentation agent"""
        try:
            logger.info("ExperimentationAgent shutting down")

            # Clear active experiments
            self.active_experiments.clear()

            # Clear experiment history and definitions
            self.experiment_history.clear()
            self.experiments.clear()
            self.sandboxes.clear()

            logger.info("ExperimentationAgent shutdown complete")
            return True

        except Exception as e:
            logger.exception(f"Error during ExperimentationAgent shutdown: {e}")
            return False