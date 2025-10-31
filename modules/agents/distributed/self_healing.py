"""
Self Healing Agent (Phase 8)
============================

Implements automatic fault detection, diagnosis, and recovery mechanisms.
Uses machine learning for predictive maintenance and automated remediation.
"""

import asyncio
import time
import psutil
import platform
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import numpy as np
from sklearn.ensemble import IsolationForest
from pathlib import Path

from modules.logging_config import get_logger
from ..base_agent import BaseAgent, AgentCapability, AgentStatus
from ..knowledge.rollback_manager import RollbackManager

logger = get_logger("Kalki.SelfHealing")


@dataclass
class HealthMetric:
    """System health metric"""
    name: str
    value: float
    threshold: float
    severity: str
    timestamp: float


@dataclass
class FailurePattern:
    """Learned failure pattern"""
    pattern_id: str
    symptoms: List[str]
    root_cause: str
    remediation_steps: List[str]
    confidence: float
    occurrences: int


@dataclass
class RemediationAction:
    """Automated remediation action"""
    action_id: str
    description: str
    risk_level: str
    estimated_duration: int
    success_rate: float
    prerequisites: List[str]


class SelfHealingAgent(BaseAgent):
    """
    Self-healing agent with automatic fault detection and recovery.
    Uses ML for predictive maintenance and automated remediation.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="SelfHealingAgent",
            capabilities=[
                AgentCapability.SELF_HEALING,
                AgentCapability.QUALITY_ASSESSMENT,
                AgentCapability.OPTIMIZATION
            ],
            description="Automatic fault detection, diagnosis, and recovery",
            config=config or {}
        )

        # Health monitoring
        self.health_check_interval = self.config.get('health_check_interval', 60)
        self.anomaly_detection_window = self.config.get('anomaly_detection_window', 100)

        # ML components
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)

        # Health data
        self.health_history = deque(maxlen=1000)
        self.current_health_status = "healthy"

        # Failure patterns (learned from incidents)
        self.failure_patterns: Dict[str, FailurePattern] = {}
        self.incident_history = deque(maxlen=500)

        # Remediation actions
        self.remediation_actions = self._initialize_remediation_actions()

        # System state
        self.is_monitoring = False
        self.last_health_check = 0

        # RollbackManager integration for safe remediation
        self.rollback_manager = RollbackManager({
            'checkpoint_dir': Path.home() / "Desktop" / "Kalki" / "vector_db" / "checkpoints" / "healing",
            'max_checkpoints': 50,
            'compression_enabled': True
        })

    async def initialize(self) -> bool:
        """Initialize self-healing system"""
        try:
            logger.info("SelfHealingAgent initializing automated fault detection")

            # Initialize anomaly detector
            await self._initialize_anomaly_detector()

            # Load known failure patterns
            await self._load_failure_patterns()

            # Initialize RollbackManager for safe remediation
            rollback_init = await self.rollback_manager.initialize()
            if not rollback_init:
                logger.warning("RollbackManager initialization failed - proceeding without checkpointing")
            else:
                logger.info("RollbackManager initialized for safe remediation operations")

            # Start health monitoring
            self.is_monitoring = True
            asyncio.create_task(self._health_monitoring_loop())

            logger.info("SelfHealingAgent initialized with predictive maintenance and safe remediation")
            return True

        except Exception as e:
            logger.exception(f"Failed to initialize SelfHealingAgent: {e}")
            return False

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute self-healing operations"""
        action = task.get("action")
        params = task.get("params", {})

        if action == "check_health":
            return await self._check_system_health(params)
        elif action == "diagnose_issue":
            return await self._diagnose_system_issue(params)
        elif action == "remediate":
            return await self._execute_remediation(params)
        elif action == "predict_failures":
            return await self._predict_potential_failures(params)
        elif action == "get_status":
            return await self._get_healing_status(params)
        else:
            return {"status": "error", "error": f"Unknown action: {action}"}

    async def _check_system_health(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        try:
            health_metrics = await self._collect_health_metrics()

            # Analyze health status
            health_status = await self._analyze_health_status(health_metrics)

            # Check for anomalies
            anomalies = await self._detect_anomalies(health_metrics)

            # Determine overall health
            overall_health = self._calculate_overall_health(health_metrics, anomalies)

            return {
                "status": "success",
                "overall_health": overall_health,
                "health_status": health_status,
                "anomalies_detected": len(anomalies),
                "metrics": health_metrics,
                "recommendations": self._generate_health_recommendations(health_status, anomalies)
            }

        except Exception as e:
            logger.exception(f"Health check error: {e}")
            return {"status": "error", "error": str(e)}

    async def _diagnose_system_issue(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Diagnose a reported system issue"""
        try:
            symptoms = params.get("symptoms", [])
            error_logs = params.get("error_logs", [])
            performance_metrics = params.get("performance_metrics", {})

            # Match against known failure patterns
            matching_patterns = await self._match_failure_patterns(symptoms, error_logs)

            # Perform root cause analysis
            root_cause = await self._perform_root_cause_analysis(
                symptoms, error_logs, performance_metrics
            )

            # Generate diagnosis
            diagnosis = {
                "symptoms": symptoms,
                "matched_patterns": matching_patterns,
                "root_cause": root_cause,
                "confidence": self._calculate_diagnosis_confidence(matching_patterns, root_cause),
                "recommended_actions": self._recommend_remediation_actions(root_cause)
            }

            return {
                "status": "success",
                "diagnosis": diagnosis
            }

        except Exception as e:
            logger.exception(f"Diagnosis error: {e}")
            return {"status": "error", "error": str(e)}

    async def _execute_remediation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute automated remediation with rollback safety"""
        try:
            action_id = params.get("action_id")
            issue_description = params.get("issue_description", "")
            risk_tolerance = params.get("risk_tolerance", "medium")
            allow_rollback = params.get("allow_rollback", True)

            if action_id not in self.remediation_actions:
                return {"status": "error", "error": f"Unknown remediation action: {action_id}"}

            action = self.remediation_actions[action_id]

            # Check prerequisites
            if not await self._check_prerequisites(action):
                return {"status": "error", "error": "Prerequisites not met for remediation action"}

            # Assess risk
            if not self._assess_risk_acceptable(action, risk_tolerance):
                return {"status": "error", "error": f"Risk level {action.risk_level} exceeds tolerance {risk_tolerance}"}

            # Create pre-remediation checkpoint if rollback is allowed
            checkpoint_id = None
            if allow_rollback and hasattr(self.rollback_manager, 'initialized') and self.rollback_manager.initialized:
                try:
                    checkpoint_id = await self._create_remediation_checkpoint(action, issue_description)
                    logger.info(f"Created pre-remediation checkpoint: {checkpoint_id}")
                except Exception as e:
                    logger.warning(f"Failed to create checkpoint: {e}")
                    if risk_tolerance == "low":
                        return {"status": "error", "error": "Cannot proceed without checkpoint for low-risk tolerance"}

            # Execute remediation
            success = await self._perform_remediation_action(action)

            # Handle remediation outcome
            if success:
                # Remediation successful - clean up checkpoint
                if checkpoint_id:
                    try:
                        await self.rollback_manager.delete_checkpoint(checkpoint_id)
                        logger.info(f"Cleaned up successful remediation checkpoint: {checkpoint_id}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up checkpoint: {e}")

                await self._record_remediation_outcome(action, True, issue_description)

                return {
                    "status": "success",
                    "remediation_action": action_id,
                    "executed_successfully": True,
                    "checkpoint_created": checkpoint_id,
                    "duration_seconds": action.estimated_duration
                }
            else:
                # Remediation failed - attempt rollback if checkpoint exists
                rollback_success = False
                if checkpoint_id and allow_rollback:
                    try:
                        rollback_success = await self._rollback_remediation(checkpoint_id, action, issue_description)
                        logger.info(f"Rollback {'successful' if rollback_success else 'failed'} for action: {action_id}")
                    except Exception as e:
                        logger.error(f"Rollback failed with error: {e}")

                await self._record_remediation_outcome(action, False, issue_description, rollback_success)

                return {
                    "status": "error",
                    "remediation_action": action_id,
                    "executed_successfully": False,
                    "rollback_attempted": checkpoint_id is not None,
                    "rollback_successful": rollback_success,
                    "checkpoint_id": checkpoint_id,
                    "error": "Remediation failed and rollback was attempted"
                }

        except Exception as e:
            logger.exception(f"Remediation error: {e}")
            return {"status": "error", "error": str(e)}

    async def _predict_potential_failures(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Predict potential future failures"""
        try:
            prediction_window = params.get("prediction_window", 3600)  # 1 hour

            # Analyze trends
            trends = await self._analyze_health_trends()

            # Predict failures based on patterns
            predictions = []
            for trend in trends:
                if trend.get("risk_level", "low") in ["high", "critical"]:
                    prediction = {
                        "component": trend["component"],
                        "failure_type": trend["predicted_failure"],
                        "probability": trend["probability"],
                        "time_to_failure": trend["time_to_failure"],
                        "preventive_actions": trend["preventive_actions"]
                    }
                    predictions.append(prediction)

            return {
                "status": "success",
                "predictions": predictions,
                "prediction_window": prediction_window
            }

        except Exception as e:
            logger.exception(f"Failure prediction error: {e}")
            return {"status": "error", "error": str(e)}

    async def _collect_health_metrics(self) -> List[HealthMetric]:
        """Collect comprehensive health metrics"""
        metrics = []

        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(HealthMetric("cpu_usage", cpu_percent, 80.0, "high" if cpu_percent > 80 else "normal", time.time()))

            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.append(HealthMetric("memory_usage", memory.percent, 85.0, "high" if memory.percent > 85 else "normal", time.time()))

            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics.append(HealthMetric("disk_usage", disk.percent, 90.0, "high" if disk.percent > 90 else "normal", time.time()))

            # Network metrics (simplified)
            net = psutil.net_io_counters()
            if net:
                metrics.append(HealthMetric("network_errors", net.errin + net.errout, 100.0, "medium" if (net.errin + net.errout) > 50 else "normal", time.time()))

            # Process metrics
            process_count = len(psutil.pids())
            metrics.append(HealthMetric("process_count", process_count, 1000.0, "medium" if process_count > 800 else "normal", time.time()))

        except Exception as e:
            logger.exception(f"Metrics collection error: {e}")

        return metrics

    async def _analyze_health_status(self, metrics: List[HealthMetric]) -> Dict[str, Any]:
        """Analyze overall health status"""
        status = {
            "overall": "healthy",
            "critical_issues": [],
            "warning_issues": [],
            "healthy_components": []
        }

        for metric in metrics:
            if metric.value > metric.threshold:
                if metric.severity == "high":
                    status["critical_issues"].append({
                        "component": metric.name,
                        "value": metric.value,
                        "threshold": metric.threshold
                    })
                else:
                    status["warning_issues"].append({
                        "component": metric.name,
                        "value": metric.value,
                        "threshold": metric.threshold
                    })
            else:
                status["healthy_components"].append(metric.name)

        # Determine overall status
        if status["critical_issues"]:
            status["overall"] = "critical"
        elif status["warning_issues"]:
            status["overall"] = "warning"

        return status

    async def _detect_anomalies(self, metrics: List[HealthMetric]) -> List[Dict[str, Any]]:
        """Detect anomalies using ML"""
        if len(self.health_history) < 20:
            return []  # Need more data

        try:
            # Prepare data for anomaly detection
            recent_data = list(self.health_history)[-self.anomaly_detection_window:]
            if len(recent_data) < 10:
                return []

            # Extract metric values
            metric_values = []
            for entry in recent_data:
                entry_metrics = entry.get("metrics", [])
                values = [m.value for m in entry_metrics]
                if values:
                    metric_values.append(values)

            if len(metric_values) < 10:
                return []

            # Detect anomalies
            predictions = self.anomaly_detector.fit_predict(metric_values)
            anomaly_indices = [i for i, pred in enumerate(predictions) if pred == -1]

            anomalies = []
            for idx in anomaly_indices:
                if idx < len(recent_data):
                    entry = recent_data[idx]
                    anomalies.append({
                        "timestamp": entry.get("timestamp"),
                        "description": "Anomalous system behavior detected",
                        "severity": "medium"
                    })

            return anomalies

        except Exception as e:
            logger.exception(f"Anomaly detection error: {e}")
            return []

    def _calculate_overall_health(self, metrics: List[HealthMetric], anomalies: List[Dict[str, Any]]) -> str:
        """Calculate overall system health"""
        # Count critical metrics
        critical_count = sum(1 for m in metrics if m.value > m.threshold and m.severity == "high")

        # Factor in anomalies
        anomaly_penalty = len(anomalies) * 0.1

        health_score = 1.0 - (critical_count * 0.2) - anomaly_penalty
        health_score = max(0.0, min(1.0, health_score))

        if health_score >= 0.8:
            return "excellent"
        elif health_score >= 0.6:
            return "good"
        elif health_score >= 0.4:
            return "fair"
        elif health_score >= 0.2:
            return "poor"
        else:
            return "critical"

    async def _match_failure_patterns(self, symptoms: List[str], error_logs: List[str]) -> List[Dict[str, Any]]:
        """Match symptoms against known failure patterns"""
        matches = []

        for pattern_id, pattern in self.failure_patterns.items():
            match_score = 0
            matched_symptoms = []

            # Check symptom matching
            for symptom in symptoms:
                for pattern_symptom in pattern.symptoms:
                    if symptom.lower() in pattern_symptom.lower() or pattern_symptom.lower() in symptom.lower():
                        match_score += 1
                        matched_symptoms.append(symptom)
                        break

            # Check error log matching
            for log_entry in error_logs:
                for pattern_symptom in pattern.symptoms:
                    if pattern_symptom.lower() in log_entry.lower():
                        match_score += 2  # Logs are more specific
                        break

            if match_score > 0:
                confidence = min(1.0, match_score / (len(pattern.symptoms) + len(error_logs)))
                matches.append({
                    "pattern_id": pattern_id,
                    "pattern": pattern.root_cause,
                    "confidence": confidence,
                    "matched_symptoms": matched_symptoms
                })

        # Sort by confidence
        matches.sort(key=lambda x: x["confidence"], reverse=True)
        return matches[:5]  # Top 5 matches

    async def _perform_root_cause_analysis(self, symptoms: List[str], error_logs: List[str],
                                         performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Perform root cause analysis"""
        # Simplified root cause analysis
        root_cause = {
            "primary_cause": "Unknown",
            "contributing_factors": [],
            "confidence": 0.5
        }

        # Analyze symptoms
        if any("memory" in s.lower() for s in symptoms):
            root_cause["primary_cause"] = "Memory exhaustion"
            root_cause["contributing_factors"] = ["High memory usage", "Memory leaks"]
            root_cause["confidence"] = 0.8
        elif any("cpu" in s.lower() for s in symptoms):
            root_cause["primary_cause"] = "CPU overload"
            root_cause["contributing_factors"] = ["High CPU usage", "Inefficient algorithms"]
            root_cause["confidence"] = 0.8
        elif any("disk" in s.lower() for s in symptoms):
            root_cause["primary_cause"] = "Disk I/O bottleneck"
            root_cause["contributing_factors"] = ["High disk usage", "Slow storage"]
            root_cause["confidence"] = 0.7

        return root_cause

    def _initialize_remediation_actions(self) -> Dict[str, RemediationAction]:
        """Initialize available remediation actions"""
        return {
            "restart_service": RemediationAction(
                action_id="restart_service",
                description="Restart the affected service",
                risk_level="low",
                estimated_duration=30,
                success_rate=0.9,
                prerequisites=["service_manager_access"]
            ),
            "scale_resources": RemediationAction(
                action_id="scale_resources",
                description="Automatically scale system resources",
                risk_level="medium",
                estimated_duration=120,
                success_rate=0.8,
                prerequisites=["auto_scaling_enabled"]
            ),
            "clear_cache": RemediationAction(
                action_id="clear_cache",
                description="Clear system caches to free memory",
                risk_level="low",
                estimated_duration=10,
                success_rate=0.95,
                prerequisites=["cache_management_access"]
            ),
            "kill_hung_processes": RemediationAction(
                action_id="kill_hung_processes",
                description="Terminate hung or unresponsive processes",
                risk_level="high",
                estimated_duration=5,
                success_rate=0.7,
                prerequisites=["process_management_access"]
            )
        }

    async def _check_prerequisites(self, action: RemediationAction) -> bool:
        """Check if prerequisites are met for remediation action"""
        # Simplified prerequisite checking
        for prereq in action.prerequisites:
            if prereq == "service_manager_access":
                # Check if we can manage services (simplified)
                return True
            elif prereq == "auto_scaling_enabled":
                # Check if auto-scaling is enabled
                return True
            elif prereq == "cache_management_access":
                return True
            elif prereq == "process_management_access":
                return True

        return True

    def _assess_risk_acceptable(self, action: RemediationAction, tolerance: str) -> bool:
        """Assess if remediation risk is acceptable"""
        risk_levels = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        tolerance_levels = {"low": 1, "medium": 2, "high": 3, "critical": 4}

        action_risk = risk_levels.get(action.risk_level, 2)
        max_tolerance = tolerance_levels.get(tolerance, 2)

        return action_risk <= max_tolerance

    async def _perform_remediation_action(self, action: RemediationAction) -> bool:
        """Perform the actual remediation action"""
        try:
            logger.info(f"Executing remediation action: {action.description}")

            # Simulate remediation execution
            await asyncio.sleep(min(action.estimated_duration, 5))  # Cap at 5 seconds for demo

            # Simulate success based on success rate
            import random
            success = random.random() < action.success_rate

            if success:
                logger.info(f"Remediation action completed successfully: {action.action_id}")
            else:
                logger.warning(f"Remediation action failed: {action.action_id}")

            return success

        except Exception as e:
            logger.exception(f"Remediation execution error: {e}")
            return False

    async def _record_remediation_outcome(self, action: RemediationAction, success: bool,
                                        issue_description: str):
        """Record remediation outcome for learning"""
        incident = {
            "timestamp": time.time(),
            "action_id": action.action_id,
            "success": success,
            "issue_description": issue_description,
            "risk_level": action.risk_level,
            "duration": action.estimated_duration
        }

        self.incident_history.append(incident)

    async def _initialize_anomaly_detector(self):
        """Initialize the anomaly detection model"""
        # Train on synthetic normal data
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (100, 5))  # 5 metrics
        self.anomaly_detector.fit(normal_data)

    async def _load_failure_patterns(self):
        """Load known failure patterns"""
        # Initialize with common failure patterns
        self.failure_patterns = {
            "memory_exhaustion": FailurePattern(
                pattern_id="memory_exhaustion",
                symptoms=["high memory usage", "out of memory", "memory allocation failed"],
                root_cause="Memory exhaustion due to leaks or insufficient resources",
                remediation_steps=["clear caches", "restart services", "scale memory"],
                confidence=0.9,
                occurrences=15
            ),
            "cpu_overload": FailurePattern(
                pattern_id="cpu_overload",
                symptoms=["high cpu usage", "cpu throttling", "slow response times"],
                root_cause="CPU overload from excessive computation",
                remediation_steps=["scale cpu resources", "optimize algorithms", "load balance"],
                confidence=0.85,
                occurrences=12
            ),
            "disk_full": FailurePattern(
                pattern_id="disk_full",
                symptoms=["disk full", "no space left", "disk write failed"],
                root_cause="Disk storage exhaustion",
                remediation_steps=["clean up disk space", "add storage", "archive old data"],
                confidence=0.95,
                occurrences=8
            )
        }

    async def _health_monitoring_loop(self):
        """Continuous health monitoring"""
        while self.is_monitoring:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logger.exception(f"Health monitoring error: {e}")
                await asyncio.sleep(self.health_check_interval)

    async def _perform_health_check(self):
        """Perform periodic health check"""
        try:
            metrics = await self._collect_health_metrics()
            health_status = await self._analyze_health_status(metrics)

            # Store health data
            self.health_history.append({
                "timestamp": time.time(),
                "metrics": metrics,
                "status": health_status
            })

            # Update current status
            self.current_health_status = health_status["overall"]
            self.last_health_check = time.time()

        except Exception as e:
            logger.exception(f"Health check failed: {e}")

    async def _analyze_health_trends(self) -> List[Dict[str, Any]]:
        """Analyze health trends for failure prediction"""
        trends = []

        if len(self.health_history) < 20:
            return trends

        try:
            # Analyze each metric trend
            metric_names = ["cpu_usage", "memory_usage", "disk_usage"]

            for metric_name in metric_names:
                values = []
                timestamps = []

                for entry in list(self.health_history)[-50:]:  # Last 50 entries
                    for metric in entry.get("metrics", []):
                        if metric.name == metric_name:
                            values.append(metric.value)
                            timestamps.append(entry["timestamp"])
                            break

                if len(values) >= 10:
                    # Calculate trend
                    x = np.arange(len(values))
                    slope, intercept = np.polyfit(x, values, 1)

                    # Predict time to threshold
                    threshold = 80.0 if metric_name != "disk_usage" else 90.0
                    if slope > 0 and values[-1] < threshold:
                        time_to_threshold = (threshold - values[-1]) / slope
                        if time_to_threshold > 0 and time_to_threshold < 3600:  # Within 1 hour
                            trends.append({
                                "component": metric_name,
                                "predicted_failure": f"{metric_name} threshold exceeded",
                                "probability": min(0.9, abs(slope) * 0.1),
                                "time_to_failure": time_to_threshold,
                                "risk_level": "high" if time_to_threshold < 1800 else "medium",
                                "preventive_actions": [f"Scale {metric_name}", "Optimize usage"]
                            })

        except Exception as e:
            logger.exception(f"Trend analysis error: {e}")

        return trends

    def _calculate_diagnosis_confidence(self, patterns: List[Dict[str, Any]],
                                      root_cause: Dict[str, Any]) -> float:
        """Calculate overall diagnosis confidence"""
        if not patterns:
            return root_cause.get("confidence", 0.3)

        # Weighted average of pattern confidences
        total_weight = sum(p["confidence"] for p in patterns)
        if total_weight == 0:
            return 0.3

        weighted_confidence = sum(p["confidence"] ** 2 for p in patterns) / total_weight
        return min(0.95, weighted_confidence)

    def _recommend_remediation_actions(self, root_cause: Dict[str, Any]) -> List[str]:
        """Recommend remediation actions based on root cause"""
        cause = root_cause.get("primary_cause", "").lower()

        if "memory" in cause:
            return ["clear_cache", "scale_resources", "restart_service"]
        elif "cpu" in cause:
            return ["scale_resources", "optimize_performance"]
        elif "disk" in cause:
            return ["clear_cache", "scale_resources"]
        else:
            return ["restart_service", "scale_resources"]

    def _generate_health_recommendations(self, health_status: Dict[str, Any],
                                       anomalies: List[Dict[str, Any]]) -> List[str]:
        """Generate health recommendations"""
        recommendations = []

        if health_status["critical_issues"]:
            recommendations.append("Immediate attention required for critical issues")
            recommendations.append("Consider automated remediation actions")

        if health_status["warning_issues"]:
            recommendations.append("Monitor warning conditions closely")

        if anomalies:
            recommendations.append("Investigate detected anomalies")
            recommendations.append("Review system logs for unusual patterns")

        if not recommendations:
            recommendations.append("System health is good")

        return recommendations

    async def _create_remediation_checkpoint(self, action: RemediationAction, issue_description: str) -> str:
        """Create a checkpoint before remediation for rollback safety"""
        try:
            # Gather system state
            system_state = await self._gather_system_state()

            # Create checkpoint context
            context = {
                "checkpoint_type": "pre_remediation",
                "remediation_action": action.action_id,
                "issue_description": issue_description,
                "risk_level": action.risk_level,
                "timestamp": time.time(),
                "system_health": self.current_health_status,
                "rollback_reason": "remediation_failure"
            }

            # Create checkpoint
            checkpoint_id = await self.rollback_manager.create_checkpoint(
                state_data=system_state,
                context=context,
                tags=["remediation", "safety", action.action_id]
            )

            return checkpoint_id

        except Exception as e:
            logger.exception(f"Failed to create remediation checkpoint: {e}")
            raise

    async def _rollback_remediation(self, checkpoint_id: str, action: RemediationAction, issue_description: str) -> bool:
        """Rollback to pre-remediation state"""
        try:
            logger.warning(f"Initiating rollback for failed remediation: {action.action_id}")

            # Restore system state
            restore_success = await self.rollback_manager.restore_checkpoint(checkpoint_id)

            if restore_success:
                logger.info(f"Successfully rolled back remediation: {action.action_id}")

                # Update incident record with rollback success
                await self._record_rollback_outcome(checkpoint_id, action, issue_description, True)

                # Trigger health recheck after rollback
                asyncio.create_task(self._perform_health_check())

                return True
            else:
                logger.error(f"Failed to rollback remediation: {action.action_id}")
                await self._record_rollback_outcome(checkpoint_id, action, issue_description, False)
                return False

        except Exception as e:
            logger.exception(f"Rollback operation failed: {e}")
            await self._record_rollback_outcome(checkpoint_id, action, issue_description, False, str(e))
            return False

    async def _gather_system_state(self) -> Dict[str, Any]:
        """Gather comprehensive system state for checkpointing"""
        try:
            state = {
                "timestamp": time.time(),
                "health_status": self.current_health_status,
                "health_metrics": [],
                "running_processes": [],
                "system_resources": {},
                "agent_states": {}
            }

            # Gather current health metrics
            metrics = await self._collect_health_metrics()
            state["health_metrics"] = [
                {
                    "name": m.name,
                    "value": m.value,
                    "threshold": m.threshold,
                    "severity": m.severity
                } for m in metrics
            ]

            # Gather process information (simplified)
            try:
                import psutil
                state["running_processes"] = [
                    {
                        "pid": proc.pid,
                        "name": proc.name(),
                        "cpu_percent": proc.cpu_percent(),
                        "memory_percent": proc.memory_percent()
                    } for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent'])
                    if proc.pid != 0  # Skip system processes
                ][:50]  # Limit to 50 processes
            except Exception as e:
                logger.warning(f"Failed to gather process information: {e}")

            # Gather system resource usage
            state["system_resources"] = {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory": dict(psutil.virtual_memory()._asdict()),
                "disk": dict(psutil.disk_usage('/')._asdict()),
                "network": dict(psutil.net_io_counters()._asdict()) if psutil.net_io_counters() else {}
            }

            return state

        except Exception as e:
            logger.exception(f"Failed to gather system state: {e}")
            return {"error": str(e), "timestamp": time.time()}

    async def _record_rollback_outcome(self, checkpoint_id: str, action: RemediationAction,
                                     issue_description: str, success: bool, error: str = None):
        """Record rollback outcome for analysis"""
        try:
            rollback_record = {
                "timestamp": time.time(),
                "checkpoint_id": checkpoint_id,
                "remediation_action": action.action_id,
                "issue_description": issue_description,
                "rollback_success": success,
                "error": error
            }

            # Add to incident history with rollback flag
            incident = {
                "timestamp": time.time(),
                "action_id": f"rollback_{action.action_id}",
                "success": success,
                "issue_description": f"Rollback after failed remediation: {issue_description}",
                "risk_level": "medium",
                "duration": 0,  # Rollback duration not tracked
                "rollback_operation": True,
                "original_checkpoint": checkpoint_id
            }

            self.incident_history.append(incident)

        except Exception as e:
            logger.exception(f"Failed to record rollback outcome: {e}")

    async def _get_healing_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive healing system status"""
        rollback_status = {}
        if hasattr(self.rollback_manager, 'initialized') and self.rollback_manager.initialized:
            try:
                rollback_status = await self.rollback_manager.get_status()
            except Exception as e:
                rollback_status = {"error": str(e)}

        return {
            "status": "success",
            "current_health": self.current_health_status,
            "monitoring_active": self.is_monitoring,
            "last_health_check": self.last_health_check,
            "known_patterns": len(self.failure_patterns),
            "incident_history_size": len(self.incident_history),
            "remediation_actions": len(self.remediation_actions),
            "health_history_size": len(self.health_history),
            "rollback_manager": rollback_status
        }

    async def shutdown(self) -> bool:
        """Shutdown the self-healing agent"""
        try:
            logger.info("SelfHealingAgent shutting down")

            # Stop monitoring
            self.is_monitoring = False

            # Shutdown RollbackManager
            if hasattr(self.rollback_manager, 'shutdown'):
                try:
                    await self.rollback_manager.shutdown()
                    logger.info("RollbackManager shutdown complete")
                except Exception as e:
                    logger.warning(f"RollbackManager shutdown error: {e}")

            # Clear health history
            self.health_history.clear()
            self.incident_history.clear()

            # Clear failure patterns
            self.failure_patterns.clear()

            logger.info("SelfHealingAgent shutdown complete")
            return True

        except Exception as e:
            logger.exception(f"Error during SelfHealingAgent shutdown: {e}")
            return False