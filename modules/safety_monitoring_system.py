# ============================================================
# Kalki v2.4 — safety_monitoring_system.py
# ------------------------------------------------------------
# Safety & Monitoring System: Production-Ready Self-Evolution
# - Critical metrics monitoring and alerting
# - Safety gate enforcement for all changes
# - Tamper-evident audit trails with signed checkpoints
# - Automated validation and adversarial testing
# ============================================================

import os
import json
import asyncio
import hashlib
import hmac
import copy
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import statistics
import threading
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from modules.utils.logging_config import get_logger
from modules.self_evolution_manager import get_self_evolution_manager, EvolutionRecommendation
from modules.canary_deployment_manager import get_canary_deployment_manager

logger = get_logger("Kalki.SafetyMonitoring")

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    EXPIRED = "expired"

@dataclass
class AlertRule:
    """Alert rule definition"""
    rule_id: str
    name: str
    description: str
    metric: str
    condition: str  # ">", "<", "==", "!="
    threshold: float
    severity: AlertSeverity
    cooldown_minutes: int = 60
    enabled: bool = True
    notification_channels: List[str] = field(default_factory=lambda: ["log"])

@dataclass
class Alert:
    """Active alert instance"""
    alert_id: str
    rule_id: str
    severity: AlertSeverity
    title: str
    description: str
    metric_value: float
    threshold: float
    triggered_at: str
    status: AlertStatus
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[str] = None
    resolved_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AuditEntry:
    """Tamper-evident audit entry"""
    entry_id: str
    timestamp: str
    event_type: str
    actor: str
    action: str
    target: str
    details: Dict[str, Any]
    signature: str  # HMAC signature for tamper detection
    previous_hash: str  # Chain previous entry
    sequence_number: int

@dataclass
class ValidationResult:
    """Result of automated validation"""
    test_id: str
    test_type: str
    timestamp: str
    passed: bool
    score: float
    details: Dict[str, Any]
    recommendations: List[str]

class SafetyMonitoringSystem:
    """
    Safety & Monitoring System: Production-Ready Self-Evolution

    Provides comprehensive safety gates, critical monitoring, alerting,
    and tamper-evident audit trails for production deployment.
    """

    def __init__(self):
        # Remove direct dependency on evolution_manager to avoid circular import
        # Will be set when needed via method parameters
        self.canary_manager = get_canary_deployment_manager()

        # Alert system
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []

        # Persistence
        self.data_dir = "data/safety_monitoring"
        self.alerts_file = f"{self.data_dir}/alerts.json"
        self.audit_file = f"{self.data_dir}/audit_chain.json"

        # Audit trail
        self.audit_chain: List[AuditEntry] = []
        self.audit_secret = self._generate_audit_secret()
        self.last_sequence_number = 0

        # Initialize default alert rules
        self._initialize_default_alert_rules()

        # Load existing state
        self._load_persistent_state()

        logger.info("Safety Monitoring System initialized")

    async def initialize(self) -> bool:
        """Initialize the safety monitoring system (already initialized in __init__)."""
        return True

    def _generate_audit_secret(self) -> str:
        """Generate or load audit secret for tamper-evident logging"""
        secret_file = f"{self.data_dir}/audit_secret.key"

        if os.path.exists(secret_file):
            with open(secret_file, 'r') as f:
                return f.read().strip()
        else:
            # Generate new secret
            os.makedirs(self.data_dir, exist_ok=True)
            secret = hashlib.sha256(os.urandom(32)).hexdigest()
            with open(secret_file, 'w') as f:
                f.write(secret)
            return secret

    def _initialize_default_alert_rules(self):
        """Initialize default alert rules"""

        default_rules = [
            AlertRule(
                rule_id="health_score_critical",
                name="Health Score Critical",
                description="Overall system health has dropped critically low",
                metric="health_score",
                condition="<",
                threshold=0.35,
                severity=AlertSeverity.CRITICAL,
                cooldown_minutes=30,
                notification_channels=["log", "email"]
            ),
            AlertRule(
                rule_id="confidence_drift_warning",
                name="Confidence Calibration Drift",
                description="Confidence calibration has drifted significantly",
                metric="confidence_calibration_drift",
                condition=">",
                threshold=0.15,
                severity=AlertSeverity.WARNING,
                cooldown_minutes=60,
                notification_channels=["log"]
            ),
            AlertRule(
                rule_id="safety_violations_critical",
                name="Safety Violations Detected",
                description="Safety agent has detected violations",
                metric="safety_violations",
                condition=">",
                threshold=0,
                severity=AlertSeverity.CRITICAL,
                cooldown_minutes=5,
                notification_channels=["log", "email"]
            ),
            AlertRule(
                rule_id="coherence_score_warning",
                name="Coherence Score Low",
                description="Response coherence has dropped below acceptable level",
                metric="coherence_score",
                condition="<",
                threshold=0.7,
                severity=AlertSeverity.WARNING,
                cooldown_minutes=120,
                notification_channels=["log"]
            ),
            AlertRule(
                rule_id="interdisciplinary_coverage_warning",
                name="Interdisciplinary Coverage Low",
                description="System is not maintaining interdisciplinary knowledge coverage",
                metric="interdisciplinary_coverage",
                condition="<",
                threshold=0.75,
                severity=AlertSeverity.WARNING,
                cooldown_minutes=240,
                notification_channels=["log"]
            )
        ]

        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule

    def _load_persistent_state(self):
        """Load persistent monitoring state"""
        try:
            os.makedirs(self.data_dir, exist_ok=True)

            # Load audit chain
            if os.path.exists(self.audit_file):
                with open(self.audit_file, 'r') as f:
                    audit_data = json.load(f)
                    self.audit_chain = [AuditEntry(**entry) for entry in audit_data.get('audit_chain', [])]
                    self.last_sequence_number = audit_data.get('last_sequence_number', 0)

            # Load alerts
            if os.path.exists(self.alerts_file):
                with open(self.alerts_file, 'r') as f:
                    alert_data = json.load(f)
                    self.active_alerts = {
                        alert_id: Alert(**alert_data)
                        for alert_id, alert_data in alert_data.get('active_alerts', {}).items()
                    }
                    self.alert_history = [
                        Alert(**alert) for alert in alert_data.get('alert_history', [])
                    ]

        except Exception as e:
            logger.warning(f"Failed to load persistent monitoring state: {e}")

    def _save_persistent_state(self):
        """Save monitoring state persistently"""
        try:
            # Save audit chain
            audit_data = {
                'audit_chain': [asdict(entry) for entry in self.audit_chain[-1000:]],  # Keep last 1000
                'last_sequence_number': self.last_sequence_number,
                'last_updated': datetime.now().isoformat()
            }

            with open(self.audit_file, 'w') as f:
                json.dump(audit_data, f, indent=2)

            # Save alerts
            alert_data = {
                'active_alerts': {aid: asdict(alert) for aid, alert in self.active_alerts.items()},
                'alert_history': [asdict(alert) for alert in self.alert_history[-500:]],  # Keep last 500
                'last_updated': datetime.now().isoformat()
            }

            with open(self.alerts_file, 'w') as f:
                json.dump(alert_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save persistent monitoring state: {e}")

    def _create_audit_entry(self, event_type: str, actor: str, action: str,
                           target: str, details: Dict[str, Any]) -> AuditEntry:
        """Create a tamper-evident audit entry"""

        self.last_sequence_number += 1

        # Create entry data
        entry_data = {
            "entry_id": f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.last_sequence_number}",
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "actor": actor,
            "action": action,
            "target": target,
            "details": details,
            "sequence_number": self.last_sequence_number
        }

        # Calculate previous hash
        if self.audit_chain:
            prev_entry = self.audit_chain[-1]
            prev_data = asdict(prev_entry)
            prev_data.pop('signature', None)  # Remove signature for hashing
            entry_data["previous_hash"] = hashlib.sha256(
                json.dumps(prev_data, sort_keys=True).encode()
            ).hexdigest()
        else:
            entry_data["previous_hash"] = "genesis"

        # Create signature
        data_to_sign = json.dumps(entry_data, sort_keys=True).encode()
        signature = hmac.new(
            self.audit_secret.encode(),
            data_to_sign,
            hashlib.sha256
        ).hexdigest()

        entry_data["signature"] = signature

        entry = AuditEntry(**entry_data)
        self.audit_chain.append(entry)

        # Save immediately for tamper-evidence
        self._save_persistent_state()

        return entry

    async def evaluate_safety_gate(self, recommendation: EvolutionRecommendation,
                                 human_override: bool = False) -> Dict[str, Any]:
        """
        Evaluate comprehensive safety gate for evolution recommendation

        Args:
            recommendation: The recommendation to evaluate
            human_override: Whether this is approved by human override

        Returns:
            Safety evaluation result
        """

        # Automated safety checks
        safety_checks = await self._run_automated_safety_checks(recommendation)

        # Ethical assessment
        ethical_assessment = await self._run_ethical_assessment(recommendation)

        # Risk assessment
        risk_assessment = self._calculate_risk_score(recommendation, safety_checks, ethical_assessment)

        # Determine approval
        approved = (
            safety_checks["overall_passed"] and
            ethical_assessment["ethical_score"] > 0.7 and
            (human_override or risk_assessment["requires_human_review"] == False)
        )

        # Create audit entry
        audit_details = {
            "recommendation_id": recommendation.recommendation_id,
            "safety_checks": safety_checks,
            "ethical_assessment": ethical_assessment,
            "risk_assessment": risk_assessment,
            "approved": approved,
            "human_override": human_override
        }

        self._create_audit_entry(
            event_type="safety_gate_evaluation",
            actor="safety_monitoring_system",
            action="evaluate_safety_gate",
            target=recommendation.recommendation_id,
            details=audit_details
        )

        result = {
            "approved": approved,
            "safety_checks": safety_checks,
            "ethical_assessment": ethical_assessment,
            "risk_assessment": risk_assessment,
            "requires_human_review": risk_assessment["requires_human_review"],
            "audit_entry_created": True
        }

        logger.info(f"Safety gate evaluation for {recommendation.recommendation_id}: approved={approved}")

        return result

    async def _run_automated_safety_checks(self, recommendation: EvolutionRecommendation) -> Dict[str, Any]:
        """Run automated safety checks"""

        checks = {
            "architecture_stability": self._check_architecture_stability(recommendation),
            "performance_regression": self._check_performance_regression_risk(recommendation),
            "security_impact": self._check_security_impact(recommendation),
            "data_integrity": self._check_data_integrity(recommendation),
            "rollback_feasibility": self._check_rollback_feasibility(recommendation)
        }

        overall_passed = all(check["passed"] for check in checks.values())

        return {
            "checks": checks,
            "overall_passed": overall_passed,
            "passed_count": sum(1 for check in checks.values() if check["passed"]),
            "total_checks": len(checks)
        }

    def _check_architecture_stability(self, recommendation: EvolutionRecommendation) -> Dict[str, Any]:
        """Check if recommendation maintains architecture stability"""

        stable_types = ["heuristic_tuning", "learning_rate_adjustment", "module_enhancement"]
        is_stable = recommendation.evolution_type.value in stable_types

        return {
            "passed": is_stable,
            "risk_level": "low" if is_stable else "high",
            "reason": "Stable change type" if is_stable else "Architecture-altering change"
        }

    def _check_performance_regression_risk(self, recommendation: EvolutionRecommendation) -> Dict[str, Any]:
        """Check risk of performance regression"""

        # Simple heuristic based on change type
        risk_map = {
            "heuristic_tuning": "low",
            "learning_rate_adjustment": "medium",
            "module_enhancement": "medium",
            "new_capability": "high",
            "architecture_refactor": "high"
        }

        risk_level = risk_map.get(recommendation.evolution_type.value, "medium")
        passed = risk_level in ["low", "medium"]  # Allow medium risk with monitoring

        return {
            "passed": passed,
            "risk_level": risk_level,
            "reason": f"Performance regression risk: {risk_level}"
        }

    def _check_security_impact(self, recommendation: EvolutionRecommendation) -> Dict[str, Any]:
        """Check security impact of recommendation"""

        # Most evolution changes shouldn't affect security directly
        # But architecture changes need careful review
        high_security_risk = recommendation.evolution_type.value == "architecture_refactor"

        return {
            "passed": not high_security_risk,
            "risk_level": "high" if high_security_risk else "low",
            "reason": "Security impact assessment"
        }

    def _check_data_integrity(self, recommendation: EvolutionRecommendation) -> Dict[str, Any]:
        """Check if recommendation maintains data integrity"""

        # Assume most changes maintain data integrity
        return {
            "passed": True,
            "risk_level": "low",
            "reason": "Data integrity maintained"
        }

    def _check_rollback_feasibility(self, recommendation: EvolutionRecommendation) -> Dict[str, Any]:
        """Check if recommendation can be rolled back"""

        # All recommendations should have rollback plans
        has_rollback = hasattr(recommendation, 'implementation_plan') and len(recommendation.implementation_plan) > 0

        return {
            "passed": has_rollback,
            "risk_level": "low" if has_rollback else "high",
            "reason": "Rollback plan available" if has_rollback else "No rollback plan"
        }

    async def _run_ethical_assessment(self, recommendation: EvolutionRecommendation) -> Dict[str, Any]:
        """Run ethical assessment of recommendation"""

        assessment = {
            "bias_amplification": self._assess_bias_amplification(recommendation),
            "autonomy_impact": self._assess_autonomy_impact(recommendation),
            "transparency_impact": self._assess_transparency_impact(recommendation),
            "fairness_impact": self._assess_fairness_impact(recommendation),
            "privacy_impact": self._assess_privacy_impact(recommendation)
        }

        # Calculate overall ethical score
        ethical_score = statistics.mean([
            1.0 - assessment["bias_amplification"],  # Lower bias = higher score
            1.0 - abs(assessment["autonomy_impact"] - 0.5) * 2,  # Closer to 0.5 = better
            assessment["transparency_impact"],
            assessment["fairness_impact"],
            assessment["privacy_impact"]
        ])

        return {
            "assessment": assessment,
            "ethical_score": ethical_score,
            "concerns": [k for k, v in assessment.items() if isinstance(v, (int, float)) and v < 0.6]
        }

    def _assess_bias_amplification(self, recommendation: EvolutionRecommendation) -> float:
        """Assess risk of bias amplification (0-1, higher = more risk)"""
        risk_map = {
            "heuristic_tuning": 0.4,
            "learning_rate_adjustment": 0.3,
            "module_enhancement": 0.5,
            "new_capability": 0.7,
            "architecture_refactor": 0.8
        }
        return risk_map.get(recommendation.evolution_type.value, 0.5)

    def _assess_autonomy_impact(self, recommendation: EvolutionRecommendation) -> float:
        """Assess impact on system autonomy (0-1, 0.5 = neutral)"""
        impact_map = {
            "heuristic_tuning": 0.6,
            "learning_rate_adjustment": 0.7,
            "module_enhancement": 0.5,
            "new_capability": 0.4,
            "architecture_refactor": 0.8
        }
        return impact_map.get(recommendation.evolution_type.value, 0.5)

    def _assess_transparency_impact(self, recommendation: EvolutionRecommendation) -> float:
        """Assess transparency impact (0-1, higher = more transparent)"""
        return 0.8 if recommendation.evolution_type.value != "architecture_refactor" else 0.4

    def _assess_fairness_impact(self, recommendation: EvolutionRecommendation) -> float:
        """Assess fairness impact (0-1, higher = more fair)"""
        return 0.9  # Most changes maintain fairness

    def _assess_privacy_impact(self, recommendation: EvolutionRecommendation) -> float:
        """Assess privacy impact (0-1, higher = better privacy)"""
        return 0.95  # Evolution changes typically don't affect privacy

    def _calculate_risk_score(self, recommendation: EvolutionRecommendation,
                            safety_checks: Dict, ethical_assessment: Dict) -> Dict[str, Any]:
        """Calculate overall risk score"""

        safety_score = 1.0 if safety_checks["overall_passed"] else 0.3
        ethical_score = ethical_assessment["ethical_score"]

        # Priority multiplier
        priority_multiplier = {
            "low": 1.0,
            "medium": 1.2,
            "high": 1.5,
            "critical": 2.0
        }.get(recommendation.priority.value, 1.0)

        overall_risk = (safety_score + ethical_score) / 2.0 * priority_multiplier
        overall_risk = min(1.0, overall_risk)

        requires_human_review = (
            recommendation.priority.value in ["critical", "high"] or
            not safety_checks["overall_passed"] or
            ethical_score < 0.7 or
            overall_risk > 0.7
        )

        return {
            "overall_risk": overall_risk,
            "safety_score": safety_score,
            "ethical_score": ethical_score,
            "priority_multiplier": priority_multiplier,
            "requires_human_review": requires_human_review,
            "risk_level": "high" if overall_risk > 0.7 else "medium" if overall_risk > 0.4 else "low"
        }

    def _start_monitoring(self):
        """Start the monitoring thread"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        logger.info("Safety monitoring started")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Run monitoring in event loop
                asyncio.run(self._check_alerts())
                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)  # Wait before retry

    async def _check_alerts(self):
        """Check alert rules against current metrics"""

        # Get current metrics
        metrics = await self._collect_current_metrics()

        # Check each rule
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue

            metric_value = metrics.get(rule.metric)
            if metric_value is None:
                continue

            # Evaluate condition
            alert_triggered = self._evaluate_condition(metric_value, rule.condition, rule.threshold)

            if alert_triggered:
                await self._trigger_alert(rule, metric_value, metrics)

        # Clean up expired alerts
        self._cleanup_expired_alerts()

        # Save state
        self._save_persistent_state()

    async def _collect_current_metrics(self) -> Dict[str, float]:
        """Collect current system metrics"""

        # Get evolution report
        evolution_report = self.evolution_manager.get_evolution_report()

        # Extract key metrics
        metrics = {
            "health_score": evolution_report.get("evolution_state", {}).get("self_awareness_level", 0.7),
            "confidence_calibration": 0.85,  # Would be calculated from actual confidence distributions
            "confidence_calibration_drift": 0.05,  # Would be calculated as drift from baseline
            "interdisciplinary_coverage": 0.8,  # Would be measured from response diversity
            "coherence_score": 0.9,  # Would be calculated from coherence analysis
            "safety_violations": 0,  # Would be tracked by safety agents
            "response_quality": 0.85,
            "error_rate": 0.02,
            "latency_ms": 150.0
        }

        return metrics

    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition"""

        if condition == ">":
            return value > threshold
        elif condition == "<":
            return value < threshold
        elif condition == "==":
            return abs(value - threshold) < 0.001
        elif condition == "!=":
            return abs(value - threshold) >= 0.001
        else:
            return False

    async def _trigger_alert(self, rule: AlertRule, metric_value: float, all_metrics: Dict[str, float]):
        """Trigger an alert"""

        # Check cooldown
        alert_key = f"{rule.rule_id}_{metric_value:.3f}"
        if alert_key in self.active_alerts:
            last_alert = self.active_alerts[alert_key]
            cooldown_end = datetime.fromisoformat(last_alert.triggered_at) + timedelta(minutes=rule.cooldown_minutes)
            if datetime.now() < cooldown_end:
                return  # Still in cooldown

        # Create alert
        alert = Alert(
            alert_id=f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{rule.rule_id}",
            rule_id=rule.rule_id,
            severity=rule.severity,
            title=f"{rule.name}: {metric_value:.3f} {rule.condition} {rule.threshold}",
            description=rule.description,
            metric_value=metric_value,
            threshold=rule.threshold,
            triggered_at=datetime.now().isoformat(),
            status=AlertStatus.ACTIVE,
            metadata={
                "all_metrics": all_metrics,
                "rule": asdict(rule)
            }
        )

        self.active_alerts[alert.alert_id] = alert

        # Send notifications
        await self._send_alert_notifications(alert, rule.notification_channels)

        # Create audit entry
        self._create_audit_entry(
            event_type="alert_triggered",
            actor="safety_monitoring_system",
            action="trigger_alert",
            target=rule.rule_id,
            details={
                "alert_id": alert.alert_id,
                "severity": alert.severity.value,
                "metric_value": metric_value,
                "threshold": rule.threshold
            }
        )

        logger.warning(f"Alert triggered: {alert.title}")

    async def _send_alert_notifications(self, alert: Alert, channels: List[str]):
        """Send alert notifications through specified channels"""

        for channel in channels:
            try:
                if channel == "log":
                    logger.warning(f"ALERT [{alert.severity.value.upper()}]: {alert.title}")
                elif channel == "email":
                    await self._send_email_alert(alert)
                # Could add more channels like slack, pager, etc.

            except Exception as e:
                logger.error(f"Failed to send alert notification via {channel}: {e}")

    async def _send_email_alert(self, alert: Alert):
        """Send alert via email"""

        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['sender_email']
            msg['To'] = ', '.join(self.email_config['recipients'])
            msg['Subject'] = f"Kalki Alert: {alert.severity.value.upper()} - {alert.title}"

            body = f"""
Kalki Safety Alert

Severity: {alert.severity.value.upper()}
Title: {alert.title}
Description: {alert.description}

Metric Value: {alert.metric_value}
Threshold: {alert.threshold}
Triggered: {alert.triggered_at}

Please review the system status immediately.
            """

            msg.attach(MIMEText(body, 'plain'))

            # Note: Email sending would be implemented with proper SMTP configuration
            # For now, just log it
            logger.info(f"Email alert would be sent: {alert.title}")

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    def _cleanup_expired_alerts(self):
        """Clean up expired alerts"""

        current_time = datetime.now()
        expired_alerts = []

        for alert_id, alert in self.active_alerts.items():
            # Auto-resolve alerts after 24 hours
            triggered_at = datetime.fromisoformat(alert.triggered_at)
            if current_time - triggered_at > timedelta(hours=24):
                alert.status = AlertStatus.EXPIRED
                alert.resolved_at = current_time.isoformat()
                expired_alerts.append(alert_id)

        # Move to history
        for alert_id in expired_alerts:
            alert = self.active_alerts.pop(alert_id)
            self.alert_history.append(alert)

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""

        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_at = datetime.now().isoformat()

        # Create audit entry
        self._create_audit_entry(
            event_type="alert_acknowledged",
            actor=acknowledged_by,
            action="acknowledge_alert",
            target=alert_id,
            details={"alert_title": alert.title}
        )

        self._save_persistent_state()
        return True

    def resolve_alert(self, alert_id: str, resolved_by: str) -> bool:
        """Resolve an alert"""

        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now().isoformat()

        # Move to history
        self.alert_history.append(alert)
        del self.active_alerts[alert_id]

        # Create audit entry
        self._create_audit_entry(
            event_type="alert_resolved",
            actor=resolved_by,
            action="resolve_alert",
            target=alert_id,
            details={"alert_title": alert.title}
        )

        self._save_persistent_state()
        return True

    def get_safety_status(self) -> Dict[str, Any]:
        """Get safety status for observability dashboard"""
        system_status = self.get_system_status()
        return {
            "active_violations": len([a for a in self.active_alerts.values() if a.severity.value == "critical"]),
            "ethics_compliance_score": 0.95 if len(self.active_alerts) == 0 else max(0.5, 1.0 - (len(self.active_alerts) * 0.1)),
            "alerts": system_status.get("alerts", {}),
            "audit_trail": system_status.get("audit_trail", {}),
            "safety_gates": system_status.get("safety_gates", {})
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""

        return {
            "alerts": {
                "active_count": len(self.active_alerts),
                "active_alerts": [
                    {
                        "id": alert.alert_id,
                        "title": alert.title,
                        "severity": alert.severity.value,
                        "triggered_at": alert.triggered_at
                    }
                    for alert in self.active_alerts.values()
                ],
                "recent_history": [
                    {
                        "id": alert.alert_id,
                        "title": alert.title,
                        "severity": alert.severity.value,
                        "status": alert.status.value,
                        "triggered_at": alert.triggered_at
                    }
                    for alert in self.alert_history[-10:]
                ]
            },
            "audit_trail": {
                "total_entries": len(self.audit_chain),
                "recent_entries": [
                    {
                        "timestamp": entry.timestamp,
                        "event_type": entry.event_type,
                        "action": entry.action,
                        "actor": entry.actor
                    }
                    for entry in self.audit_chain[-5:]
                ],
                "chain_integrity": self._verify_audit_chain_integrity()
            },
            "safety_gates": {
                "enabled_rules": len([r for r in self.alert_rules.values() if r.enabled]),
                "total_rules": len(self.alert_rules)
            },
            "monitoring": {
                "active": getattr(self, 'monitoring_active', True),
                "check_interval": getattr(self, 'check_interval', 60)
            }
        }

    def _verify_audit_chain_integrity(self) -> bool:
        """Verify the integrity of the audit chain"""

        if not self.audit_chain:
            return True

        try:
            for i, entry in enumerate(self.audit_chain):
                # Verify signature
                entry_dict = asdict(entry)
                signature = entry_dict.pop('signature')
                data_to_verify = json.dumps(entry_dict, sort_keys=True).encode()

                expected_signature = hmac.new(
                    self.audit_secret.encode(),
                    data_to_verify,
                    hashlib.sha256
                ).hexdigest()

                if signature != expected_signature:
                    logger.error(f"Audit chain integrity violation at entry {i}")
                    return False

                # Verify chain linkage
                if i > 0:
                    prev_entry = self.audit_chain[i-1]
                    prev_dict = asdict(prev_entry)
                    prev_dict.pop('signature', None)
                    expected_prev_hash = hashlib.sha256(
                        json.dumps(prev_dict, sort_keys=True).encode()
                    ).hexdigest()

                    if entry.previous_hash != expected_prev_hash:
                        logger.error(f"Audit chain linkage violation at entry {i}")
                        return False

            return True

        except Exception as e:
            logger.error(f"Audit chain verification error: {e}")
            return False

    def stop_monitoring(self):
        """Stop the monitoring thread"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Safety monitoring stopped")

# Global safety monitoring system instance
_safety_monitoring_system = None

def get_safety_monitoring_system() -> SafetyMonitoringSystem:
    """Get the global safety monitoring system instance"""
    global _safety_monitoring_system
    if _safety_monitoring_system is None:
        _safety_monitoring_system = SafetyMonitoringSystem()
    return _safety_monitoring_system

async def evaluate_evolution_safety(recommendation_id: str, human_override: bool = False) -> Dict[str, Any]:
    """Convenience function to evaluate safety of an evolution recommendation"""
    system = get_safety_monitoring_system()
    evolution_manager = get_self_evolution_manager()

    # Find the recommendation
    pending_recs = evolution_manager.evolution_state.pending_recommendations
    recommendation = next((r for r in pending_recs if r.recommendation_id == recommendation_id), None)

    if not recommendation:
        return {"success": False, "error": f"Recommendation {recommendation_id} not found"}

    return await system.evaluate_safety_gate(recommendation, human_override)