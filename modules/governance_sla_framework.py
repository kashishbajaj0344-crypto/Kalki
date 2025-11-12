# ============================================================
# Kalki v2.4 — governance_sla_framework.py
# ------------------------------------------------------------
# Governance & SLA Framework: Formal Change Management
# - Service Level Agreements for system performance and availability
# - Change management procedures and approval workflows
# - Governance committees and decision-making processes
# - Risk management and incident response frameworks
# - Performance guarantees and penalty clauses
# ============================================================

import os
import json
import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import statistics

from modules.utils.logging_config import get_logger
from modules.self_evolution_manager import get_self_evolution_manager
from modules.safety_monitoring_system import get_safety_monitoring_system
from modules.production_observability_dashboard import get_production_observability_dashboard

logger = get_logger("Kalki.GovernanceSLA")

class SLAMetric(Enum):
    """SLA metric types"""
    AVAILABILITY = "availability"              # System uptime percentage
    RESPONSE_TIME = "response_time"            # Response time percentiles
    ERROR_RATE = "error_rate"                  # Error rate percentage
    DATA_ACCURACY = "data_accuracy"            # Data accuracy percentage
    SECURITY_INCIDENTS = "security_incidents"  # Security incidents per month
    COMPLIANCE_VIOLATIONS = "compliance_violations"  # Compliance violations

class ChangeType(Enum):
    """Types of system changes"""
    EVOLUTION_RECOMMENDATION = "evolution_recommendation"  # AI self-improvement
    SECURITY_PATCH = "security_patch"                     # Security updates
    PERFORMANCE_OPTIMIZATION = "performance_optimization" # Performance improvements
    FEATURE_ADDITION = "feature_addition"                # New features
    CONFIGURATION_CHANGE = "configuration_change"        # Configuration updates
    INFRASTRUCTURE_CHANGE = "infrastructure_change"      # Infrastructure changes

class ApprovalStatus(Enum):
    """Change approval status"""
    PENDING_REVIEW = "pending_review"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    CONDITIONAL_APPROVAL = "conditional_approval"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    IMPLEMENTED = "implemented"
    ROLLED_BACK = "rolled_back"

class GovernanceRole(Enum):
    """Governance committee roles"""
    EXECUTIVE_SPONSOR = "executive_sponsor"        # Executive oversight
    TECHNICAL_LEAD = "technical_lead"             # Technical decision making
    SECURITY_OFFICER = "security_officer"         # Security approval
    COMPLIANCE_OFFICER = "compliance_officer"     # Compliance oversight
    ETHICS_REVIEWER = "ethics_reviewer"           # Ethical considerations
    RISK_MANAGER = "risk_manager"                 # Risk assessment
    BUSINESS_STAKEHOLDER = "business_stakeholder" # Business impact assessment

@dataclass
class ServiceLevelAgreement:
    """Service Level Agreement definition"""
    sla_id: str
    name: str
    description: str
    metric: SLAMetric
    target_value: float
    measurement_period: str  # "monthly", "quarterly", "yearly"
    penalty_clause: str
    warning_threshold: float
    critical_threshold: float
    status: str = "active"  # "active", "suspended", "terminated"
    last_assessment: Optional[str] = None
    compliance_history: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class ChangeRequest:
    """Change request for system modification"""
    change_id: str
    title: str
    description: str
    change_type: ChangeType
    requester: str
    priority: str  # "low", "medium", "high", "critical"
    impact_assessment: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    rollback_plan: str
    testing_requirements: List[str]
    approval_workflow: List[GovernanceRole]
    status: ApprovalStatus
    created_date: str
    required_approvals: Dict[GovernanceRole, bool] = field(default_factory=dict)
    approvals_received: Dict[GovernanceRole, Dict[str, Any]] = field(default_factory=dict)
    implementation_date: Optional[str] = None
    completion_date: Optional[str] = None
    post_implementation_review: Optional[Dict[str, Any]] = None

@dataclass
class GovernanceCommittee:
    """Governance committee definition"""
    committee_id: str
    name: str
    description: str
    members: Dict[GovernanceRole, List[str]]  # role -> list of member emails
    meeting_frequency: str  # "weekly", "monthly", "quarterly"
    quorum_requirement: int
    decision_authority: List[str]
    active: bool = True
    last_meeting: Optional[str] = None
    meeting_history: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class IncidentReport:
    """Incident report for governance tracking"""
    incident_id: str
    title: str
    description: str
    severity: str  # "low", "medium", "high", "critical"
    category: str  # "security", "performance", "availability", "data", "compliance"
    detection_time: str
    response_time: str
    impact_assessment: Dict[str, Any]
    root_cause: str
    corrective_actions: List[str]
    preventive_measures: List[str]
    resolution_time: Optional[str] = None
    sla_breach: bool = False
    cost_impact: Optional[float] = None
    lessons_learned: List[str] = field(default_factory=list)

@dataclass
class GovernanceDecision:
    """Governance committee decision"""
    decision_id: str
    committee_id: str
    decision_type: str  # "approval", "rejection", "escalation", "policy_change"
    decision_summary: str
    rationale: str
    decision_date: str
    change_request_id: Optional[str] = None
    incident_id: Optional[str] = None
    conditions: List[str] = field(default_factory=list)
    voting_results: Dict[str, str] = field(default_factory=dict)  # member -> vote
    implementation_deadline: Optional[str] = None

class GovernanceSLAFramework:
    """
    Governance & SLA Framework: Formal Change Management

    Implements comprehensive governance procedures, service level agreements,
    change management workflows, and incident response for the Kalki system.
    """

    def __init__(self):
        self.evolution_manager = get_self_evolution_manager()
        self.safety_monitor = get_safety_monitoring_system()
        self.observability = get_production_observability_dashboard()

        # SLA management
        self.service_level_agreements: Dict[str, ServiceLevelAgreement] = {}

        # Change management
        self.change_requests: Dict[str, ChangeRequest] = {}
        self.approval_workflows: Dict[ChangeType, List[GovernanceRole]] = {}

        # Governance committees
        self.governance_committees: Dict[str, GovernanceCommittee] = {}
        self.governance_decisions: Dict[str, GovernanceDecision] = {}

        # Incident management
        self.incident_reports: Dict[str, IncidentReport] = {}

        # Configuration
        self.sla_assessment_frequency = "monthly"
        self.change_approval_timeout_days = 14
        self.incident_response_sla_hours = {
            "critical": 1,
            "high": 4,
            "medium": 24,
            "low": 72
        }

        # Email configuration
        self.email_config = {
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "sender_email": "governance@kalki.ai",
            "sender_password": os.getenv("KALKI_GOVERNANCE_EMAIL_PASSWORD", "")
        }

        # Persistence
        self.data_dir = "data/governance"
        self.sla_file = f"{self.data_dir}/service_level_agreements.json"
        self.changes_file = f"{self.data_dir}/change_requests.json"
        self.committees_file = f"{self.data_dir}/governance_committees.json"
        self.incidents_file = f"{self.data_dir}/incident_reports.json"

        # Initialize
        self._initialize_governance_framework()

        logger.info("Governance & SLA Framework initialized")

    def _initialize_governance_framework(self):
        """Initialize the governance and SLA framework"""

        # Load existing data
        self._load_governance_data()

        # Set up default SLAs
        self._setup_default_slas()

        # Set up approval workflows
        self._setup_approval_workflows()

        # Set up governance committees
        self._setup_governance_committees()

    def _setup_default_slas(self):
        """Set up default Service Level Agreements"""

        default_slas = [
            ServiceLevelAgreement(
                sla_id="system_availability",
                name="System Availability",
                description="System must maintain 99.9% uptime",
                metric=SLAMetric.AVAILABILITY,
                target_value=0.999,  # 99.9%
                measurement_period="monthly",
                penalty_clause="Financial penalty of $10,000 per 0.1% below target",
                warning_threshold=0.995,
                critical_threshold=0.990
            ),
            ServiceLevelAgreement(
                sla_id="response_time_p95",
                name="Response Time P95",
                description="95th percentile response time must be under 2 seconds",
                metric=SLAMetric.RESPONSE_TIME,
                target_value=2.0,  # seconds
                measurement_period="monthly",
                penalty_clause="Performance bonus reduction of 5% per 0.5s over target",
                warning_threshold=2.5,
                critical_threshold=3.0
            ),
            ServiceLevelAgreement(
                sla_id="error_rate",
                name="Error Rate",
                description="System error rate must be below 0.1%",
                metric=SLAMetric.ERROR_RATE,
                target_value=0.001,  # 0.1%
                measurement_period="monthly",
                penalty_clause="Service credit of 10% for each 0.05% over target",
                warning_threshold=0.005,
                critical_threshold=0.010
            ),
            ServiceLevelAgreement(
                sla_id="security_incidents",
                name="Security Incidents",
                description="Maximum 2 security incidents per quarter",
                metric=SLAMetric.SECURITY_INCIDENTS,
                target_value=2.0,
                measurement_period="quarterly",
                penalty_clause="Security audit requirement and potential contract termination",
                warning_threshold=1.0,
                critical_threshold=3.0
            )
        ]

        for sla in default_slas:
            self.service_level_agreements[sla.sla_id] = sla

    def _setup_approval_workflows(self):
        """Set up change approval workflows"""

        self.approval_workflows = {
            ChangeType.EVOLUTION_RECOMMENDATION: [
                GovernanceRole.ETHICS_REVIEWER,
                GovernanceRole.SECURITY_OFFICER,
                GovernanceRole.TECHNICAL_LEAD,
                GovernanceRole.EXECUTIVE_SPONSOR
            ],
            ChangeType.SECURITY_PATCH: [
                GovernanceRole.SECURITY_OFFICER,
                GovernanceRole.TECHNICAL_LEAD
            ],
            ChangeType.PERFORMANCE_OPTIMIZATION: [
                GovernanceRole.TECHNICAL_LEAD,
                GovernanceRole.RISK_MANAGER
            ],
            ChangeType.FEATURE_ADDITION: [
                GovernanceRole.TECHNICAL_LEAD,
                GovernanceRole.BUSINESS_STAKEHOLDER,
                GovernanceRole.SECURITY_OFFICER
            ],
            ChangeType.CONFIGURATION_CHANGE: [
                GovernanceRole.TECHNICAL_LEAD,
                GovernanceRole.RISK_MANAGER
            ],
            ChangeType.INFRASTRUCTURE_CHANGE: [
                GovernanceRole.TECHNICAL_LEAD,
                GovernanceRole.SECURITY_OFFICER,
                GovernanceRole.EXECUTIVE_SPONSOR
            ]
        }

    def _setup_governance_committees(self):
        """Set up governance committees"""

        committees = [
            {
                "committee_id": "executive_oversight",
                "name": "Executive Oversight Committee",
                "description": "High-level governance and strategic decisions",
                "members": {
                    GovernanceRole.EXECUTIVE_SPONSOR: ["ceo@kalki.ai", "cfo@kalki.ai"],
                    GovernanceRole.RISK_MANAGER: ["risk@kalki.ai"]
                },
                "meeting_frequency": "monthly",
                "quorum_requirement": 2,
                "decision_authority": ["Strategic changes", "Budget approvals", "Risk acceptance"]
            },
            {
                "committee_id": "technical_governance",
                "name": "Technical Governance Board",
                "description": "Technical decisions and architecture oversight",
                "members": {
                    GovernanceRole.TECHNICAL_LEAD: ["techlead@kalki.ai"],
                    GovernanceRole.SECURITY_OFFICER: ["security@kalki.ai"],
                    GovernanceRole.ETHICS_REVIEWER: ["ethics@kalki.ai"]
                },
                "meeting_frequency": "weekly",
                "quorum_requirement": 2,
                "decision_authority": ["Technical changes", "Architecture decisions", "Security policies"]
            },
            {
                "committee_id": "change_control",
                "name": "Change Control Board",
                "description": "Review and approval of system changes",
                "members": {
                    GovernanceRole.TECHNICAL_LEAD: ["techlead@kalki.ai"],
                    GovernanceRole.BUSINESS_STAKEHOLDER: ["product@kalki.ai"],
                    GovernanceRole.COMPLIANCE_OFFICER: ["compliance@kalki.ai"]
                },
                "meeting_frequency": "weekly",
                "quorum_requirement": 2,
                "decision_authority": ["Change approvals", "Deployment authorizations", "Rollback decisions"]
            }
        ]

        for committee_data in committees:
            committee = GovernanceCommittee(**committee_data)
            self.governance_committees[committee.committee_id] = committee

    def submit_change_request(self, title: str, description: str, change_type: ChangeType,
                            requester: str, priority: str, impact_assessment: Dict[str, Any],
                            risk_assessment: Dict[str, Any], rollback_plan: str,
                            testing_requirements: List[str]) -> str:
        """
        Submit a change request for approval

        Returns:
            Change request ID
        """

        change_id = f"change_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(title.encode()).hexdigest()[:8]}"

        # Get approval workflow for this change type
        approval_workflow = self.approval_workflows.get(change_type, [
            GovernanceRole.TECHNICAL_LEAD,
            GovernanceRole.SECURITY_OFFICER
        ])

        change_request = ChangeRequest(
            change_id=change_id,
            title=title,
            description=description,
            change_type=change_type,
            requester=requester,
            priority=priority,
            impact_assessment=impact_assessment,
            risk_assessment=risk_assessment,
            rollback_plan=rollback_plan,
            testing_requirements=testing_requirements,
            approval_workflow=approval_workflow,
            status=ApprovalStatus.PENDING_REVIEW,
            created_date=datetime.now().isoformat(),
            required_approvals={role: False for role in approval_workflow}
        )

        self.change_requests[change_id] = change_request

        # Send notifications
        asyncio.create_task(self._notify_change_request_submitted(change_request))

        logger.info(f"Change request submitted: {change_id} - {title}")

        return change_id

    def approve_change_request(self, change_id: str, approver_role: GovernanceRole,
                             approver_email: str, decision: str, comments: str = "") -> bool:
        """
        Approve or reject a change request

        Returns:
            True if approval recorded successfully
        """

        if change_id not in self.change_requests:
            logger.error(f"Change request {change_id} not found")
            return False

        change_request = self.change_requests[change_id]

        if approver_role not in change_request.approval_workflow:
            logger.error(f"Role {approver_role.value} not required for change {change_id}")
            return False

        # Record approval
        approval_data = {
            "approver": approver_email,
            "decision": decision,
            "comments": comments,
            "timestamp": datetime.now().isoformat()
        }

        change_request.approvals_received[approver_role] = approval_data

        if decision.lower() in ["approved", "approve"]:
            change_request.required_approvals[approver_role] = True
        else:
            change_request.status = ApprovalStatus.REJECTED

        # Check if all approvals received
        self._update_change_status(change_request)

        # Send notifications
        asyncio.create_task(self._notify_change_decision(change_request, approver_role, approval_data))

        logger.info(f"Change {change_id} {decision} by {approver_role.value}")

        return True

    def _update_change_status(self, change_request: ChangeRequest):
        """Update change request status based on approvals"""

        # Check if all required approvals received
        all_approved = all(change_request.required_approvals.values())

        if all_approved:
            change_request.status = ApprovalStatus.APPROVED
        elif any(not approved for approved in change_request.required_approvals.values()):
            # Check if any rejection
            for role, approval in change_request.approvals_received.items():
                if approval["decision"].lower() in ["rejected", "reject"]:
                    change_request.status = ApprovalStatus.REJECTED
                    break
            else:
                change_request.status = ApprovalStatus.UNDER_REVIEW

    def report_incident(self, title: str, description: str, severity: str,
                       category: str, impact_assessment: Dict[str, Any]) -> str:
        """
        Report a system incident

        Returns:
            Incident ID
        """

        incident_id = f"incident_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(title.encode()).hexdigest()[:8]}"

        incident = IncidentReport(
            incident_id=incident_id,
            title=title,
            description=description,
            severity=severity,
            category=category,
            detection_time=datetime.now().isoformat(),
            response_time=datetime.now().isoformat(),  # Immediate response
            impact_assessment=impact_assessment,
            root_cause="To be determined",
            corrective_actions=[],
            preventive_measures=[]
        )

        self.incident_reports[incident_id] = incident

        # Check SLA breach
        incident.sla_breach = self._check_sla_breach(incident)

        # Send notifications
        asyncio.create_task(self._notify_incident_reported(incident))

        logger.info(f"Incident reported: {incident_id} - {severity} severity")

        return incident_id

    def _check_sla_breach(self, incident: IncidentReport) -> bool:
        """Check if incident constitutes an SLA breach"""

        # Simple SLA breach detection based on incident characteristics
        if incident.severity == "critical" and incident.category in ["availability", "security"]:
            return True

        if incident.severity == "high" and incident.category == "availability":
            # Check downtime duration
            if "downtime_minutes" in incident.impact_assessment:
                return incident.impact_assessment["downtime_minutes"] > 60

        return False

    def assess_sla_compliance(self) -> Dict[str, Any]:
        """Assess SLA compliance across all agreements"""

        compliance_report = {
            "assessment_date": datetime.now().isoformat(),
            "period": "Last 30 days",
            "overall_compliance_score": 0.0,
            "sla_status": {},
            "breaches": [],
            "warnings": [],
            "recommendations": []
        }

        total_slas = len(self.service_level_agreements)
        compliant_slas = 0

        for sla in self.service_level_agreements.values():
            if sla.status != "active":
                continue

            # Get current metric value (placeholder - would integrate with observability)
            current_value = self._get_current_sla_metric(sla)

            # Assess compliance
            status = "compliant"
            if current_value <= sla.critical_threshold:
                status = "breach"
                compliance_report["breaches"].append({
                    "sla_id": sla.sla_id,
                    "name": sla.name,
                    "target": sla.target_value,
                    "actual": current_value,
                    "penalty": sla.penalty_clause
                })
            elif current_value <= sla.warning_threshold:
                status = "warning"
                compliance_report["warnings"].append({
                    "sla_id": sla.sla_id,
                    "name": sla.name,
                    "target": sla.target_value,
                    "actual": current_value
                })

            if status == "compliant":
                compliant_slas += 1

            compliance_report["sla_status"][sla.sla_id] = {
                "name": sla.name,
                "target": sla.target_value,
                "current": current_value,
                "status": status
            }

        compliance_report["overall_compliance_score"] = compliant_slas / total_slas if total_slas > 0 else 0

        # Generate recommendations
        compliance_report["recommendations"] = self._generate_sla_recommendations(compliance_report)

        return compliance_report

    def _get_current_sla_metric(self, sla: ServiceLevelAgreement) -> float:
        """Get current value for SLA metric (placeholder)"""

        # This would integrate with the observability dashboard
        # For now, return simulated values

        metric_simulations = {
            "system_availability": 0.998,  # 99.8% (slightly below 99.9% target)
            "response_time_p95": 2.2,      # 2.2s (slightly over 2.0s target)
            "error_rate": 0.0005,          # 0.05% (well below 0.1% target)
            "security_incidents": 1.0      # 1 incident (below 2 target)
        }

        return metric_simulations.get(sla.sla_id, sla.target_value * 0.95)

    def _generate_sla_recommendations(self, compliance_report: Dict[str, Any]) -> List[str]:
        """Generate SLA compliance recommendations"""

        recommendations = []

        if compliance_report["breaches"]:
            recommendations.append(f"Address {len(compliance_report['breaches'])} SLA breaches immediately")

        if compliance_report["warnings"]:
            recommendations.append(f"Review {len(compliance_report['warnings'])} SLA warnings")

        if compliance_report["overall_compliance_score"] < 0.9:
            recommendations.append("Implement comprehensive SLA improvement plan")

        recommendations.extend([
            "Regular SLA performance monitoring and reporting",
            "Automated alerting for SLA threshold breaches",
            "Root cause analysis for SLA violations",
            "Continuous improvement of system performance"
        ])

        return recommendations

    async def _notify_change_request_submitted(self, change_request: ChangeRequest):
        """Notify stakeholders of change request submission"""

        if not self.email_config["sender_password"]:
            return

        subject = f"Kalki Change Request: {change_request.title}"

        # Notify all required approvers
        recipients = set()
        for role in change_request.approval_workflow:
            for committee in self.governance_committees.values():
                if role in committee.members:
                    recipients.update(committee.members[role])

        body = f"""
Change Request Submitted

ID: {change_request.change_id}
Title: {change_request.title}
Type: {change_request.change_type.value.replace('_', ' ').title()}
Priority: {change_request.priority.upper()}
Requester: {change_request.requester}

Description:
{change_request.description}

Impact Assessment:
{json.dumps(change_request.impact_assessment, indent=2)}

Risk Assessment:
{json.dumps(change_request.risk_assessment, indent=2)}

Rollback Plan:
{change_request.rollback_plan}

Testing Requirements:
{chr(10).join(f"- {req}" for req in change_request.testing_requirements)}

Please review and provide approval decision within {self.change_approval_timeout_days} days.

Best regards,
Kalki Governance System
"""

        for email in list(recipients):
            await self._send_email(email, subject, body)

    async def _notify_change_decision(self, change_request: ChangeRequest,
                                    approver_role: GovernanceRole, approval_data: Dict[str, Any]):
        """Notify of change decision"""

        if not self.email_config["sender_password"]:
            return

        subject = f"Kalki Change Decision: {change_request.title}"

        # Notify requester and all approvers
        recipients = {change_request.requester}

        for committee in self.governance_committees.values():
            for role_members in committee.members.values():
                recipients.update(role_members)

        decision_emoji = "✅" if approval_data["decision"].lower() in ["approved", "approve"] else "❌"

        body = f"""
{decision_emoji} Change Request Decision

Change: {change_request.title}
Approver: {approver_role.value.replace('_', ' ').title()}
Decision: {approval_data['decision'].upper()}

Comments:
{approval_data.get('comments', 'No comments provided')}

Current Status: {change_request.status.value.replace('_', ' ').title()}

Required Approvals:
{chr(10).join(f"- {role.value.replace('_', ' ').title()}: {'✅' if approved else '⏳'}" for role, approved in change_request.required_approvals.items())}

Best regards,
Kalki Governance System
"""

        for email in list(recipients):
            await self._send_email(email, subject, body)

    async def _notify_incident_reported(self, incident: IncidentReport):
        """Notify of incident report"""

        if not self.email_config["sender_password"]:
            return

        subject = f"🚨 Kalki Incident Report: {incident.title}"

        # Notify incident response team and executives
        recipients = ["incident@kalki.ai", "executives@kalki.ai", "security@kalki.ai"]

        severity_emoji = {
            "critical": "🚨",
            "high": "⚠️",
            "medium": "⚡",
            "low": "ℹ️"
        }.get(incident.severity, "❓")

        sla_indicator = " (SLA BREACH)" if incident.sla_breach else ""

        body = f"""
{severity_emoji} INCIDENT REPORT {sla_indicator}

ID: {incident.incident_id}
Title: {incident.title}
Severity: {incident.severity.upper()}
Category: {incident.category}

Description:
{incident.description}

Detection Time: {incident.detection_time}
Response Time: {incident.response_time}

Impact Assessment:
{json.dumps(incident.impact_assessment, indent=2)}

Immediate actions required. Incident response team has been notified.

Best regards,
Kalki Monitoring System
"""

        for email in recipients:
            await self._send_email(email, subject, body)

    async def _send_email(self, recipient: str, subject: str, body: str):
        """Send email notification"""

        if not self.email_config["sender_password"]:
            logger.warning("Email password not configured, skipping email send")
            return

        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config["sender_email"]
            msg['To'] = recipient
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(self.email_config["smtp_server"], self.email_config["smtp_port"])
            server.starttls()
            server.login(self.email_config["sender_email"], self.email_config["sender_password"])
            text = msg.as_string()
            server.sendmail(self.email_config["sender_email"], recipient, text)
            server.quit()

        except Exception as e:
            logger.error(f"Failed to send email to {recipient}: {e}")

    def get_governance_status(self) -> Dict[str, Any]:
        """Get overall governance status"""

        pending_changes = len([c for c in self.change_requests.values() if c.status in [ApprovalStatus.PENDING_REVIEW, ApprovalStatus.UNDER_REVIEW]])
        approved_changes = len([c for c in self.change_requests.values() if c.status == ApprovalStatus.APPROVED])
        rejected_changes = len([c for c in self.change_requests.values() if c.status == ApprovalStatus.REJECTED])

        open_incidents = len([i for i in self.incident_reports.values() if not i.resolution_time])
        sla_compliance = self.assess_sla_compliance()

        return {
            "pending_change_requests": pending_changes,
            "approved_changes": approved_changes,
            "rejected_changes": rejected_changes,
            "open_incidents": open_incidents,
            "sla_compliance_score": sla_compliance["overall_compliance_score"],
            "active_slas": len([s for s in self.service_level_agreements.values() if s.status == "active"]),
            "active_committees": len([c for c in self.governance_committees.values() if c.active]),
            "last_updated": datetime.now().isoformat()
        }

    def _load_governance_data(self):
        """Load governance data from files"""
        try:
            os.makedirs(self.data_dir, exist_ok=True)

            files_data = [
                (self.sla_file, self.service_level_agreements, ServiceLevelAgreement),
                (self.changes_file, self.change_requests, ChangeRequest),
                (self.committees_file, self.governance_committees, GovernanceCommittee),
                (self.incidents_file, self.incident_reports, IncidentReport)
            ]

            for file_path, data_dict, data_class in files_data:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        items = json.load(f)
                        for item_data in items.values():
                            item = data_class(**item_data)
                            data_dict[item_data[list(item_data.keys())[0]]] = item

        except Exception as e:
            logger.warning(f"Failed to load governance data: {e}")

    def _save_governance_data(self):
        """Save governance data to files"""
        try:
            files_data = [
                (self.sla_file, self.service_level_agreements),
                (self.changes_file, self.change_requests),
                (self.committees_file, self.governance_committees),
                (self.incidents_file, self.incident_reports)
            ]

            for file_path, data_dict in files_data:
                data = {k: asdict(v) for k, v in data_dict.items()}
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save governance data: {e}")

# Global governance and SLA framework instance
_governance_sla_framework = None

def get_governance_sla_framework() -> GovernanceSLAFramework:
    """Get the global governance and SLA framework instance"""
    global _governance_sla_framework
    if _governance_sla_framework is None:
        _governance_sla_framework = GovernanceSLAFramework()
    return _governance_sla_framework

# Convenience functions
def submit_evolution_change_request(recommendation_data: Dict[str, Any]) -> str:
    """Submit an evolution recommendation as a change request"""
    governance = get_governance_sla_framework()

    return governance.submit_change_request(
        title=f"Evolution: {recommendation_data.get('type', 'System Improvement')}",
        description=recommendation_data.get('description', 'AI system self-improvement recommendation'),
        change_type=ChangeType.EVOLUTION_RECOMMENDATION,
        requester="Kalki AI System",
        priority="high" if recommendation_data.get('risk_level') == 'high' else "medium",
        impact_assessment={
            "technical_impact": "System architecture modification",
            "performance_impact": recommendation_data.get('performance_impact', 'Unknown'),
            "security_impact": recommendation_data.get('security_impact', 'To be assessed'),
            "user_impact": "Potential behavior changes"
        },
        risk_assessment={
            "risk_level": recommendation_data.get('risk_level', 'medium'),
            "failure_probability": recommendation_data.get('failure_probability', 0.2),
            "impact_if_failed": "System instability or degraded performance"
        },
        rollback_plan="Revert to previous system state using backup checkpoints",
        testing_requirements=[
            "Unit tests for modified components",
            "Integration testing with existing systems",
            "Performance regression testing",
            "Safety validation testing"
        ]
    )

def report_system_incident(title: str, description: str, severity: str, category: str) -> str:
    """Report a system incident"""
    governance = get_governance_sla_framework()

    return governance.report_incident(
        title=title,
        description=description,
        severity=severity,
        category=category,
        impact_assessment={
            "affected_users": "Internal system operations",
            "downtime_minutes": 0,  # Would be calculated
            "data_loss": "None",
            "financial_impact": "To be assessed"
        }
    )