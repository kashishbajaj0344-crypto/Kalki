# ============================================================
# Kalki v2.4 — external_red_teaming_certification.py
# ------------------------------------------------------------
# External Red Teaming & Certification: Independent Security Audits
# - Automated red teaming simulation and coordination
# - Third-party certification management and tracking
# - External audit scheduling and compliance monitoring
# - Vulnerability disclosure program and bug bounty coordination
# - Regulatory compliance assessment and reporting
# ============================================================

import os
import json
import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import requests
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import hmac
import uuid

from modules.utils.logging_config import get_logger
from modules.self_evolution_manager import get_self_evolution_manager
from modules.safety_monitoring_system import get_safety_monitoring_system
from modules.simulated_adversarial_tests import get_simulated_adversarial_tests

logger = get_logger("Kalki.RedTeamingCertification")

class AuditType(Enum):
    """Types of external audits"""
    SECURITY_PENETRATION = "security_penetration"    # Penetration testing
    ETHICS_REVIEW = "ethics_review"                # Ethical AI assessment
    SAFETY_CERTIFICATION = "safety_certification"   # Safety standards compliance
    PERFORMANCE_AUDIT = "performance_audit"         # Performance benchmarking
    CODE_REVIEW = "code_review"                     # External code review
    COMPLIANCE_ASSESSMENT = "compliance_assessment" # Regulatory compliance

class CertificationStandard(Enum):
    """Certification standards and frameworks"""
    ISO_IEC_27001 = "ISO/IEC 27001"          # Information security management
    NIST_AI_RMF = "NIST AI RMF"              # AI Risk Management Framework
    IEEE_7001 = "IEEE 7001"                  # Transparency of autonomous systems
    EU_AI_ACT = "EU AI Act"                  # European AI regulation
    SOC_2 = "SOC 2"                         # Trust services criteria
    ISO_IEC_42001 = "ISO/IEC 42001"          # AI management systems

class RedTeamEngagement(Enum):
    """Red teaming engagement types"""
    SIMULATED_ATTACK = "simulated_attack"        # Automated attack simulation
    HUMAN_RED_TEAM = "human_red_team"           # Professional red team engagement
    BUG_BOUNTY = "bug_bounty"                   # Public bug bounty program
    ETHICS_CHALLENGE = "ethics_challenge"       # Ethical boundary testing
    ADVERSARIAL_ML = "adversarial_ml"           # ML-specific attacks

@dataclass
class ExternalAudit:
    """External audit engagement"""
    audit_id: str
    title: str
    description: str
    audit_type: AuditType
    auditor_organization: str
    auditor_contact: Dict[str, str]
    scope: List[str]
    start_date: str
    end_date: str
    status: str  # "scheduled", "in_progress", "completed", "failed"
    deliverables: List[str] = field(default_factory=list)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    compliance_score: Optional[float] = None
    report_url: Optional[str] = None
    next_audit_date: Optional[str] = None

@dataclass
class Certification:
    """Certification tracking"""
    certification_id: str
    standard: CertificationStandard
    issuing_body: str
    issue_date: str
    expiry_date: str
    status: str  # "active", "expired", "revoked", "pending_renewal"
    scope: str
    certificate_url: Optional[str] = None
    audit_history: List[str] = field(default_factory=list)  # audit_ids

@dataclass
class RedTeamExercise:
    """Red teaming exercise"""
    exercise_id: str
    name: str
    engagement_type: RedTeamEngagement
    red_team_organization: str
    objectives: List[str]
    rules_of_engagement: Dict[str, Any]
    start_date: str
    end_date: str
    status: str  # "planned", "active", "completed", "cancelled"
    findings: List[Dict[str, Any]] = field(default_factory=list)
    impact_assessment: Dict[str, Any] = field(default_factory=dict)
    lessons_learned: List[str] = field(default_factory=list)

@dataclass
class VulnerabilityReport:
    """Vulnerability disclosure report"""
    report_id: str
    submitter: str
    submission_date: str
    vulnerability_type: str
    severity: str  # "critical", "high", "medium", "low", "info"
    description: str
    impact: str
    proof_of_concept: Optional[str] = None
    remediation_suggestion: Optional[str] = None
    status: str = "received"  # "received", "triaged", "confirmed", "fixed", "rejected"
    bounty_amount: Optional[float] = None
    disclosure_date: Optional[str] = None

@dataclass
class ComplianceAssessment:
    """Regulatory compliance assessment"""
    assessment_id: str
    regulation: str
    assessment_date: str
    assessor: str
    compliance_level: str  # "compliant", "non_compliant", "partial_compliance"
    next_assessment_date: str
    gaps_identified: List[str] = field(default_factory=list)
    remediation_plan: List[str] = field(default_factory=list)
    documentation: List[str] = field(default_factory=list)

class ExternalRedTeamingCertification:
    """
    External Red Teaming & Certification: Independent Security Audits

    Manages external security audits, certifications, red teaming exercises,
    and regulatory compliance for the Kalki self-evolving AI system.
    """

    def __init__(self):
        self.evolution_manager = get_self_evolution_manager()
        self.safety_monitor = get_safety_monitoring_system()
        self.adversarial_tests = get_simulated_adversarial_tests()

        # Audit management
        self.external_audits: Dict[str, ExternalAudit] = {}
        self.certifications: Dict[str, Certification] = {}
        self.red_team_exercises: Dict[str, RedTeamExercise] = {}
        self.vulnerability_reports: Dict[str, VulnerabilityReport] = {}
        self.compliance_assessments: Dict[str, ComplianceAssessment] = {}

        # Configuration
        self.audit_schedule_months = 12  # Annual audits
        self.certification_renewal_months = 36  # 3-year certifications
        self.red_team_frequency_months = 6  # Bi-annual red teaming

        # Bug bounty program
        self.bug_bounty_active = True
        self.bounty_tiers = {
            "critical": 10000,
            "high": 5000,
            "medium": 1000,
            "low": 100
        }

        # Notification settings
        self.email_config = {
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "sender_email": "security@kalki.ai",
            "sender_password": os.getenv("KALKI_SECURITY_EMAIL_PASSWORD", "")
        }

        # Persistence
        self.data_dir = "data/red_teaming"
        self.audits_file = f"{self.data_dir}/external_audits.json"
        self.certifications_file = f"{self.data_dir}/certifications.json"
        self.red_team_file = f"{self.data_dir}/red_team_exercises.json"
        self.vulnerabilities_file = f"{self.data_dir}/vulnerability_reports.json"
        self.compliance_file = f"{self.data_dir}/compliance_assessments.json"

        # Initialize
        self._initialize_red_teaming_system()

        logger.info("External Red Teaming & Certification initialized")

    def _initialize_red_teaming_system(self):
        """Initialize the red teaming and certification system"""

        # Load existing data
        self._load_red_teaming_data()

        # Set up default audit schedule
        self._setup_default_audit_schedule()

        # Set up certification tracking
        self._setup_certification_tracking()

    def _setup_default_audit_schedule(self):
        """Set up default external audit schedule"""

        default_audits = [
            {
                "title": "Annual Security Penetration Testing",
                "description": "Comprehensive penetration testing of all system components",
                "audit_type": AuditType.SECURITY_PENETRATION,
                "auditor_organization": "Offensive Security Inc.",
                "auditor_contact": {
                    "name": "Marcus Chen",
                    "email": "marcus@offensivesecurity.com",
                    "phone": "+1-555-0101"
                },
                "scope": ["Network security", "Application security", "API security", "Data protection"],
                "start_date": (datetime.now() + timedelta(days=30)).isoformat(),
                "end_date": (datetime.now() + timedelta(days=45)).isoformat()
            },
            {
                "title": "Ethical AI Assessment",
                "description": "Independent review of AI ethics and responsible AI practices",
                "audit_type": AuditType.ETHICS_REVIEW,
                "auditor_organization": "AI Ethics Institute",
                "auditor_contact": {
                    "name": "Dr. Sarah Ethics",
                    "email": "sarah@aiethics.org",
                    "phone": "+1-555-0102"
                },
                "scope": ["Bias assessment", "Fairness evaluation", "Transparency review", "Accountability measures"],
                "start_date": (datetime.now() + timedelta(days=60)).isoformat(),
                "end_date": (datetime.now() + timedelta(days=75)).isoformat()
            },
            {
                "title": "Safety Certification Audit",
                "description": "Safety standards compliance and risk assessment",
                "audit_type": AuditType.SAFETY_CERTIFICATION,
                "auditor_organization": "Global Safety Standards Organization",
                "auditor_contact": {
                    "name": "Prof. Jordan Safety",
                    "email": "jordan@safetyglobal.org",
                    "phone": "+1-555-0103"
                },
                "scope": ["Safety mechanisms", "Fail-safe systems", "Risk mitigation", "Incident response"],
                "start_date": (datetime.now() + timedelta(days=90)).isoformat(),
                "end_date": (datetime.now() + timedelta(days=105)).isoformat()
            }
        ]

        for audit_data in default_audits:
            audit_id = f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(audit_data['title'].encode()).hexdigest()[:8]}"

            audit = ExternalAudit(
                audit_id=audit_id,
                status="scheduled",
                **audit_data
            )

            self.external_audits[audit_id] = audit

    def _setup_certification_tracking(self):
        """Set up certification tracking"""

        # Placeholder certifications - would be populated with actual certifications
        certifications = [
            {
                "standard": CertificationStandard.NIST_AI_RMF,
                "issuing_body": "National Institute of Standards and Technology",
                "issue_date": (datetime.now() - timedelta(days=180)).isoformat(),
                "expiry_date": (datetime.now() + timedelta(days=545)).isoformat(),  # 1.5 years
                "status": "active",
                "scope": "AI system development and deployment",
                "certificate_url": "https://www.nist.gov/certificates/kalki-nist-ai-rmf-2024"
            }
        ]

        for cert_data in certifications:
            cert_id = f"cert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(str(cert_data).encode()).hexdigest()[:8]}"

            certification = Certification(
                certification_id=cert_id,
                **cert_data
            )

            self.certifications[cert_id] = certification

    def schedule_external_audit(self, title: str, description: str, audit_type: AuditType,
                              auditor_org: str, auditor_contact: Dict[str, str],
                              scope: List[str], start_date: str, end_date: str) -> str:
        """
        Schedule a new external audit

        Returns:
            Audit ID
        """

        audit_id = f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(title.encode()).hexdigest()[:8]}"

        audit = ExternalAudit(
            audit_id=audit_id,
            title=title,
            description=description,
            audit_type=audit_type,
            auditor_organization=auditor_org,
            auditor_contact=auditor_contact,
            scope=scope,
            start_date=start_date,
            end_date=end_date,
            status="scheduled"
        )

        self.external_audits[audit_id] = audit

        # Send notification
        asyncio.create_task(self._notify_audit_scheduled(audit))

        logger.info(f"Scheduled external audit: {audit_id} - {title}")

        return audit_id

    def submit_vulnerability_report(self, submitter: str, vulnerability_type: str,
                                 severity: str, description: str, impact: str,
                                 poc: Optional[str] = None, remediation: Optional[str] = None) -> str:
        """
        Submit a vulnerability report (for bug bounty or responsible disclosure)

        Returns:
            Report ID
        """

        report_id = f"vuln_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        report = VulnerabilityReport(
            report_id=report_id,
            submitter=submitter,
            submission_date=datetime.now().isoformat(),
            vulnerability_type=vulnerability_type,
            severity=severity,
            description=description,
            impact=impact,
            proof_of_concept=poc,
            remediation_suggestion=remediation
        )

        self.vulnerability_reports[report_id] = report

        # Calculate bounty amount
        if severity in self.bounty_tiers:
            report.bounty_amount = self.bounty_tiers[severity]
            report.status = "triaged"

        # Send acknowledgment
        asyncio.create_task(self._notify_vulnerability_received(report))

        logger.info(f"Vulnerability report submitted: {report_id} - {severity} severity")

        return report_id

    def update_audit_status(self, audit_id: str, status: str,
                          findings: List[Dict[str, Any]] = None,
                          recommendations: List[str] = None,
                          compliance_score: float = None) -> bool:
        """
        Update audit status and results

        Returns:
            True if update successful
        """

        if audit_id not in self.external_audits:
            logger.error(f"Audit {audit_id} not found")
            return False

        audit = self.external_audits[audit_id]
        audit.status = status

        if findings:
            audit.findings.extend(findings)

        if recommendations:
            audit.recommendations.extend(recommendations)

        if compliance_score is not None:
            audit.compliance_score = compliance_score

        if status == "completed":
            # Schedule next audit
            next_audit = datetime.fromisoformat(audit.end_date) + timedelta(days=self.audit_schedule_months * 30)
            audit.next_audit_date = next_audit.isoformat()

            # Send completion notification
            asyncio.create_task(self._notify_audit_completed(audit))

        logger.info(f"Audit {audit_id} status updated to {status}")

        return True

    def conduct_red_team_exercise(self, name: str, engagement_type: RedTeamEngagement,
                                red_team_org: str, objectives: List[str],
                                rules: Dict[str, Any], duration_days: int) -> str:
        """
        Conduct a red teaming exercise

        Returns:
            Exercise ID
        """

        exercise_id = f"redteam_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(name.encode()).hexdigest()[:8]}"

        start_date = datetime.now().isoformat()
        end_date = (datetime.now() + timedelta(days=duration_days)).isoformat()

        exercise = RedTeamExercise(
            exercise_id=exercise_id,
            name=name,
            engagement_type=engagement_type,
            red_team_organization=red_team_org,
            objectives=objectives,
            rules_of_engagement=rules,
            start_date=start_date,
            end_date=end_date,
            status="active"
        )

        self.red_team_exercises[exercise_id] = exercise

        # Send notification
        asyncio.create_task(self._notify_red_team_started(exercise))

        logger.info(f"Red team exercise started: {exercise_id} - {name}")

        return exercise_id

    def assess_regulatory_compliance(self, regulation: str, assessor: str) -> str:
        """
        Initiate regulatory compliance assessment

        Returns:
            Assessment ID
        """

        assessment_id = f"compliance_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(regulation.encode()).hexdigest()[:8]}"

        next_assessment = datetime.now() + timedelta(days=365)  # Annual assessment

        assessment = ComplianceAssessment(
            assessment_id=assessment_id,
            regulation=regulation,
            assessment_date=datetime.now().isoformat(),
            assessor=assessor,
            compliance_level="pending_assessment",
            next_assessment_date=next_assessment.isoformat()
        )

        self.compliance_assessments[assessment_id] = assessment

        logger.info(f"Compliance assessment initiated: {assessment_id} - {regulation}")

        return assessment_id

    async def _notify_audit_scheduled(self, audit: ExternalAudit):
        """Notify stakeholders of scheduled audit"""

        if not self.email_config["sender_password"]:
            return

        subject = f"Kalki External Audit Scheduled: {audit.title}"

        # Notify internal stakeholders
        recipients = ["security@kalki.ai", "compliance@kalki.ai", "executives@kalki.ai"]

        body = f"""
External Audit Scheduled

Title: {audit.title}
Type: {audit.audit_type.value.replace('_', ' ').title()}
Auditor: {audit.auditor_organization}
Contact: {audit.auditor_contact['name']} ({audit.auditor_contact['email']})

Scope:
{chr(10).join(f"- {item}" for item in audit.scope)}

Schedule: {audit.start_date} to {audit.end_date}

Please ensure all necessary preparations are completed and documentation is ready.

Best regards,
Kalki Security Team
"""

        for email in recipients:
            await self._send_email(email, subject, body)

    async def _notify_vulnerability_received(self, report: VulnerabilityReport):
        """Notify of vulnerability report receipt"""

        if not self.email_config["sender_password"]:
            return

        subject = f"Vulnerability Report Received - {report.severity.upper()} Severity"

        body = f"""
Vulnerability Report Acknowledgment

Report ID: {report.report_id}
Submitted by: {report.submitter}
Severity: {report.severity.upper()}
Type: {report.vulnerability_type}

Description:
{report.description}

Impact:
{report.impact}

{f"Bounty Amount: ${report.bounty_amount}" if report.bounty_amount else "Bounty: To be determined"}

We will investigate this report and respond within 48 hours.

Thank you for helping keep Kalki secure!

Best regards,
Kalki Security Team
"""

        # Send to submitter and security team
        recipients = [report.submitter, "security@kalki.ai"]

        for email in recipients:
            await self._send_email(email, subject, body)

    async def _notify_audit_completed(self, audit: ExternalAudit):
        """Notify of audit completion"""

        if not self.email_config["sender_password"]:
            return

        subject = f"External Audit Completed: {audit.title}"

        recipients = ["security@kalki.ai", "compliance@kalki.ai", "executives@kalki.ai"]

        status_emoji = "✅" if (audit.compliance_score or 0) >= 0.8 else "⚠️"

        body = f"""
{status_emoji} External Audit Completed

Title: {audit.title}
Auditor: {audit.auditor_organization}
Completed: {audit.end_date}

Compliance Score: {audit.compliance_score:.1% if audit.compliance_score else 'N/A'}

Key Findings:
{chr(10).join(f"- {f.get('summary', 'Finding')}" for f in audit.findings[:5])}

Recommendations:
{chr(10).join(f"- {rec}" for rec in audit.recommendations[:5])}

{"Full report available at: " + audit.report_url if audit.report_url else ""}

Next Audit: {audit.next_audit_date or 'To be scheduled'}

Best regards,
Kalki Security Team
"""

        for email in recipients:
            await self._send_email(email, subject, body)

    async def _notify_red_team_started(self, exercise: RedTeamExercise):
        """Notify of red team exercise start"""

        if not self.email_config["sender_password"]:
            return

        subject = f"Red Team Exercise Started: {exercise.name}"

        recipients = ["security@kalki.ai", "executives@kalki.ai"]

        body = f"""
🔴 Red Team Exercise Initiated

Name: {exercise.name}
Type: {exercise.engagement_type.value.replace('_', ' ').title()}
Red Team: {exercise.red_team_organization}

Objectives:
{chr(10).join(f"- {obj}" for obj in exercise.objectives)}

Duration: {exercise.start_date} to {exercise.end_date}

Rules of Engagement:
- Authorized activities only
- No production system disruption without approval
- Immediate cessation if safety concerns arise

Please monitor system alerts and be prepared to respond to findings.

Best regards,
Kalki Security Team
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

    def get_security_posture(self) -> Dict[str, Any]:
        """Get overall security posture assessment"""

        # Calculate security metrics
        active_audits = len([a for a in self.external_audits.values() if a.status in ["scheduled", "in_progress"]])
        completed_audits = len([a for a in self.external_audits.values() if a.status == "completed"])

        avg_compliance = 0.0
        compliance_scores = [a.compliance_score for a in self.external_audits.values() if a.compliance_score]
        if compliance_scores:
            avg_compliance = sum(compliance_scores) / len(compliance_scores)

        active_certifications = len([c for c in self.certifications.values() if c.status == "active"])
        expiring_soon = len([
            c for c in self.certifications.values()
            if c.status == "active" and
            datetime.fromisoformat(c.expiry_date) < datetime.now() + timedelta(days=90)
        ])

        open_vulnerabilities = len([
            v for v in self.vulnerability_reports.values()
            if v.status not in ["fixed", "rejected"]
        ])

        active_red_team = len([
            e for e in self.red_team_exercises.values()
            if e.status == "active"
        ])

        # Overall security score (0-1 scale)
        security_components = [
            min(avg_compliance, 1.0),  # Compliance score
            1.0 if active_certifications > 0 else 0.5,  # Certification coverage
            max(0, 1.0 - (open_vulnerabilities * 0.1)),  # Vulnerability management
            1.0 if active_red_team > 0 else 0.7,  # Red teaming activity
        ]

        overall_security_score = sum(security_components) / len(security_components)

        return {
            "overall_security_score": overall_security_score,
            "active_audits": active_audits,
            "completed_audits": completed_audits,
            "average_compliance_score": avg_compliance,
            "active_certifications": active_certifications,
            "certifications_expiring_soon": expiring_soon,
            "open_vulnerability_reports": open_vulnerabilities,
            "active_red_team_exercises": active_red_team,
            "last_updated": datetime.now().isoformat()
        }

    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report"""

        report = {
            "report_id": f"security_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "generated_at": datetime.now().isoformat(),
            "period": "Last 30 days",
            "executive_summary": {},
            "audit_results": {},
            "certification_status": {},
            "vulnerability_management": {},
            "red_teaming_activities": {},
            "compliance_assessments": {},
            "recommendations": []
        }

        # Executive summary
        posture = self.get_security_posture()
        report["executive_summary"] = {
            "overall_security_score": f"{posture['overall_security_score']:.1%}",
            "key_metrics": {
                "Active Audits": posture["active_audits"],
                "Compliance Score": f"{posture['average_compliance_score']:.1%}",
                "Active Certifications": posture["active_certifications"],
                "Open Vulnerabilities": posture["open_vulnerability_reports"]
            }
        }

        # Audit results
        recent_audits = [
            a for a in self.external_audits.values()
            if datetime.fromisoformat(a.end_date) > datetime.now() - timedelta(days=30)
        ]
        report["audit_results"] = {
            "total_recent_audits": len(recent_audits),
            "passed_audits": len([a for a in recent_audits if (a.compliance_score or 0) >= 0.8]),
            "failed_audits": len([a for a in recent_audits if a.compliance_score and a.compliance_score < 0.8])
        }

        # Certification status
        report["certification_status"] = {
            "active_certifications": [
                {
                    "standard": c.standard.value,
                    "issuing_body": c.issuing_body,
                    "expiry_date": c.expiry_date
                }
                for c in self.certifications.values() if c.status == "active"
            ],
            "expiring_soon": posture["certifications_expiring_soon"]
        }

        # Vulnerability management
        report["vulnerability_management"] = {
            "total_reports": len(self.vulnerability_reports),
            "open_reports": posture["open_vulnerability_reports"],
            "by_severity": {
                severity: len([v for v in self.vulnerability_reports.values() if v.severity == severity])
                for severity in ["critical", "high", "medium", "low", "info"]
            },
            "avg_resolution_time_days": 7  # placeholder
        }

        # Red teaming activities
        report["red_teaming_activities"] = {
            "total_exercises": len(self.red_team_exercises),
            "active_exercises": posture["active_red_team_exercises"],
            "completed_this_period": len([
                e for e in self.red_team_exercises.values()
                if e.status == "completed" and
                datetime.fromisoformat(e.end_date) > datetime.now() - timedelta(days=30)
            ])
        }

        # Generate recommendations
        report["recommendations"] = self._generate_security_recommendations(posture)

        return report

    def _generate_security_recommendations(self, posture: Dict[str, Any]) -> List[str]:
        """Generate security recommendations"""

        recommendations = []

        if posture["average_compliance_score"] < 0.8:
            recommendations.append("Improve audit compliance scores through targeted remediation")

        if posture["certifications_expiring_soon"] > 0:
            recommendations.append("Renew expiring certifications before expiry dates")

        if posture["open_vulnerability_reports"] > 5:
            recommendations.append("Address backlog of open vulnerability reports")

        if posture["active_red_team_exercises"] == 0:
            recommendations.append("Schedule regular red teaming exercises")

        if posture["overall_security_score"] < 0.7:
            recommendations.append("Implement comprehensive security improvement plan")

        recommendations.extend([
            "Conduct quarterly threat modeling exercises",
            "Implement automated security testing in CI/CD pipeline",
            "Establish security awareness training program",
            "Regular security architecture reviews"
        ])

        return recommendations

    def _load_red_teaming_data(self):
        """Load red teaming data from files"""
        try:
            os.makedirs(self.data_dir, exist_ok=True)

            files_data = [
                (self.audits_file, self.external_audits, ExternalAudit),
                (self.certifications_file, self.certifications, Certification),
                (self.red_team_file, self.red_team_exercises, RedTeamExercise),
                (self.vulnerabilities_file, self.vulnerability_reports, VulnerabilityReport),
                (self.compliance_file, self.compliance_assessments, ComplianceAssessment)
            ]

            for file_path, data_dict, data_class in files_data:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        items = json.load(f)
                        for item_data in items.values():
                            item = data_class(**item_data)
                            data_dict[item_data[list(item_data.keys())[0]]] = item

        except Exception as e:
            logger.warning(f"Failed to load red teaming data: {e}")

    def _save_red_teaming_data(self):
        """Save red teaming data to files"""
        try:
            files_data = [
                (self.audits_file, self.external_audits),
                (self.certifications_file, self.certifications),
                (self.red_team_file, self.red_team_exercises),
                (self.vulnerabilities_file, self.vulnerability_reports),
                (self.compliance_file, self.compliance_assessments)
            ]

            for file_path, data_dict in files_data:
                data = {k: asdict(v) for k, v in data_dict.items()}
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save red teaming data: {e}")

# Global red teaming and certification instance
_red_teaming_certification = None

def get_external_red_teaming_certification() -> ExternalRedTeamingCertification:
    """Get the global external red teaming and certification instance"""
    global _red_teaming_certification
    if _red_teaming_certification is None:
        _red_teaming_certification = ExternalRedTeamingCertification()
    return _red_teaming_certification

# Convenience functions
def schedule_security_audit(auditor_org: str, scope: List[str], start_date: str) -> str:
    """Schedule a security audit"""
    rtc = get_external_red_teaming_certification()
    end_date = (datetime.fromisoformat(start_date) + timedelta(days=14)).isoformat()

    return rtc.schedule_external_audit(
        title=f"Security Audit by {auditor_org}",
        description=f"Comprehensive security assessment by {auditor_org}",
        audit_type=AuditType.SECURITY_PENETRATION,
        auditor_org=auditor_org,
        auditor_contact={"name": "Contact", "email": "contact@auditor.com", "phone": "555-0000"},
        scope=scope,
        start_date=start_date,
        end_date=end_date
    )

def submit_security_vulnerability(submitter: str, vuln_type: str, severity: str,
                                description: str, impact: str) -> str:
    """Submit a security vulnerability"""
    rtc = get_external_red_teaming_certification()
    return rtc.submit_vulnerability_report(submitter, vuln_type, severity, description, impact)