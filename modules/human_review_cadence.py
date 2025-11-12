# ============================================================
# Kalki v2.4 — human_review_cadence.py
# ------------------------------------------------------------
# Human Review Cadence: Weekly Manual Review Process
# - Automated scheduling of human reviews for high-impact changes
# - Review queue management and prioritization
# - Stakeholder notification and coordination
# - Review feedback integration into learning system
# - Audit trail of human oversight decisions
# ============================================================

import os
import json
import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import schedule
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import hmac

from modules.utils.logging_config import get_logger
from modules.self_evolution_manager import get_self_evolution_manager
from modules.safety_monitoring_system import get_safety_monitoring_system

logger = get_logger("Kalki.HumanReview")

class ReviewPriority(Enum):
    """Review priority levels"""
    CRITICAL = "critical"    # Immediate review required
    HIGH = "high"           # Review within 24 hours
    MEDIUM = "medium"       # Review within 1 week
    LOW = "low"            # Review within 1 month
    MONITOR = "monitor"     # Monitor but no immediate action

class ReviewStatus(Enum):
    """Review status"""
    PENDING = "pending"         # Awaiting review
    IN_REVIEW = "in_review"      # Currently being reviewed
    APPROVED = "approved"        # Approved for implementation
    REJECTED = "rejected"        # Rejected
    REQUIRES_CHANGES = "requires_changes"  # Needs modifications
    ESCALATED = "escalated"      # Escalated to higher authority
    EXPIRED = "expired"         # Review period expired

class ReviewerRole(Enum):
    """Reviewer roles and responsibilities"""
    ETHICS_REVIEWER = "ethics_reviewer"
    SECURITY_REVIEWER = "security_reviewer"
    TECHNICAL_REVIEWER = "technical_reviewer"
    DOMAIN_EXPERT = "domain_expert"
    LEGAL_REVIEWER = "legal_reviewer"
    EXECUTIVE_SPONSOR = "executive_sponsor"

@dataclass
class ReviewItem:
    """Individual item requiring human review"""
    item_id: str
    title: str
    description: str
    category: str
    priority: ReviewPriority
    status: ReviewStatus
    created_timestamp: str
    due_date: str
    required_reviewers: List[ReviewerRole]
    assigned_reviewers: List[str] = field(default_factory=list)
    review_criteria: Dict[str, Any] = field(default_factory=dict)
    supporting_documents: List[str] = field(default_factory=list)
    related_items: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReviewDecision:
    """Human review decision and feedback"""
    decision_id: str
    item_id: str
    reviewer_id: str
    reviewer_role: str
    decision: ReviewStatus
    timestamp: str
    rationale: str
    confidence_level: float  # 0.0 to 1.0
    required_changes: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    follow_up_actions: List[str] = field(default_factory=list)
    review_duration_minutes: int = 0

@dataclass
class ReviewCycle:
    """Weekly review cycle"""
    cycle_id: str
    start_date: str
    end_date: str
    status: str  # "active", "completed", "overdue"
    items_reviewed: int = 0
    items_pending: int = 0
    items_approved: int = 0
    items_rejected: int = 0
    average_review_time: float = 0.0
    reviewer_participation: Dict[str, int] = field(default_factory=dict)

@dataclass
class HumanReviewCadence:
    """
    Human Review Cadence: Weekly Manual Review Process

    Manages the scheduling, coordination, and tracking of human reviews
    for high-impact AI system changes and recommendations.
    """

    def __init__(self):
        self.evolution_manager = get_self_evolution_manager()
        self.safety_monitor = get_safety_monitoring_system()

        # Review management
        self.review_queue: Dict[str, ReviewItem] = {}
        self.review_decisions: Dict[str, List[ReviewDecision]] = {}
        self.review_cycles: Dict[str, ReviewCycle] = {}

        # Reviewer management
        self.reviewers: Dict[str, Dict[str, Any]] = {}
        self.reviewer_schedule: Dict[str, List[str]] = {}  # reviewer_id -> available_dates

        # Configuration
        self.weekly_review_day = "friday"  # Day of week for reviews
        self.weekly_review_time = "14:00"  # Time for reviews
        self.review_deadlines = {
            ReviewPriority.CRITICAL: timedelta(hours=4),
            ReviewPriority.HIGH: timedelta(days=1),
            ReviewPriority.MEDIUM: timedelta(days=7),
            ReviewPriority.LOW: timedelta(days=30)
        }

        # Notification settings
        self.email_enabled = True
        self.email_config = {
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "sender_email": "kalki.notifications@example.com",
            "sender_password": os.getenv("KALKI_EMAIL_PASSWORD", "")
        }

        # Persistence
        self.data_dir = "data/human_review"
        self.queue_file = f"{self.data_dir}/review_queue.json"
        self.decisions_file = f"{self.data_dir}/review_decisions.json"
        self.cycles_file = f"{self.data_dir}/review_cycles.json"

        # Initialize
        self._initialize_review_system()

        logger.info("Human Review Cadence initialized")

    def _initialize_review_system(self):
        """Initialize the human review system"""

        # Load existing data
        self._load_review_data()

        # Set up default reviewers
        self._setup_default_reviewers()

        # Schedule weekly reviews
        self._schedule_weekly_reviews()

    def start_scheduler(self):
        """Start the background scheduler for automated reviews"""
        if not hasattr(self, '_scheduler_task') or self._scheduler_task is None:
            self._scheduler_task = asyncio.create_task(self._run_scheduler())

    def _setup_default_reviewers(self):
        """Set up default reviewer team"""

        default_reviewers = [
            {
                "reviewer_id": "ethics_lead",
                "name": "Dr. Sarah Ethics",
                "email": "ethics@kalki.ai",
                "roles": [ReviewerRole.ETHICS_REVIEWER, ReviewerRole.EXECUTIVE_SPONSOR],
                "expertise": ["AI Ethics", "Philosophy", "Regulatory Compliance"],
                "availability": ["monday", "wednesday", "friday"]
            },
            {
                "reviewer_id": "security_lead",
                "name": "Marcus Security",
                "email": "security@kalki.ai",
                "roles": [ReviewerRole.SECURITY_REVIEWER],
                "expertise": ["Cybersecurity", "AI Safety", "Risk Assessment"],
                "availability": ["tuesday", "thursday", "friday"]
            },
            {
                "reviewer_id": "technical_lead",
                "name": "Dr. Alex Technical",
                "email": "technical@kalki.ai",
                "roles": [ReviewerRole.TECHNICAL_REVIEWER],
                "expertise": ["Machine Learning", "Software Architecture", "Performance"],
                "availability": ["monday", "tuesday", "wednesday", "thursday", "friday"]
            },
            {
                "reviewer_id": "domain_expert_engineering",
                "name": "Prof. Jordan Engineering",
                "email": "engineering@kalki.ai",
                "roles": [ReviewerRole.DOMAIN_EXPERT],
                "expertise": ["Mechanical Engineering", "CAD/CAM", "Manufacturing"],
                "availability": ["monday", "wednesday"]
            },
            {
                "reviewer_id": "legal_counsel",
                "name": "Attorney Riley Legal",
                "email": "legal@kalki.ai",
                "roles": [ReviewerRole.LEGAL_REVIEWER],
                "expertise": ["Technology Law", "IP Law", "Compliance"],
                "availability": ["tuesday", "thursday"]
            }
        ]

        for reviewer in default_reviewers:
            self.reviewers[reviewer["reviewer_id"]] = reviewer

    def _schedule_weekly_reviews(self):
        """Schedule weekly review meetings"""

        # Schedule the weekly review function
        if self.weekly_review_day.lower() == "friday":
            schedule.every().friday.at(self.weekly_review_time).do(self._conduct_weekly_review)

        # Note: Scheduler will be started when start_scheduler() is called

    async def _run_scheduler(self):
        """Run the background scheduler"""
        while True:
            schedule.run_pending()
            await asyncio.sleep(60)  # Check every minute

    def submit_for_review(self, title: str, description: str, category: str,
                         priority: ReviewPriority, review_criteria: Dict[str, Any],
                         supporting_docs: List[str] = None, metadata: Dict[str, Any] = None) -> str:
        """
        Submit an item for human review

        Args:
            title: Brief title of the item
            description: Detailed description
            category: Category (e.g., "evolution_recommendation", "safety_change")
            priority: Review priority level
            review_criteria: Specific criteria for review
            supporting_docs: List of document paths
            metadata: Additional metadata

        Returns:
            Review item ID
        """

        item_id = f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(title.encode()).hexdigest()[:8]}"

        # Determine required reviewers based on category and priority
        required_reviewers = self._determine_required_reviewers(category, priority)

        # Calculate due date
        due_date = datetime.now() + self.review_deadlines[priority]

        review_item = ReviewItem(
            item_id=item_id,
            title=title,
            description=description,
            category=category,
            priority=priority,
            status=ReviewStatus.PENDING,
            created_timestamp=datetime.now().isoformat(),
            due_date=due_date.isoformat(),
            required_reviewers=required_reviewers,
            review_criteria=review_criteria,
            supporting_documents=supporting_docs or [],
            metadata=metadata or {}
        )

        self.review_queue[item_id] = review_item
        self.review_decisions[item_id] = []

        # Auto-assign reviewers if possible
        self._auto_assign_reviewers(review_item)

        # Send notifications
        asyncio.create_task(self._notify_review_submission(review_item))

        # Immediate notification for critical items
        if priority == ReviewPriority.CRITICAL:
            asyncio.create_task(self._send_critical_notification(review_item))

        logger.info(f"Submitted item for review: {item_id} - {title}")

        return item_id

    def _determine_required_reviewers(self, category: str, priority: ReviewPriority) -> List[ReviewerRole]:
        """Determine which reviewer roles are required"""

        base_reviewers = [ReviewerRole.TECHNICAL_REVIEWER]

        if priority in [ReviewPriority.CRITICAL, ReviewPriority.HIGH]:
            base_reviewers.extend([ReviewerRole.ETHICS_REVIEWER, ReviewerRole.SECURITY_REVIEWER])

        if category in ["evolution_recommendation", "behavior_change"]:
            base_reviewers.append(ReviewerRole.ETHICS_REVIEWER)

        if category in ["safety_change", "security_update"]:
            base_reviewers.append(ReviewerRole.SECURITY_REVIEWER)

        if category in ["legal_compliance", "regulatory_change"]:
            base_reviewers.append(ReviewerRole.LEGAL_REVIEWER)

        if category in ["domain_integration", "engineering_workflow"]:
            base_reviewers.append(ReviewerRole.DOMAIN_EXPERT)

        if priority == ReviewPriority.CRITICAL:
            base_reviewers.append(ReviewerRole.EXECUTIVE_SPONSOR)

        return list(set(base_reviewers))  # Remove duplicates

    def _auto_assign_reviewers(self, review_item: ReviewItem):
        """Auto-assign reviewers based on availability and expertise"""

        assigned = []

        for role in review_item.required_reviewers:
            # Find available reviewers for this role
            available_reviewers = [
                r_id for r_id, reviewer in self.reviewers.items()
                if role in reviewer["roles"] and self._is_reviewer_available(r_id, review_item.due_date)
            ]

            if available_reviewers:
                # Assign the first available reviewer (could be more sophisticated)
                assigned.append(available_reviewers[0])

        review_item.assigned_reviewers = assigned

    def _is_reviewer_available(self, reviewer_id: str, due_date: str) -> bool:
        """Check if reviewer is available for the due date"""

        due_datetime = datetime.fromisoformat(due_date)
        day_name = due_datetime.strftime('%A').lower()

        reviewer = self.reviewers.get(reviewer_id, {})
        availability = reviewer.get("availability", [])

        return day_name in availability

    async def submit_review_decision(self, item_id: str, reviewer_id: str,
                                   decision: ReviewStatus, rationale: str,
                                   confidence_level: float, required_changes: List[str] = None,
                                   recommendations: List[str] = None, review_duration: int = 0) -> bool:
        """
        Submit a review decision

        Returns:
            True if decision was recorded successfully
        """

        if item_id not in self.review_queue:
            logger.error(f"Review item {item_id} not found")
            return False

        review_item = self.review_queue[item_id]

        # Validate reviewer
        if reviewer_id not in review_item.assigned_reviewers:
            logger.error(f"Reviewer {reviewer_id} not assigned to item {item_id}")
            return False

        # Create decision record
        decision_record = ReviewDecision(
            decision_id=f"decision_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            item_id=item_id,
            reviewer_id=reviewer_id,
            reviewer_role=self._get_reviewer_role(reviewer_id),
            decision=decision,
            timestamp=datetime.now().isoformat(),
            rationale=rationale,
            confidence_level=confidence_level,
            required_changes=required_changes or [],
            recommendations=recommendations or [],
            review_duration_minutes=review_duration
        )

        # Record decision
        if item_id not in self.review_decisions:
            self.review_decisions[item_id] = []
        self.review_decisions[item_id].append(decision_record)

        # Update review item status
        self._update_review_status(review_item)

        # Notify stakeholders
        await self._notify_decision_submitted(review_item, decision_record)

        logger.info(f"Review decision submitted for {item_id}: {decision.value}")

        return True

    def _get_reviewer_role(self, reviewer_id: str) -> str:
        """Get the primary role of a reviewer"""
        reviewer = self.reviewers.get(reviewer_id, {})
        roles = reviewer.get("roles", [])
        return roles[0].value if roles else "unknown"

    def _update_review_status(self, review_item: ReviewItem):
        """Update review item status based on decisions"""

        decisions = self.review_decisions.get(review_item.item_id, [])

        if not decisions:
            return

        # Check if all required reviewers have decided
        required_roles = {r.value for r in review_item.required_reviewers}
        decided_roles = {d.reviewer_role for d in decisions}

        if not required_roles.issubset(decided_roles):
            # Still waiting for some reviewers
            review_item.status = ReviewStatus.IN_REVIEW
            return

        # All reviewers have decided - determine final status
        decision_counts = {}
        for decision in decisions:
            decision_counts[decision.decision] = decision_counts.get(decision.decision, 0) + 1

        # Simple majority rules (could be more sophisticated)
        final_decision = max(decision_counts, key=decision_counts.get)

        if final_decision == ReviewStatus.APPROVED:
            review_item.status = ReviewStatus.APPROVED
        elif final_decision == ReviewStatus.REJECTED:
            review_item.status = ReviewStatus.REJECTED
        elif final_decision == ReviewStatus.REQUIRES_CHANGES:
            review_item.status = ReviewStatus.REQUIRES_CHANGES
        else:
            review_item.status = ReviewStatus.ESCALATED

    def _conduct_weekly_review(self):
        """Conduct the weekly review meeting"""

        logger.info("Conducting weekly human review cadence")

        current_cycle = self._get_current_review_cycle()

        # Get pending items
        pending_items = [
            item for item in self.review_queue.values()
            if item.status in [ReviewStatus.PENDING, ReviewStatus.IN_REVIEW]
        ]

        # Prioritize items
        prioritized_items = sorted(pending_items, key=lambda x: (
            self._priority_score(x.priority),
            datetime.fromisoformat(x.created_timestamp)
        ), reverse=True)

        # Generate review summary
        summary = self._generate_weekly_summary(prioritized_items)

        # Send review notifications
        asyncio.create_task(self._send_weekly_review_notifications(summary))

        # Update cycle statistics
        current_cycle.items_pending = len(pending_items)
        current_cycle.items_reviewed = len([
            item for item in self.review_queue.values()
            if item.status not in [ReviewStatus.PENDING, ReviewStatus.IN_REVIEW]
        ])

        logger.info(f"Weekly review completed. {len(pending_items)} items pending review")

    def _priority_score(self, priority: ReviewPriority) -> int:
        """Get numeric priority score"""
        scores = {
            ReviewPriority.CRITICAL: 5,
            ReviewPriority.HIGH: 4,
            ReviewPriority.MEDIUM: 3,
            ReviewPriority.LOW: 2,
            ReviewPriority.MONITOR: 1
        }
        return scores.get(priority, 0)

    def _get_current_review_cycle(self) -> ReviewCycle:
        """Get or create current review cycle"""

        today = date.today()
        # Find Friday of current week
        days_until_friday = (4 - today.weekday()) % 7  # 4 = Friday
        friday = today + timedelta(days=days_until_friday)

        cycle_id = f"cycle_{friday.isoformat()}"

        if cycle_id not in self.review_cycles:
            self.review_cycles[cycle_id] = ReviewCycle(
                cycle_id=cycle_id,
                start_date=(friday - timedelta(days=6)).isoformat(),
                end_date=friday.isoformat(),
                status="active"
            )

        return self.review_cycles[cycle_id]

    def _generate_weekly_summary(self, prioritized_items: List[ReviewItem]) -> Dict[str, Any]:
        """Generate weekly review summary"""

        summary = {
            "cycle_id": self._get_current_review_cycle().cycle_id,
            "generated_at": datetime.now().isoformat(),
            "total_pending": len(prioritized_items),
            "critical_items": len([i for i in prioritized_items if i.priority == ReviewPriority.CRITICAL]),
            "high_priority": len([i for i in prioritized_items if i.priority == ReviewPriority.HIGH]),
            "categories": {},
            "overdue_items": [],
            "reviewer_workload": {}
        }

        # Category breakdown
        for item in prioritized_items:
            summary["categories"][item.category] = summary["categories"].get(item.category, 0) + 1

        # Check for overdue items
        now = datetime.now()
        for item in prioritized_items:
            due_date = datetime.fromisoformat(item.due_date)
            if now > due_date:
                summary["overdue_items"].append({
                    "item_id": item.item_id,
                    "title": item.title,
                    "days_overdue": (now - due_date).days
                })

        # Reviewer workload
        for reviewer_id in self.reviewers.keys():
            assigned_items = [
                item for item in prioritized_items
                if reviewer_id in item.assigned_reviewers
            ]
            summary["reviewer_workload"][reviewer_id] = len(assigned_items)

        return summary

    async def _notify_review_submission(self, review_item: ReviewItem):
        """Notify reviewers of new review item"""

        if not self.email_enabled:
            return

        subject = f"Kalki Review Required: {review_item.title}"

        # Create message for each assigned reviewer
        for reviewer_id in review_item.assigned_reviewers:
            reviewer = self.reviewers.get(reviewer_id)
            if not reviewer:
                continue

            body = f"""
Dear {reviewer['name']},

A new item requires your review:

Title: {review_item.title}
Priority: {review_item.priority.value.upper()}
Due Date: {review_item.due_date}
Category: {review_item.category}

Description:
{review_item.description}

Review Criteria:
{json.dumps(review_item.review_criteria, indent=2)}

Please review this item and submit your decision through the Kalki review interface.

Best regards,
Kalki Review System
"""

            await self._send_email(reviewer["email"], subject, body)

    async def _send_critical_notification(self, review_item: ReviewItem):
        """Send immediate notification for critical items"""

        if not self.email_enabled:
            return

        subject = f"URGENT: Critical Kalki Review Required - {review_item.title}"

        # Notify all reviewers and executives
        recipients = []
        for reviewer in self.reviewers.values():
            if ReviewerRole.EXECUTIVE_SPONSOR in reviewer["roles"] or review_item.priority == ReviewPriority.CRITICAL:
                recipients.append(reviewer["email"])

        body = f"""
URGENT REVIEW REQUIRED

Title: {review_item.title}
Priority: CRITICAL
Created: {review_item.created_timestamp}
Due: IMMEDIATE REVIEW

Description:
{review_item.description}

This item requires immediate attention. Please review ASAP.
"""

        for email in recipients:
            await self._send_email(email, subject, body)

    async def _notify_decision_submitted(self, review_item: ReviewItem, decision: ReviewDecision):
        """Notify stakeholders of review decision"""

        if not self.email_enabled:
            return

        subject = f"Kalki Review Decision: {review_item.title}"

        # Notify all assigned reviewers and executives
        recipients = set()
        for reviewer_id in review_item.assigned_reviewers:
            reviewer = self.reviewers.get(reviewer_id)
            if reviewer:
                recipients.add(reviewer["email"])

        # Add executives for final decisions
        if review_item.status in [ReviewStatus.APPROVED, ReviewStatus.REJECTED, ReviewStatus.ESCALATED]:
            for reviewer in self.reviewers.values():
                if ReviewerRole.EXECUTIVE_SPONSOR in reviewer["roles"]:
                    recipients.add(reviewer["email"])

        reviewer_name = self.reviewers.get(decision.reviewer_id, {}).get("name", decision.reviewer_id)

        body = f"""
Review Decision Submitted

Item: {review_item.title}
Reviewer: {reviewer_name}
Decision: {decision.decision.value.upper()}
Confidence: {decision.confidence_level:.2f}

Rationale:
{decision.rationale}

{"Required Changes:" + chr(10) + chr(10).join(decision.required_changes) if decision.required_changes else ""}

{"Recommendations:" + chr(10) + chr(10).join(decision.recommendations) if decision.recommendations else ""}
"""

        for email in recipients:
            await self._send_email(email, subject, body)

    async def _send_weekly_review_notifications(self, summary: Dict[str, Any]):
        """Send weekly review summary notifications"""

        if not self.email_enabled:
            return

        subject = f"Weekly Kalki Review Summary - {summary['cycle_id']}"

        body = f"""
Weekly Human Review Cadence Summary

Cycle: {summary['cycle_id']}
Generated: {summary['generated_at']}

Pending Reviews: {summary['total_pending']}
Critical Items: {summary['critical_items']}
High Priority: {summary['high_priority']}

Category Breakdown:
{chr(10).join(f"- {cat}: {count}" for cat, count in summary['categories'].items())}

Overdue Items: {len(summary['overdue_items'])}
"""

        if summary['overdue_items']:
            body += "\nOverdue Items:\n"
            for item in summary['overdue_items'][:5]:  # Show first 5
                body += f"- {item['title']} ({item['days_overdue']} days overdue)\n"

        # Send to all reviewers
        recipients = [r["email"] for r in self.reviewers.values()]

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

    def get_review_status(self) -> Dict[str, Any]:
        """Get current review system status"""

        pending_items = len([i for i in self.review_queue.values() if i.status == ReviewStatus.PENDING])
        in_review_items = len([i for i in self.review_queue.values() if i.status == ReviewStatus.IN_REVIEW])
        completed_items = len([i for i in self.review_queue.values() if i.status != ReviewStatus.PENDING and i.status != ReviewStatus.IN_REVIEW])

        overdue_items = []
        now = datetime.now()
        for item in self.review_queue.values():
            if item.status in [ReviewStatus.PENDING, ReviewStatus.IN_REVIEW]:
                due_date = datetime.fromisoformat(item.due_date)
                if now > due_date:
                    overdue_items.append(item.item_id)

        return {
            "total_items": len(self.review_queue),
            "pending": pending_items,
            "in_review": in_review_items,
            "completed": completed_items,
            "overdue": len(overdue_items),
            "reviewers_active": len(self.reviewers),
            "current_cycle": self._get_current_review_cycle().cycle_id
        }

    def _load_review_data(self):
        """Load review data from files"""
        try:
            os.makedirs(self.data_dir, exist_ok=True)

            if os.path.exists(self.queue_file):
                with open(self.queue_file, 'r') as f:
                    queue_data = json.load(f)
                    for item_data in queue_data.values():
                        item = ReviewItem(**item_data)
                        self.review_queue[item.item_id] = item

            if os.path.exists(self.decisions_file):
                with open(self.decisions_file, 'r') as f:
                    self.review_decisions = json.load(f)

            if os.path.exists(self.cycles_file):
                with open(self.cycles_file, 'r') as f:
                    cycles_data = json.load(f)
                    for cycle_data in cycles_data.values():
                        cycle = ReviewCycle(**cycle_data)
                        self.review_cycles[cycle.cycle_id] = cycle

        except Exception as e:
            logger.warning(f"Failed to load review data: {e}")

    def _save_review_data(self):
        """Save review data to files"""
        try:
            # Save queue
            queue_data = {item_id: asdict(item) for item_id, item in self.review_queue.items()}
            with open(self.queue_file, 'w') as f:
                json.dump(queue_data, f, indent=2)

            # Save decisions
            with open(self.decisions_file, 'w') as f:
                json.dump(self.review_decisions, f, indent=2)

            # Save cycles
            cycles_data = {cycle_id: asdict(cycle) for cycle_id, cycle in self.review_cycles.items()}
            with open(self.cycles_file, 'w') as f:
                json.dump(cycles_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save review data: {e}")

# Global human review cadence instance
_human_review_cadence = None

def get_human_review_cadence() -> HumanReviewCadence:
    """Get the global human review cadence instance"""
    global _human_review_cadence
    if _human_review_cadence is None:
        _human_review_cadence = HumanReviewCadence()
    return _human_review_cadence

# Convenience functions for integration
def submit_evolution_recommendation_for_review(recommendation_data: Dict[str, Any]) -> str:
    """Submit an evolution recommendation for human review"""
    cadence = get_human_review_cadence()

    priority = ReviewPriority.HIGH if recommendation_data.get("risk_level") == "high" else ReviewPriority.MEDIUM

    return cadence.submit_for_review(
        title=f"Evolution Recommendation: {recommendation_data.get('type', 'Unknown')}",
        description=recommendation_data.get('description', 'Evolution recommendation requiring review'),
        category="evolution_recommendation",
        priority=priority,
        review_criteria={
            "impact_assessment": True,
            "ethical_considerations": True,
            "safety_implications": True,
            "rollback_feasibility": True
        },
        metadata=recommendation_data
    )