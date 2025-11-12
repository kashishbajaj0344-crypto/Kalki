"""
Construction Journey Manager
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Manages the end-to-end construction project journey:
- Stage assessment (Discovery → Occupancy)
- Progress tracking
- Milestone management
- Blocker detection
- Next action recommendations
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ProjectStage:
    """Represents a stage in the construction journey"""
    name: str
    order: int
    key_milestones: List[str]
    typical_duration_weeks: int
    description: str
    required_deliverables: List[str]


@dataclass
class Milestone:
    """Represents a milestone within a stage"""
    id: str
    name: str
    stage: str
    completed: bool = False
    completion_date: Optional[datetime] = None
    notes: str = ""


class ConstructionJourneyManager:
    """
    Manages construction project journey from idea → completion.
    
    Integrates with:
    - KALKI's LLM for intelligent stage assessment
    - KALKI's consciousness for reasoning about progress
    - KALKI's meta-learning for timeline estimation
    """
    
    # Define the 12 stages of construction journey
    STAGES = [
        ProjectStage(
            name="discovery",
            order=1,
            key_milestones=[
                "Project vision defined",
                "Budget range established",
                "Property assessed"
            ],
            typical_duration_weeks=2,
            description="Initial exploration and feasibility",
            required_deliverables=["Vision statement", "Budget estimate"]
        ),
        ProjectStage(
            name="design",
            order=2,
            key_milestones=[
                "Architect hired",
                "Schematic design complete",
                "Design development complete",
                "Construction documents complete"
            ],
            typical_duration_weeks=12,
            description="Architectural design and engineering",
            required_deliverables=["Stamped architectural plans", "Structural calculations", "Energy calcs"]
        ),
        ProjectStage(
            name="permitting",
            order=3,
            key_milestones=[
                "Permit application submitted",
                "Plan check corrections submitted",
                "Permit approved"
            ],
            typical_duration_weeks=8,
            description="Building permit approval",
            required_deliverables=["Building permit", "Approved plans"]
        ),
        ProjectStage(
            name="pre_construction",
            order=4,
            key_milestones=[
                "Contractor selected",
                "Contract signed",
                "Insurance verified",
                "Materials ordered"
            ],
            typical_duration_weeks=3,
            description="Pre-construction preparation",
            required_deliverables=["Construction contract", "Insurance certificates", "Material orders"]
        ),
        ProjectStage(
            name="foundation",
            order=5,
            key_milestones=[
                "Site excavation complete",
                "Forms installed",
                "Rebar inspected",
                "Concrete poured",
                "Foundation inspection passed"
            ],
            typical_duration_weeks=3,
            description="Foundation construction",
            required_deliverables=["Foundation inspection report"]
        ),
        ProjectStage(
            name="framing",
            order=6,
            key_milestones=[
                "Floor framing complete",
                "Wall framing complete",
                "Roof framing complete",
                "Framing inspection passed"
            ],
            typical_duration_weeks=4,
            description="Structural framing",
            required_deliverables=["Framing inspection report"]
        ),
        ProjectStage(
            name="rough_ins",
            order=7,
            key_milestones=[
                "Plumbing rough-in complete",
                "Electrical rough-in complete",
                "HVAC rough-in complete",
                "Rough inspections passed"
            ],
            typical_duration_weeks=3,
            description="Mechanical, electrical, plumbing installation",
            required_deliverables=["Rough inspection reports"]
        ),
        ProjectStage(
            name="insulation",
            order=8,
            key_milestones=[
                "Insulation installed",
                "Insulation inspection passed",
                "Drywall hung"
            ],
            typical_duration_weeks=2,
            description="Insulation and drywall",
            required_deliverables=["Insulation inspection report"]
        ),
        ProjectStage(
            name="finishes",
            order=9,
            key_milestones=[
                "Drywall finished",
                "Interior paint complete",
                "Flooring installed",
                "Cabinets installed",
                "Countertops installed",
                "Fixtures installed"
            ],
            typical_duration_weeks=6,
            description="Interior finishes",
            required_deliverables=["Punch list"]
        ),
        ProjectStage(
            name="exterior",
            order=10,
            key_milestones=[
                "Siding/exterior complete",
                "Windows/doors installed",
                "Exterior paint complete",
                "Landscaping complete"
            ],
            typical_duration_weeks=4,
            description="Exterior completion",
            required_deliverables=["Exterior photos"]
        ),
        ProjectStage(
            name="final_inspection",
            order=11,
            key_milestones=[
                "Final building inspection",
                "Final electrical inspection",
                "Final plumbing inspection",
                "Certificate of Occupancy issued"
            ],
            typical_duration_weeks=2,
            description="Final inspections and approvals",
            required_deliverables=["Certificate of Occupancy"]
        ),
        ProjectStage(
            name="occupancy",
            order=12,
            key_milestones=[
                "Utilities connected",
                "Final walkthrough complete",
                "Keys handed over"
            ],
            typical_duration_weeks=1,
            description="Move-in and project closeout",
            required_deliverables=["Warranty documentation", "As-built drawings"]
        )
    ]
    
    
    def __init__(self, llm_engine, consciousness=None, meta_learning=None):
        """
        Initialize journey manager.
        
        Args:
            llm_engine: KALKI's LLM engine
            consciousness: KALKI's consciousness engine (optional)
            meta_learning: KALKI's meta-learning system (optional)
        """
        self.llm = llm_engine
        self.consciousness = consciousness
        self.meta_learning = meta_learning
        self.logger = logging.getLogger(__name__)
        
        # Project tracking
        self.projects: Dict[str, Dict[str, Any]] = {}
    
    
    async def assess_current_stage(
        self,
        project_id: str,
        user_responses: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Assess where user is in their construction journey.
        
        Args:
            project_id: Unique project identifier
            user_responses: User's answers to assessment questions
        
        Returns:
            {
                'current_stage': str,
                'stage_order': int,
                'confidence': float,
                'completed_milestones': List[str],
                'next_milestones': List[str],
                'estimated_progress': float (0.0 to 1.0)
            }
        """
        self.logger.info(f"Assessing current stage for project: {project_id}")
        
        # Use LLM to analyze user responses
        assessment_prompt = f"""Analyze where this user is in their construction project journey:

User responses:
{self._format_responses(user_responses)}

Construction journey stages:
{self._format_stages()}

Determine:
1. Current stage (discovery, design, permitting, etc.)
2. Completed milestones
3. Progress percentage (0-100)
4. Confidence in assessment (0-100)

Format as JSON."""
        
        analysis = await self.llm.generate(
            prompt=assessment_prompt,
            task='construction_reasoning',
            max_tokens=600
        )
        
        # Parse LLM response
        current_stage_name = self._extract_stage(analysis['text'])
        completed_milestones = self._extract_milestones(analysis['text'])
        progress = self._extract_progress(analysis['text'])
        confidence = self._extract_confidence(analysis['text'])
        
        # Get stage details
        current_stage = self._get_stage(current_stage_name)
        
        # Store assessment
        if project_id not in self.projects:
            self.projects[project_id] = {
                'milestones': {},
                'blockers': [],
                'history': []
            }
        
        self.projects[project_id]['current_stage'] = current_stage_name
        self.projects[project_id]['last_assessment'] = datetime.now()
        
        # Mark completed milestones
        for milestone in completed_milestones:
            self.projects[project_id]['milestones'][milestone] = {
                'completed': True,
                'date': datetime.now()
            }
        
        # Get next milestones
        next_milestones = self._get_next_milestones(current_stage_name, completed_milestones)
        
        result = {
            'current_stage': current_stage_name,
            'stage_order': current_stage.order if current_stage else 0,
            'stage_description': current_stage.description if current_stage else '',
            'confidence': confidence,
            'completed_milestones': completed_milestones,
            'next_milestones': next_milestones[:3],  # Top 3 next actions
            'estimated_progress': progress / 100.0,
            'total_stages': len(self.STAGES),
            'stages_remaining': len(self.STAGES) - (current_stage.order if current_stage else 0)
        }
        
        self.logger.info(f"Stage assessed: {current_stage_name} ({progress}% progress)")
        
        return result
    
    
    def get_current_milestone(self, project_id: str) -> str:
        """Get current milestone for project"""
        project = self.projects.get(project_id)
        if not project:
            return "Start project assessment"
        
        current_stage = project.get('current_stage', 'discovery')
        stage = self._get_stage(current_stage)
        
        if not stage:
            return "Continue project"
        
        # Find first incomplete milestone
        for milestone in stage.key_milestones:
            milestone_id = self._milestone_id(current_stage, milestone)
            if not project['milestones'].get(milestone_id, {}).get('completed'):
                return milestone
        
        # All milestones complete - move to next stage
        next_stage = self._get_next_stage(current_stage)
        if next_stage:
            return f"Begin {next_stage.name.replace('_', ' ').title()}"
        
        return "Project complete!"
    
    
    def get_next_milestone(self, project_id: str) -> str:
        """Get next milestone after current"""
        project = self.projects.get(project_id)
        if not project:
            return "Continue assessment"
        
        current_stage = project.get('current_stage', 'discovery')
        stage = self._get_stage(current_stage)
        
        if not stage:
            return "Continue project"
        
        # Find first incomplete milestone, then return the one after
        milestones = stage.key_milestones
        found_current = False
        
        for milestone in milestones:
            milestone_id = self._milestone_id(current_stage, milestone)
            is_complete = project['milestones'].get(milestone_id, {}).get('completed')
            
            if found_current:
                return milestone
            
            if not is_complete:
                found_current = True
        
        # No more milestones in current stage
        next_stage = self._get_next_stage(current_stage)
        if next_stage and next_stage.key_milestones:
            return next_stage.key_milestones[0]
        
        return "Project completion"
    
    
    async def mark_milestone_complete(
        self,
        project_id: str,
        milestone_name: str
    ) -> Dict[str, Any]:
        """Mark a milestone as complete"""
        if project_id not in self.projects:
            self.projects[project_id] = {
                'milestones': {},
                'blockers': [],
                'history': []
            }
        
        project = self.projects[project_id]
        current_stage = project.get('current_stage', 'discovery')
        milestone_id = self._milestone_id(current_stage, milestone_name)
        
        project['milestones'][milestone_id] = {
            'completed': True,
            'date': datetime.now(),
            'name': milestone_name
        }
        
        project['history'].append({
            'event': 'milestone_completed',
            'milestone': milestone_name,
            'date': datetime.now()
        })
        
        self.logger.info(f"Milestone completed: {milestone_name}")
        
        return {
            'status': 'completed',
            'milestone': milestone_name,
            'next_milestone': self.get_next_milestone(project_id)
        }
    
    
    def detect_blockers(
        self,
        project_id: str,
        current_state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect potential blockers to progress"""
        blockers = []
        
        project = self.projects.get(project_id)
        if not project:
            return blockers
        
        current_stage = project.get('current_stage', 'discovery')
        
        # Check for common blockers by stage
        if current_stage == 'design':
            if not current_state.get('architect_hired'):
                blockers.append({
                    'type': 'missing_professional',
                    'severity': 'high',
                    'description': 'Architect not yet hired',
                    'impact': 'Cannot proceed with design',
                    'resolution': 'Interview and hire licensed architect'
                })
        
        if current_stage == 'permitting':
            if not current_state.get('plans_complete'):
                blockers.append({
                    'type': 'incomplete_deliverable',
                    'severity': 'high',
                    'description': 'Construction documents not complete',
                    'impact': 'Cannot submit permit application',
                    'resolution': 'Complete architectural plans and engineering'
                })
        
        if current_stage == 'pre_construction':
            if not current_state.get('permit_approved'):
                blockers.append({
                    'type': 'regulatory',
                    'severity': 'critical',
                    'description': 'Building permit not approved',
                    'impact': 'Cannot legally begin construction',
                    'resolution': 'Wait for permit approval or address plan check issues'
                })
        
        # Check timeline blockers
        last_assessment = project.get('last_assessment')
        if last_assessment:
            days_since_update = (datetime.now() - last_assessment).days
            if days_since_update > 30:
                blockers.append({
                    'type': 'stalled_project',
                    'severity': 'moderate',
                    'description': f'No progress update in {days_since_update} days',
                    'impact': 'Project may be stalled',
                    'resolution': 'Review current status and identify obstacles'
                })
        
        return blockers
    
    
    # Helper methods
    
    def _get_stage(self, stage_name: str) -> Optional[ProjectStage]:
        """Get stage by name"""
        for stage in self.STAGES:
            if stage.name == stage_name:
                return stage
        return None
    
    
    def _get_next_stage(self, current_stage_name: str) -> Optional[ProjectStage]:
        """Get next stage after current"""
        current = self._get_stage(current_stage_name)
        if not current:
            return None
        
        for stage in self.STAGES:
            if stage.order == current.order + 1:
                return stage
        
        return None
    
    
    def _get_next_milestones(
        self,
        stage_name: str,
        completed: List[str]
    ) -> List[str]:
        """Get next incomplete milestones"""
        stage = self._get_stage(stage_name)
        if not stage:
            return []
        
        next_milestones = []
        for milestone in stage.key_milestones:
            if milestone not in completed:
                next_milestones.append(milestone)
        
        return next_milestones
    
    
    def _milestone_id(self, stage: str, milestone: str) -> str:
        """Generate unique milestone ID"""
        return f"{stage}_{milestone.lower().replace(' ', '_')}"
    
    
    def _format_responses(self, responses: Dict[str, str]) -> str:
        """Format user responses for LLM"""
        formatted = []
        for question, answer in responses.items():
            formatted.append(f"Q: {question}\nA: {answer}")
        return "\n\n".join(formatted)
    
    
    def _format_stages(self) -> str:
        """Format stages for LLM"""
        formatted = []
        for stage in self.STAGES:
            formatted.append(
                f"{stage.order}. {stage.name.replace('_', ' ').title()}: {stage.description}"
            )
        return "\n".join(formatted)
    
    
    def _extract_stage(self, text: str) -> str:
        """Extract stage name from LLM response"""
        text_lower = text.lower()
        
        # Try to find stage mention
        for stage in self.STAGES:
            if stage.name in text_lower or stage.name.replace('_', ' ') in text_lower:
                return stage.name
        
        # Default to discovery
        return "discovery"
    
    
    def _extract_milestones(self, text: str) -> List[str]:
        """Extract completed milestones from LLM response"""
        milestones = []
        
        # Look for milestones in all stages
        for stage in self.STAGES:
            for milestone in stage.key_milestones:
                if milestone.lower() in text.lower():
                    milestones.append(milestone)
        
        return milestones
    
    
    def _extract_progress(self, text: str) -> float:
        """Extract progress percentage from LLM response"""
        import re
        
        # Look for percentage
        match = re.search(r'(\d+)%', text)
        if match:
            return float(match.group(1))
        
        # Default based on stage mention
        for stage in self.STAGES:
            if stage.name in text.lower():
                return (stage.order / len(self.STAGES)) * 100
        
        return 10.0  # Default: early stage
    
    
    def _extract_confidence(self, text: str) -> float:
        """Extract confidence from LLM response"""
        import re
        
        # Look for confidence mention
        match = re.search(r'confidence["\s:]+(\d+)', text, re.IGNORECASE)
        if match:
            return float(match.group(1)) / 100.0
        
        return 0.7  # Default moderate confidence
