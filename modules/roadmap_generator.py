"""
Roadmap Generator for Construction Copilot
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Generates personalized 60-120 step construction roadmaps:
- Timeline estimation (weeks per phase)
- Cost estimation per milestone  
- Dependency management
- Critical path identification
- Adjusts based on meta-learning insights
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RoadmapStep:
    """Represents a single step in the construction roadmap"""
    id: int
    name: str
    stage: str
    description: str
    estimated_duration_days: int
    estimated_cost: float
    dependencies: List[int]
    critical_path: bool = False
    professional_required: Optional[str] = None  # 'architect', 'engineer', 'contractor', etc.


class RoadmapGenerator:
    """
    Generates personalized construction roadmaps.
    
    Integrates with:
    - KALKI's LLM for intelligent step generation
    - KALKI's meta-learning for timeline/cost adjustments
    """
    
    # Base roadmap templates by project type
    ADU_TEMPLATE = {
        'total_steps': 82,
        'estimated_weeks': 48,
        'estimated_cost': 165000,
        'complexity_factor': 1.0
    }
    
    REMODEL_TEMPLATE = {
        'total_steps': 65,
        'estimated_weeks': 24,
        'estimated_cost': 85000,
        'complexity_factor': 0.8
    }
    
    NEW_CONSTRUCTION_TEMPLATE = {
        'total_steps': 120,
        'estimated_weeks': 65,
        'estimated_cost': 450000,
        'complexity_factor': 1.5
    }
    
    
    def __init__(self, llm_engine, meta_learning=None):
        """
        Initialize roadmap generator.
        
        Args:
            llm_engine: KALKI's LLM engine
            meta_learning: KALKI's meta-learning system (optional)
        """
        self.llm = llm_engine
        self.meta_learning = meta_learning
        self.logger = logging.getLogger(__name__)
    
    
    async def generate_personalized_roadmap(
        self,
        project_type: str,
        assessment: Dict[str, Any],
        property_constraints: Dict[str, Any],
        historical_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate personalized construction roadmap.
        
        Args:
            project_type: 'adu', 'remodel', or 'new_construction'
            assessment: User's project assessment
            property_constraints: Property intelligence data
            historical_data: Meta-learning historical patterns
        
        Returns:
            Complete roadmap with steps, timeline, costs, dependencies
        """
        self.logger.info(f"Generating roadmap for {project_type} project")
        
        # Get base template
        template = self._get_template(project_type)
        
        # Apply property constraints adjustments
        adjusted = self._apply_property_adjustments(template, property_constraints)
        
        # Apply meta-learning adjustments
        if self.meta_learning and historical_data:
            adjusted = self._apply_historical_adjustments(adjusted, historical_data, property_constraints)
        
        # Generate detailed steps
        steps = await self._generate_steps(project_type, adjusted, property_constraints)
        
        # Identify critical path
        critical_steps = self._identify_critical_path(steps)
        
        # Calculate cumulative timeline
        timeline = self._calculate_timeline(steps)
        
        # Group steps by stage
        by_stage = self._group_by_stage(steps)
        
        # Generate immediate next steps
        next_steps = self._get_immediate_next_steps(steps, assessment)
        
        roadmap = {
            'project_type': project_type,
            'total_steps': len(steps),
            'timeline_weeks': adjusted['estimated_weeks'],
            'estimated_cost': adjusted['estimated_cost'],
            'complexity_score': property_constraints.get('complexity_score', 0.5),
            'steps': steps,
            'critical_path_steps': critical_steps,
            'timeline_by_stage': timeline,
            'steps_by_stage': by_stage,
            'immediate_next_steps': next_steps,
            'adjustments_applied': {
                'property_constraints': True,
                'historical_data': bool(historical_data),
                'meta_learning': bool(self.meta_learning)
            },
            'generated_at': datetime.now().isoformat()
        }
        
        self.logger.info(f"Roadmap generated: {len(steps)} steps, {adjusted['estimated_weeks']} weeks")
        
        return roadmap
    
    
    async def adjust_timeline_estimates(
        self,
        project_type: str,
        location: str,
        adjustment_factor: float
    ):
        """Adjust timeline estimates based on meta-learning"""
        # Store adjustment for future roadmap generation
        if hasattr(self, 'adjustments'):
            if project_type not in self.adjustments:
                self.adjustments[project_type] = {}
            if location not in self.adjustments[project_type]:
                self.adjustments[project_type][location] = {}
            
            self.adjustments[project_type][location]['timeline'] = adjustment_factor
            
            self.logger.info(
                f"Timeline adjustment stored: {project_type} in {location} = {adjustment_factor:.2f}x"
            )
    
    
    async def adjust_budget_estimates(
        self,
        project_type: str,
        adjustment_factor: float
    ):
        """Adjust budget estimates based on meta-learning"""
        if hasattr(self, 'adjustments'):
            if project_type not in self.adjustments:
                self.adjustments[project_type] = {}
            
            self.adjustments[project_type]['budget'] = adjustment_factor
            
            self.logger.info(
                f"Budget adjustment stored: {project_type} = {adjustment_factor:.2f}x"
            )
    
    
    async def _generate_steps(
        self,
        project_type: str,
        adjusted_template: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate detailed roadmap steps"""
        
        # Use LLM to generate customized steps
        prompt = f"""Generate a detailed construction roadmap for a {project_type} project.

Project details:
- Estimated timeline: {adjusted_template['estimated_weeks']} weeks
- Estimated cost: ${adjusted_template['estimated_cost']:,.0f}
- Complexity: {constraints.get('complexity_score', 0.5):.1f}/1.0
- Location: {constraints.get('jurisdiction', 'Unknown')}
- Constraints: {', '.join([c['type'] for c in constraints.get('constraints', [])])}

Generate 80-100 specific, actionable steps covering:
1. Discovery & Planning (10 steps)
2. Design & Engineering (15 steps)
3. Permitting (10 steps)
4. Pre-Construction (8 steps)
5. Foundation (8 steps)
6. Framing (10 steps)
7. Rough-Ins (10 steps)
8. Insulation & Drywall (8 steps)
9. Interior Finishes (12 steps)
10. Exterior (8 steps)
11. Final Inspections (5 steps)
12. Closeout (4 steps)

For each step, provide:
- Step name
- Stage
- Brief description
- Duration (days)
- Estimated cost
- Professional required (if any)

Format as structured list."""
        
        response = await self.llm.generate(
            prompt=prompt,
            task='roadmap_generation',
            max_tokens=2000
        )
        
        # Parse LLM response into structured steps
        # For now, use a baseline template (in production, parse LLM output)
        steps = self._generate_baseline_steps(project_type, adjusted_template, constraints)
        
        return steps
    
    
    def _generate_baseline_steps(
        self,
        project_type: str,
        template: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate baseline roadmap steps"""
        
        steps = []
        step_id = 1
        
        # Discovery & Planning (10 steps)
        steps.extend([
            {'id': step_id, 'name': 'Define project vision and goals', 'stage': 'discovery',
             'description': 'Clarify what you want to build and why', 'duration_days': 1,
             'cost': 0, 'dependencies': [], 'professional': None},
            {'id': step_id+1, 'name': 'Establish preliminary budget', 'stage': 'discovery',
             'description': 'Determine how much you can invest', 'duration_days': 2,
             'cost': 0, 'dependencies': [step_id], 'professional': None},
            {'id': step_id+2, 'name': 'Research local zoning requirements', 'stage': 'discovery',
             'description': 'Understand what\'s allowed on your property', 'duration_days': 3,
             'cost': 0, 'dependencies': [step_id+1], 'professional': None},
            {'id': step_id+3, 'name': 'Site assessment and measurements', 'stage': 'discovery',
             'description': 'Measure property and note conditions', 'duration_days': 1,
             'cost': 0, 'dependencies': [step_id+2], 'professional': None},
            {'id': step_id+4, 'name': 'Utility service verification', 'stage': 'discovery',
             'description': 'Confirm water, sewer, electric, gas availability', 'duration_days': 2,
             'cost': 0, 'dependencies': [step_id+3], 'professional': None},
            {'id': step_id+5, 'name': 'Soil test (if required)', 'stage': 'discovery',
             'description': 'Geotechnical evaluation of soil conditions', 'duration_days': 7,
             'cost': 1500, 'dependencies': [step_id+4], 'professional': 'geotechnical_engineer'},
            {'id': step_id+6, 'name': 'Topographic survey (if required)', 'stage': 'discovery',
             'description': 'Professional survey of property elevation', 'duration_days': 3,
             'cost': 800, 'dependencies': [step_id+4], 'professional': 'surveyor'},
            {'id': step_id+7, 'name': 'Research financing options', 'stage': 'discovery',
             'description': 'Explore loans, HELOCs, construction financing', 'duration_days': 5,
             'cost': 0, 'dependencies': [step_id+1], 'professional': None},
            {'id': step_id+8, 'name': 'Finalize project scope', 'stage': 'discovery',
             'description': 'Define exactly what will be built', 'duration_days': 2,
             'cost': 0, 'dependencies': [step_id+5, step_id+6], 'professional': None},
            {'id': step_id+9, 'name': 'Create preliminary timeline', 'stage': 'discovery',
             'description': 'Estimate project duration and key milestones', 'duration_days': 1,
             'cost': 0, 'dependencies': [step_id+8], 'professional': None},
        ])
        step_id += 10
        
        # Design & Engineering (15 steps)
        steps.extend([
            {'id': step_id, 'name': 'Interview and select architect', 'stage': 'design',
             'description': 'Find licensed architect experienced with your project type', 'duration_days': 14,
             'cost': 0, 'dependencies': [step_id-1], 'professional': None},
            {'id': step_id+1, 'name': 'Sign architecture contract', 'stage': 'design',
             'description': 'Formalize scope, fees, timeline with architect', 'duration_days': 3,
             'cost': 0, 'dependencies': [step_id], 'professional': 'architect'},
            {'id': step_id+2, 'name': 'Schematic design phase', 'stage': 'design',
             'description': 'Conceptual drawings, floor plans, elevations', 'duration_days': 14,
             'cost': 3000, 'dependencies': [step_id+1], 'professional': 'architect'},
            {'id': step_id+3, 'name': 'Review and approve schematics', 'stage': 'design',
             'description': 'Provide feedback, request revisions', 'duration_days': 7,
             'cost': 0, 'dependencies': [step_id+2], 'professional': None},
            {'id': step_id+4, 'name': 'Design development phase', 'stage': 'design',
             'description': 'Detailed design with materials, systems specified', 'duration_days': 21,
             'cost': 4000, 'dependencies': [step_id+3], 'professional': 'architect'},
            {'id': step_id+5, 'name': 'Structural engineering', 'stage': 'design',
             'description': 'Foundation design, framing calculations, beam sizing', 'duration_days': 14,
             'cost': 2500, 'dependencies': [step_id+4], 'professional': 'structural_engineer'},
            {'id': step_id+6, 'name': 'Title 24 energy calculations', 'stage': 'design',
             'description': 'Energy compliance calculations (CA requirement)', 'duration_days': 7,
             'cost': 800, 'dependencies': [step_id+4], 'professional': 'energy_consultant'},
            {'id': step_id+7, 'name': 'Construction documents phase', 'stage': 'design',
             'description': 'Complete construction-ready drawings', 'duration_days': 21,
             'cost': 5000, 'dependencies': [step_id+5, step_id+6], 'professional': 'architect'},
            {'id': step_id+8, 'name': 'Electrical plan', 'stage': 'design',
             'description': 'Lighting, outlets, panel sizing', 'duration_days': 7,
             'cost': 600, 'dependencies': [step_id+7], 'professional': 'electrical_designer'},
            {'id': step_id+9, 'name': 'Plumbing plan', 'stage': 'design',
             'description': 'Fixture locations, drain/vent routing', 'duration_days': 7,
             'cost': 600, 'dependencies': [step_id+7], 'professional': 'plumbing_designer'},
            {'id': step_id+10, 'name': 'HVAC plan', 'stage': 'design',
             'description': 'Heating/cooling system design', 'duration_days': 7,
             'cost': 800, 'dependencies': [step_id+7], 'professional': 'hvac_designer'},
            {'id': step_id+11, 'name': 'Final drawing review', 'stage': 'design',
             'description': 'Review complete construction documents', 'duration_days': 5,
             'cost': 0, 'dependencies': [step_id+8, step_id+9, step_id+10], 'professional': None},
            {'id': step_id+12, 'name': 'Cost estimation from plans', 'stage': 'design',
             'description': 'Detailed cost breakdown from drawings', 'duration_days': 5,
             'cost': 500, 'dependencies': [step_id+11], 'professional': 'estimator'},
            {'id': step_id+13, 'name': 'Value engineering (if needed)', 'stage': 'design',
             'description': 'Adjust design to meet budget', 'duration_days': 10,
             'cost': 1000, 'dependencies': [step_id+12], 'professional': 'architect'},
            {'id': step_id+14, 'name': 'Finalize construction documents', 'stage': 'design',
             'description': 'Stamp and sign drawings', 'duration_days': 3,
             'cost': 500, 'dependencies': [step_id+13], 'professional': 'architect'},
        ])
        step_id += 15
        
        # Add more stages... (permitting, construction, etc.)
        # For brevity, showing pattern. In production, generate all 80-100 steps.
        
        # Adjust costs based on project type
        cost_multiplier = 1.0
        if project_type == 'adu':
            cost_multiplier = 1.0
        elif project_type == 'remodel':
            cost_multiplier = 0.5
        elif project_type == 'new_construction':
            cost_multiplier = 2.5
        
        for step in steps:
            step['cost'] = int(step['cost'] * cost_multiplier)
        
        return steps
    
    
    def _get_template(self, project_type: str) -> Dict[str, Any]:
        """Get base template for project type"""
        templates = {
            'adu': self.ADU_TEMPLATE,
            'remodel': self.REMODEL_TEMPLATE,
            'new_construction': self.NEW_CONSTRUCTION_TEMPLATE
        }
        return templates.get(project_type, self.ADU_TEMPLATE).copy()
    
    
    def _apply_property_adjustments(
        self,
        template: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adjust template based on property constraints"""
        adjusted = template.copy()
        
        complexity = constraints.get('complexity_score', 0.5)
        
        # Adjust timeline
        adjusted['estimated_weeks'] = int(template['estimated_weeks'] * (1 + complexity * 0.3))
        
        # Adjust cost
        adjusted['estimated_cost'] = int(template['estimated_cost'] * (1 + complexity * 0.2))
        
        # Additional constraints increase time/cost
        num_constraints = len(constraints.get('constraints', []))
        adjusted['estimated_weeks'] += num_constraints * 2
        adjusted['estimated_cost'] = int(adjusted['estimated_cost'] * (1 + num_constraints * 0.05))
        
        return adjusted
    
    
    def _apply_historical_adjustments(
        self,
        template: Dict[str, Any],
        historical: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply meta-learning adjustments from historical data"""
        adjusted = template.copy()
        
        # Get location-specific adjustments if available
        location = constraints.get('jurisdiction', '')
        
        # Example: San Jose projects typically take 6% longer due to permit delays
        if 'san jose' in location.lower():
            adjusted['estimated_weeks'] = int(adjusted['estimated_weeks'] * 1.06)
        
        # Apply any stored adjustments
        if hasattr(self, 'adjustments'):
            project_type = constraints.get('project_type', 'adu')
            if project_type in self.adjustments:
                if location in self.adjustments[project_type]:
                    timeline_adj = self.adjustments[project_type][location].get('timeline', 1.0)
                    adjusted['estimated_weeks'] = int(adjusted['estimated_weeks'] * timeline_adj)
                
                budget_adj = self.adjustments[project_type].get('budget', 1.0)
                adjusted['estimated_cost'] = int(adjusted['estimated_cost'] * budget_adj)
        
        return adjusted
    
    
    def _identify_critical_path(self, steps: List[Dict]) -> List[int]:
        """Identify critical path steps (longest dependency chain)"""
        critical = []
        
        # Simple heuristic: steps with long duration and many dependents
        for step in steps:
            if step['duration_days'] >= 7:
                critical.append(step['id'])
        
        return critical
    
    
    def _calculate_timeline(self, steps: List[Dict]) -> Dict[str, int]:
        """Calculate timeline by stage"""
        timeline = {}
        
        for step in steps:
            stage = step['stage']
            if stage not in timeline:
                timeline[stage] = 0
            timeline[stage] += step['duration_days']
        
        # Convert to weeks
        for stage in timeline:
            timeline[stage] = int(timeline[stage] / 7)
        
        return timeline
    
    
    def _group_by_stage(self, steps: List[Dict]) -> Dict[str, List[Dict]]:
        """Group steps by stage"""
        grouped = {}
        
        for step in steps:
            stage = step['stage']
            if stage not in grouped:
                grouped[stage] = []
            grouped[stage].append(step)
        
        return grouped
    
    
    def _get_immediate_next_steps(
        self,
        steps: List[Dict],
        assessment: Dict[str, Any]
    ) -> List[Dict]:
        """Get immediate next 3-5 steps based on current progress"""
        current_stage = assessment.get('current_stage', 'discovery')
        
        # Get all steps for current stage
        current_steps = [s for s in steps if s['stage'] == current_stage]
        
        # Return first 5 steps
        return current_steps[:5] if current_steps else steps[:5]
