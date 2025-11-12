"""
Aerospace Domain

Specialized domain for aerospace projects including:
- UAVs and drones
- Fixed-wing aircraft
- Multirotor vehicles
- Aerodynamics and flight dynamics
- Propulsion systems
"""

from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from modules.domains.base_domain import (
    BaseDomain,
    ProjectStateMachine,
    ProjectPhase,
    ValidationResult,
    KnowledgeExtractor,
    ComplexityScore,
    DeliverableSpec
)


class AerospacePhase(str, Enum):
    """Aerospace project phases"""
    CONCEPT = "concept"
    REQUIREMENTS = "requirements"
    CONCEPTUAL_DESIGN = "conceptual_design"
    DETAILED_DESIGN = "detailed_design"
    AERODYNAMIC_ANALYSIS = "aerodynamic_analysis"
    STRUCTURAL_ANALYSIS = "structural_analysis"
    PROTOTYPING = "prototyping"
    TESTING = "testing"
    CERTIFICATION = "certification"


class AircraftType(str, Enum):
    """Aircraft types"""
    FIXED_WING = "fixed_wing"
    MULTIROTOR = "multirotor"
    VTOL = "vtol"
    HELICOPTER = "helicopter"
    GLIDER = "glider"
    HYBRID = "hybrid"


class AerospaceProjectStateMachine(ProjectStateMachine):
    """State machine for aerospace projects"""
    
    def __init__(self, project_id: str, description: str):
        super().__init__(project_id, description, "aerospace")
        self.current_phase = AerospacePhase.CONCEPT
        
        # Aircraft specifications
        self.aircraft_type = None
        self.wingspan_m = 0.0
        self.mtow_kg = 0.0  # Maximum takeoff weight
        self.cruise_speed_ms = 0.0
        self.endurance_min = 0
        self.range_km = 0.0
        
        # Propulsion
        self.propulsion_type = None  # electric, combustion, hybrid
        self.num_motors = 0
        
        # Flight control
        self.autopilot = None  # pixhawk, ardupilot, custom
        self.flight_modes = []
        
        # Budget tracking
        self.budget = {
            "estimated_total": 0,
            "actual_spent": 0,
            "by_category": {
                "airframe": 0,
                "propulsion": 0,
                "avionics": 0,
                "batteries": 0,
                "testing": 0,
                "certification": 0
            }
        }
        
        # Milestones
        self.milestones = {
            AerospacePhase.CONCEPT: [
                {"name": "Define mission requirements", "complete": False},
                {"name": "Select aircraft configuration", "complete": False},
                {"name": "Estimate performance parameters", "complete": False}
            ],
            AerospacePhase.REQUIREMENTS: [
                {"name": "Specify flight envelope", "complete": False},
                {"name": "Define payload requirements", "complete": False},
                {"name": "Set regulatory compliance goals", "complete": False},
                {"name": "Establish safety requirements", "complete": False}
            ],
            AerospacePhase.CONCEPTUAL_DESIGN: [
                {"name": "Size aircraft components", "complete": False},
                {"name": "Design wing and control surfaces", "complete": False},
                {"name": "Select propulsion system", "complete": False},
                {"name": "Create initial CAD model", "complete": False}
            ],
            AerospacePhase.DETAILED_DESIGN: [
                {"name": "Finalize CAD models", "complete": False},
                {"name": "Design structural elements", "complete": False},
                {"name": "Plan electrical system", "complete": False},
                {"name": "Create manufacturing drawings", "complete": False}
            ],
            AerospacePhase.AERODYNAMIC_ANALYSIS: [
                {"name": "Perform CFD analysis", "complete": False},
                {"name": "Calculate lift and drag", "complete": False},
                {"name": "Analyze stability and control", "complete": False},
                {"name": "Optimize airfoil selection", "complete": False}
            ],
            AerospacePhase.STRUCTURAL_ANALYSIS: [
                {"name": "Perform FEA analysis", "complete": False},
                {"name": "Calculate load factors", "complete": False},
                {"name": "Verify structural integrity", "complete": False},
                {"name": "Optimize weight distribution", "complete": False}
            ],
            AerospacePhase.PROTOTYPING: [
                {"name": "Fabricate airframe", "complete": False},
                {"name": "Install propulsion system", "complete": False},
                {"name": "Integrate avionics", "complete": False},
                {"name": "Perform ground tests", "complete": False}
            ],
            AerospacePhase.TESTING: [
                {"name": "Conduct wind tunnel tests", "complete": False},
                {"name": "Perform flight tests", "complete": False},
                {"name": "Validate performance", "complete": False},
                {"name": "Test failure modes", "complete": False}
            ],
            AerospacePhase.CERTIFICATION: [
                {"name": "Prepare certification documentation", "complete": False},
                {"name": "Conduct regulatory review", "complete": False},
                {"name": "Complete compliance testing", "complete": False},
                {"name": "Obtain flight authorization", "complete": False}
            ]
        }
    
    async def validate_phase_complete(self, phase: AerospacePhase) -> ValidationResult:
        """Validate if phase requirements are met"""
        errors = []
        warnings = []
        suggestions = []
        
        phase_milestones = self.milestones.get(phase, [])
        completed = sum(1 for m in phase_milestones if m["complete"])
        total = len(phase_milestones)
        
        if phase == AerospacePhase.CONCEPT:
            if not self.aircraft_type:
                errors.append("Aircraft type not specified")
            if completed < total:
                warnings.append(f"Only {completed}/{total} concept milestones complete")
        
        elif phase == AerospacePhase.REQUIREMENTS:
            if self.mtow_kg == 0:
                warnings.append("Maximum takeoff weight not specified")
        
        elif phase == AerospacePhase.CONCEPTUAL_DESIGN:
            if self.wingspan_m == 0:
                warnings.append("Wingspan not specified")
            if not self.propulsion_type:
                errors.append("Propulsion type not selected")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    async def advance_phase(self, next_phase: AerospacePhase) -> bool:
        """Advance to next project phase"""
        self.phase_history.append({
            "from": self.current_phase,
            "to": next_phase,
            "timestamp": datetime.now().isoformat()
        })
        self.current_phase = next_phase
        return True
    
    def get_available_phases(self) -> list:
        """Get all aerospace phases"""
        return list(AerospacePhase)
    
    def mark_milestone_complete(self, milestone_name: str) -> bool:
        """Mark a milestone as complete"""
        phase_milestones = self.milestones.get(self.current_phase, [])
        for milestone in phase_milestones:
            if milestone["name"] == milestone_name:
                milestone["complete"] = True
                return True
        return False
    
    def get_phase_progress(self) -> Dict[str, Any]:
        """Get progress statistics for current phase"""
        phase_milestones = self.milestones.get(self.current_phase, [])
        total = len(phase_milestones)
        complete = sum(1 for m in phase_milestones if m["complete"])
        
        return {
            "phase": self.current_phase.value,
            "total_milestones": total,
            "completed_milestones": complete,
            "percent_complete": (complete / total * 100) if total > 0 else 0,
            "milestones": phase_milestones
        }
    
    async def get_contextual_help(self, user_query: str) -> str:
        """Provide help relevant to current aerospace phase"""
        phase_help = {
            AerospacePhase.CONCEPT: "Concept phase: Define mission and select aircraft configuration",
            AerospacePhase.REQUIREMENTS: "Requirements phase: Specify flight envelope, payload, regulations",
            AerospacePhase.CONCEPTUAL_DESIGN: "Conceptual design: Size components, design wing, select propulsion",
            AerospacePhase.DETAILED_DESIGN: "Detailed design: Finalize CAD, structural elements, electrical system",
            AerospacePhase.AERODYNAMIC_ANALYSIS: "Aero analysis: CFD, lift/drag calculations, stability",
            AerospacePhase.STRUCTURAL_ANALYSIS: "Structural analysis: FEA, load factors, weight optimization",
            AerospacePhase.PROTOTYPING: "Prototyping: Fabricate airframe, install systems, ground tests",
            AerospacePhase.TESTING: "Testing: Wind tunnel, flight tests, performance validation",
            AerospacePhase.CERTIFICATION: "Certification: Documentation, regulatory review, flight authorization"
        }
        
        progress = self.get_phase_progress()
        
        help_text = f"""
Current Phase: {self.current_phase.value}
Progress: {progress['completed_milestones']}/{progress['total_milestones']} milestones ({progress['percent_complete']:.1f}%)

{phase_help.get(self.current_phase, 'Aerospace development in progress')}

Pending Milestones:
"""
        
        for milestone in progress['milestones']:
            if not milestone['complete']:
                help_text += f"  • {milestone['name']}\n"
        
        return help_text.strip()


class AerospaceDomain(BaseDomain):
    """Aerospace Domain"""
    
    def __init__(self):
        super().__init__(
            name="aerospace",
            description="Aerospace projects - UAVs, aircraft design, flight systems"
        )
    
        # Professional systems integration (lazy initialization)
        self._professional_integration = None
    
    def get_knowledge_extractors(self) -> list:
        """Return aerospace knowledge extractors"""
        return [
            KnowledgeExtractor(
                name="aerodynamics",
                description="Airfoils, lift, drag, CFD analysis",
                patterns=["airfoil", "lift", "drag", "reynolds"],
                extractor_func=None,
                storage_db="aerodynamics"
            ),
            KnowledgeExtractor(
                name="propulsion",
                description="Engines, motors, propellers, thrust",
                patterns=["propeller", "motor", "thrust", "efficiency"],
                extractor_func=None,
                storage_db="propulsion"
            ),
            KnowledgeExtractor(
                name="flight_control",
                description="Autopilots, stability, control systems",
                patterns=["autopilot", "pid", "stability"],
                extractor_func=None,
                storage_db="flight_control"
            ),
            KnowledgeExtractor(
                name="regulations",
                description="FAA, Transport Canada, EASA rules",
                patterns=["faa", "part 107", "certification"],
                extractor_func=None,
                storage_db="regulations"
            )
        ]
    
    async def create_project(
        self,
        description: str,
        requirements: Optional[Dict[str, Any]] = None
    ) -> AerospaceProjectStateMachine:
        """Create a new aerospace project"""
        import uuid
        project_id = f"aero-{uuid.uuid4().hex[:8]}"
        
        project = AerospaceProjectStateMachine(project_id, description)
        
        if requirements:
            if "aircraft_type" in requirements:
                project.aircraft_type = AircraftType(requirements["aircraft_type"])
            if "wingspan_m" in requirements:
                project.wingspan_m = requirements["wingspan_m"]
            if "mtow_kg" in requirements:
                project.mtow_kg = requirements["mtow_kg"]
            if "propulsion_type" in requirements:
                project.propulsion_type = requirements["propulsion_type"]
            if "budget" in requirements:
                project.budget["estimated_total"] = requirements["budget"]
        
        return project
    
    def get_deliverable_types(self) -> list:
        """List all aerospace deliverables"""
        return [
            DeliverableSpec(
                name="cfd_analysis",
                description="Computational fluid dynamics analysis",
                file_types=["pdf", "csv"],
                generator_func=None,
                required_knowledge=["aerodynamics"]
            ),
            DeliverableSpec(
                name="weight_balance",
                description="Weight and balance calculations",
                file_types=["json", "pdf"],
                generator_func=None,
                required_knowledge=[]
            ),
            DeliverableSpec(
                name="flight_envelope",
                description="Flight envelope and performance charts",
                file_types=["json", "pdf"],
                generator_func=None,
                required_knowledge=["aerodynamics"]
            ),
            DeliverableSpec(
                name="test_report",
                description="Flight test report and data",
                file_types=["json", "pdf"],
                generator_func=None,
                required_knowledge=[]
            )
        ]
    
    async def generate_deliverables(
        self,
        project: AerospaceProjectStateMachine,
        deliverable_types: List[str],
        output_dir: Path
    ) -> Dict[str, Path]:
        """Generate aerospace deliverables"""
        generated_files = {}
        
        for deliverable_type in deliverable_types:
            if deliverable_type == "weight_balance":
                wb = self._generate_weight_balance(project)
                file_path = output_dir / "weight_balance.json"
                with open(file_path, 'w') as f:
                    import json
                    json.dump(wb, f, indent=2)
                generated_files["weight_balance"] = file_path
        
        return generated_files
    
    def _generate_weight_balance(self, project: AerospaceProjectStateMachine) -> Dict[str, Any]:
        """Generate weight and balance document"""
        return {
            "aircraft": project.description,
            "type": project.aircraft_type.value if project.aircraft_type else "unknown",
            "mtow_kg": project.mtow_kg,
            "empty_weight_kg": project.mtow_kg * 0.6,
            "payload_kg": project.mtow_kg * 0.2,
            "fuel_kg": project.mtow_kg * 0.2,
            "cg_limits": {
                "forward": "25% MAC",
                "aft": "35% MAC"
            }
        }
    
    async def validate_requirements(
        self,
        requirements: Dict[str, Any]
    ) -> ValidationResult:
        """Validate aerospace project requirements"""
        errors = []
        warnings = []
        suggestions = []
        
        if "aircraft_type" not in requirements:
            errors.append("Aircraft type is required")
        
        if "mtow_kg" not in requirements:
            warnings.append("Maximum takeoff weight should be specified")
        
        # Check physics constraints
        if "propulsion_type" in requirements and "mtow_kg" in requirements:
            if requirements["propulsion_type"] == "electric" and requirements["mtow_kg"] > 50:
                suggestions.append("Consider hybrid propulsion for aircraft over 50kg")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    async def estimate_complexity(
        self,
        project: AerospaceProjectStateMachine
    ) -> ComplexityScore:
        """Estimate aerospace project complexity"""
        score = 40.0  # Base complexity for aerospace
        factors = {}
        
        # Aircraft type complexity
        type_scores = {
            AircraftType.FIXED_WING: 30,
            AircraftType.MULTIROTOR: 20,
            AircraftType.VTOL: 50,
            AircraftType.HELICOPTER: 60,
            AircraftType.GLIDER: 25
        }
        
        if project.aircraft_type:
            type_score = type_scores.get(project.aircraft_type, 30)
            score += type_score
            factors["aircraft_type"] = type_score
        
        # Weight complexity
        if project.mtow_kg > 25:
            weight_score = 15
            score += weight_score
            factors["mtow"] = weight_score
        
        score = min(score, 100)
        
        # Time estimate
        time_estimate = int(score * 3)  # 3 days per complexity point
        
        # Cost estimate
        cost_estimate = time_estimate * 600  # $600/day for aerospace
        
        # Risk level
        if score < 50:
            risk = "medium"
        elif score < 75:
            risk = "high"
        else:
            risk = "very_high"
        
        return ComplexityScore(
            overall_score=score,
            time_estimate_days=time_estimate,
            cost_estimate_usd=cost_estimate,
            risk_level=risk,
            factors=factors
        )
    
    def get_knowledge_stats(self) -> Dict[str, int]:
        """Get knowledge base statistics"""
        return {
            "aerodynamics": 0,
            "propulsion": 0,
            "flight_control": 0,
            "regulations": 0
        }
    
    async def _get_professional_integration(self):
        """Get or initialize professional integration"""
        if self._professional_integration is None:
            from modules.domains.domain_professional_integration import DomainProfessionalIntegration
            self._professional_integration = DomainProfessionalIntegration(
                domain_name="aerospace"
            )
            await self._professional_integration.initialize()
            
            # Initialize aerospace professional roles
            await self._professional_integration.initialize_roles([
                ("SYSTEMS_ENGINEER", "DESIGN"),
                ("TEST_ENGINEER", "ANALYSIS"),
                ("SYSTEMS_ENGINEER", "PLANNING")
            ])
        
        return self._professional_integration
    
    async def get_team_orchestrator(self):
        """Get professional team orchestrator"""
        integration = await self._get_professional_integration()
        return integration.team_orchestrator
    
    async def get_deliverable_generator(self):
        """Get professional deliverable generator"""
        integration = await self._get_professional_integration()
        return integration.deliverable_generator
    
    async def get_cross_learning(self):
        """Get cross-domain learning system"""
        integration = await self._get_professional_integration()
        return integration.cross_learning
    
    async def get_workflow_executor(self):
        """Get professional workflow executor"""
        integration = await self._get_professional_integration()
        return integration.workflow_executor
    
    async def get_quality_framework(self):
        """Get quality assurance framework"""
        integration = await self._get_professional_integration()
        return integration.quality_framework
