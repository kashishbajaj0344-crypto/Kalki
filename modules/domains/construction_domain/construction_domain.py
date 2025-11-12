"""
Construction Domain

KALKI's expertise in construction, architecture, and building systems.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import List, Dict, Any, Optional
from pathlib import Path
import sqlite3
import json

from modules.domains.base_domain import (
    BaseDomain,
    ProjectStateMachine,
    ValidationResult,
    ComplexityScore,
    KnowledgeExtractor,
    DeliverableSpec,
    ProjectPhase
)
from enum import Enum


class ConstructionPhase(str, Enum):
    """Construction-specific project phases"""
    REQUIREMENTS = "requirements_gathering"
    DESIGN = "design_generation"
    PERMIT_PREP = "permit_preparation"
    FOUNDATION = "foundation"
    FRAMING = "framing"
    ROUGH_MEP = "rough_mechanical_electrical_plumbing"
    INSULATION = "insulation"
    DRYWALL = "drywall"
    FINISHING = "finishing"
    FINAL_INSPECTION = "final_inspection"
    OCCUPANCY = "occupancy"
    DIGITAL_TWIN = "digital_twin_creation"


class ConstructionProjectStateMachine(ProjectStateMachine):
    """State machine for construction projects with enhanced tracking"""
    
    def __init__(self, project_id: str, description: str):
        super().__init__(project_id, description, "construction")
        self.current_phase = ConstructionPhase.REQUIREMENTS
        
        # Budget tracking
        self.budget = {
            "estimated_total": 0,
            "actual_spent": 0,
            "by_phase": {},
            "contingency_percent": 10
        }
        
        # Timeline management
        self.timeline = {
            "start_date": None,
            "target_completion": None,
            "phase_durations": {},
            "actual_phase_durations": {}
        }
        
        # Project details
        self.location = None
        self.building_type = None
        self.size_sqft = None
        self.stories = None
        
        # Milestone tracking per phase
        self.milestones = {
            ConstructionPhase.REQUIREMENTS: [
                {"name": "Site survey complete", "complete": False},
                {"name": "Budget approved", "complete": False},
                {"name": "Design brief finalized", "complete": False}
            ],
            ConstructionPhase.DESIGN: [
                {"name": "Schematic design approved", "complete": False},
                {"name": "Construction drawings complete", "complete": False},
                {"name": "Structural calculations verified", "complete": False}
            ],
            ConstructionPhase.PERMIT_PREP: [
                {"name": "Permit documents submitted", "complete": False},
                {"name": "Plan check responses completed", "complete": False},
                {"name": "Building permit issued", "complete": False}
            ],
            ConstructionPhase.FOUNDATION: [
                {"name": "Excavation complete", "complete": False},
                {"name": "Footing inspection passed", "complete": False},
                {"name": "Foundation walls complete", "complete": False}
            ],
            ConstructionPhase.FRAMING: [
                {"name": "Floor framing complete", "complete": False},
                {"name": "Wall framing complete", "complete": False},
                {"name": "Roof framing complete", "complete": False},
                {"name": "Framing inspection passed", "complete": False}
            ],
            ConstructionPhase.ROUGH_MEP: [
                {"name": "HVAC rough-in complete", "complete": False},
                {"name": "Electrical rough-in complete", "complete": False},
                {"name": "Plumbing rough-in complete", "complete": False},
                {"name": "MEP inspection passed", "complete": False}
            ],
            ConstructionPhase.INSULATION: [
                {"name": "Insulation installed", "complete": False},
                {"name": "Vapor barrier installed", "complete": False},
                {"name": "Insulation inspection passed", "complete": False}
            ],
            ConstructionPhase.DRYWALL: [
                {"name": "Drywall hung", "complete": False},
                {"name": "Taping and mudding complete", "complete": False},
                {"name": "Sanding complete", "complete": False}
            ],
            ConstructionPhase.FINISHING: [
                {"name": "Painting complete", "complete": False},
                {"name": "Flooring installed", "complete": False},
                {"name": "Fixtures installed", "complete": False},
                {"name": "Trim and cabinetry complete", "complete": False}
            ],
            ConstructionPhase.FINAL_INSPECTION: [
                {"name": "Final building inspection passed", "complete": False},
                {"name": "Deficiencies corrected", "complete": False}
            ],
            ConstructionPhase.OCCUPANCY: [
                {"name": "Occupancy permit issued", "complete": False},
                {"name": "Owner handover complete", "complete": False}
            ],
            ConstructionPhase.DIGITAL_TWIN: [
                {"name": "As-built drawings complete", "complete": False},
                {"name": "Digital twin model created", "complete": False}
            ]
        }
    
    async def advance_phase(self, next_phase: ConstructionPhase) -> bool:
        """Advance to next construction phase with validation"""
        from datetime import datetime
        
        # Validate current phase is complete
        validation = await self.validate_phase_complete(self.current_phase)
        if not validation.valid:
            print(f"Cannot advance: {', '.join(validation.errors)}")
            return False
        
        # Record actual duration
        if self.current_phase in self.timeline.get("phase_durations", {}):
            # In a real system, calculate from start time
            self.timeline["actual_phase_durations"][self.current_phase.value] = "calculated_duration"
        
        # Record transition
        self.phase_history.append({
            "from": self.current_phase.value,
            "to": next_phase.value,
            "timestamp": datetime.now().isoformat(),
            "milestones_complete": sum(1 for m in self.milestones.get(self.current_phase, []) if m["complete"])
        })
        
        self.current_phase = next_phase
        return True
    
    
    async def validate_phase_complete(self, phase: ConstructionPhase) -> ValidationResult:
        """Check if phase requirements are met"""
        errors = []
        warnings = []
        suggestions = []
        
        # Check critical milestones for phase
        phase_milestones = self.milestones.get(phase, [])
        incomplete_critical = [m["name"] for m in phase_milestones if not m["complete"] and "inspection" in m["name"].lower()]
        
        if incomplete_critical:
            errors.append(f"Critical milestones incomplete: {', '.join(incomplete_critical)}")
        
        # Phase-specific validations
        if phase == ConstructionPhase.REQUIREMENTS:
            if not self.location:
                errors.append("Location not specified")
            if not self.building_type:
                errors.append("Building type not specified")
            if not self.budget.get("estimated_total"):
                warnings.append("Budget not estimated")
            if not self.size_sqft:
                warnings.append("Building size not specified")
                
        elif phase == ConstructionPhase.DESIGN:
            incomplete = sum(1 for m in phase_milestones if not m["complete"])
            if incomplete > 0:
                errors.append(f"{incomplete} design milestones incomplete")
            suggestions.append("Ensure structural calculations are peer-reviewed")
            
        elif phase == ConstructionPhase.PERMIT_PREP:
            if not any(m["complete"] for m in phase_milestones if "permit issued" in m["name"].lower()):
                errors.append("Building permit not yet issued")
                
        elif phase == ConstructionPhase.FOUNDATION:
            if not any(m["complete"] for m in phase_milestones if "inspection" in m["name"].lower()):
                errors.append("Foundation inspection not passed")
                
        elif phase == ConstructionPhase.FRAMING:
            if not any(m["complete"] for m in phase_milestones if "inspection" in m["name"].lower()):
                errors.append("Framing inspection not passed")
            # Check budget tracking
            if self.budget["actual_spent"] > self.budget["estimated_total"] * 0.7:
                warnings.append("70% of budget spent, monitor costs closely")
                
        elif phase == ConstructionPhase.ROUGH_MEP:
            if not any(m["complete"] for m in phase_milestones if "inspection" in m["name"].lower()):
                errors.append("MEP rough-in inspection not passed")
                
        elif phase == ConstructionPhase.INSULATION:
            if not any(m["complete"] for m in phase_milestones if "inspection" in m["name"].lower()):
                errors.append("Insulation inspection not passed")
                
        elif phase == ConstructionPhase.FINAL_INSPECTION:
            if not any(m["complete"] for m in phase_milestones if "final" in m["name"].lower()):
                errors.append("Final building inspection not passed")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    
    def get_available_phases(self) -> List[ConstructionPhase]:
        """Get all construction phases"""
        return list(ConstructionPhase)
    
    def mark_milestone_complete(self, milestone_name: str) -> bool:
        """Mark a milestone as complete in current phase"""
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
    
    def update_budget(self, category: str, amount: float, is_actual: bool = False):
        """Update budget tracking"""
        if is_actual:
            self.budget["actual_spent"] += amount
            phase_key = self.current_phase.value
            if phase_key not in self.budget["by_phase"]:
                self.budget["by_phase"][phase_key] = {"estimated": 0, "actual": 0}
            self.budget["by_phase"][phase_key]["actual"] += amount
        else:
            # Update estimate
            phase_key = self.current_phase.value
            if phase_key not in self.budget["by_phase"]:
                self.budget["by_phase"][phase_key] = {"estimated": 0, "actual": 0}
            self.budget["by_phase"][phase_key]["estimated"] += amount
    
    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status"""
        estimated = self.budget.get("estimated_total", 0)
        actual = self.budget.get("actual_spent", 0)
        contingency = self.budget.get("contingency_percent", 10)
        
        budget_with_contingency = estimated * (1 + contingency / 100)
        remaining = budget_with_contingency - actual
        percent_spent = (actual / budget_with_contingency * 100) if budget_with_contingency > 0 else 0
        
        return {
            "estimated_total": estimated,
            "contingency_amount": estimated * contingency / 100,
            "budget_with_contingency": budget_with_contingency,
            "actual_spent": actual,
            "remaining": remaining,
            "percent_spent": percent_spent,
            "status": "on_budget" if percent_spent < 90 else "over_budget" if percent_spent > 100 else "warning",
            "by_phase": self.budget.get("by_phase", {})
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state machine to dictionary for JSON storage"""
        return {
            "project_id": self.project_id,
            "description": self.description,
            "domain": self.domain,
            "current_phase": self.current_phase.value,
            "phase_history": self.phase_history,
            "budget": self.budget,
            "timeline": self.timeline,
            "location": self.location,
            "building_type": self.building_type,
            "size_sqft": self.size_sqft,
            "stories": self.stories,
            "milestones": {k.value: v for k, v in self.milestones.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConstructionProjectStateMachine':
        """Reconstruct state machine from saved JSON"""
        state_machine = cls(data["project_id"], data["description"])
        
        # Restore state - handle both "REQUIREMENTS" and "ConstructionPhase.REQUIREMENTS" formats
        current_phase_str = data["current_phase"]
        if "." in current_phase_str:
            # Strip "ConstructionPhase." prefix
            current_phase_str = current_phase_str.split(".")[-1]
        state_machine.current_phase = ConstructionPhase(current_phase_str)
        
        state_machine.phase_history = data.get("phase_history", [])
        state_machine.budget = data.get("budget", {})
        state_machine.timeline = data.get("timeline", {})
        state_machine.location = data.get("location")
        state_machine.building_type = data.get("building_type")
        state_machine.size_sqft = data.get("size_sqft")
        state_machine.stories = data.get("stories")
        
        # Restore milestones - handle both formats
        if "milestones" in data:
            state_machine.milestones = {}
            for k, v in data["milestones"].items():
                # Strip enum class name if present
                phase_key = k.split(".")[-1] if "." in k else k
                state_machine.milestones[ConstructionPhase(phase_key)] = v
        
        return state_machine
    
    async def get_contextual_help(self, user_query: str) -> str:
        """Provide help relevant to current construction phase"""
        phase_help = {
            ConstructionPhase.REQUIREMENTS: "Gathering requirements: lot size, zoning, budget, timeline",
            ConstructionPhase.DESIGN: "Creating construction drawings and specifications",
            ConstructionPhase.PERMIT_PREP: "Preparing permit application documents",
            ConstructionPhase.FOUNDATION: "Foundation excavation, forming, and pouring",
            ConstructionPhase.FRAMING: "Wall framing, roof framing, structural members",
            ConstructionPhase.ROUGH_MEP: "Mechanical, electrical, and plumbing rough-in",
            ConstructionPhase.INSULATION: "Installing insulation and vapor barriers",
            ConstructionPhase.DRYWALL: "Drywall installation and finishing",
            ConstructionPhase.FINISHING: "Painting, flooring, fixtures, trim",
            ConstructionPhase.FINAL_INSPECTION: "Final building inspection and sign-off",
            ConstructionPhase.OCCUPANCY: "Occupancy permit and handover",
            ConstructionPhase.DIGITAL_TWIN: "Creating digital twin for ongoing monitoring"
        }
        
        # Get phase progress
        progress = self.get_phase_progress()
        budget_status = self.get_budget_status()
        
        help_text = f"""
Current Phase: {self.current_phase.value}
Progress: {progress['completed_milestones']}/{progress['total_milestones']} milestones ({progress['percent_complete']:.1f}%)
Budget: ${budget_status['actual_spent']:,.2f} spent of ${budget_status['budget_with_contingency']:,.2f} ({budget_status['percent_spent']:.1f}%)

{phase_help.get(self.current_phase, 'Phase info unavailable')}

Pending Milestones:
"""
        for milestone in progress['milestones']:
            if not milestone['complete']:
                help_text += f"  • {milestone['name']}\n"
        
        return help_text


class ConstructionDomain(BaseDomain):
    """
    Construction domain expertise.
    
    Handles:
    - Residential and commercial building design
    - BC Building Code compliance
    - Construction sequencing and scheduling
    - Cost estimation
    - Professional deliverables (drawings, BOM, schedules)
    """
    
    def __init__(self):
        super().__init__(
            name="construction",
            description="Building design, construction management, and delivery"
        )
        
        # Database paths (using existing v2.5 databases)
        self.data_dir = Path("data")
        self.span_tables_db = self.data_dir / "span_tables.db"
        self.procedures_db = self.data_dir / "procedures.db"
        self.inspection_criteria_db = self.data_dir / "inspection_criteria.db"
        self.cost_data_db = self.data_dir / "cost_data.db"
        self.load_parameters_db = self.data_dir / "load_parameters.db"
        self.decision_trees_db = self.data_dir / "decision_trees.db"
    
    def get_knowledge_extractors(self) -> List[KnowledgeExtractor]:
        """Return construction-specific knowledge extractors"""
        # Import extraction functions from hybrid_learning_system
        from modules.hybrid_learning_system import KnowledgeExtractor as KE
        
        ke = KE()
        
        return [
            KnowledgeExtractor(
                name="span_tables",
                description="Structural member sizing tables (joists, beams, rafters)",
                patterns=[r"(\d+x\d+)\s+@\s+(\d+)\""],
                extractor_func=ke._extract_span_tables,
                storage_db="span_tables.db"
            ),
            KnowledgeExtractor(
                name="procedures",
                description="Step-by-step construction sequences",
                patterns=[r"Step\s+(\d+)[:.]\s+([A-Z][^\n\.]+)"],
                extractor_func=ke._extract_procedures,
                storage_db="procedures.db"
            ),
            KnowledgeExtractor(
                name="inspection_criteria",
                description="Quality control validation points",
                patterns=[r"[Ii]nspect\s+([^for]+?)\s+for\s+([^\n\.]+)"],
                extractor_func=ke._extract_inspection_criteria,
                storage_db="inspection_criteria.db"
            ),
            KnowledgeExtractor(
                name="cost_data",
                description="Material and labor unit costs",
                patterns=[r"([A-Za-z0-9][^\n:$]+?):\s*\$(\d+\.?\d*)"],
                extractor_func=ke._extract_cost_data,
                storage_db="cost_data.db"
            ),
            KnowledgeExtractor(
                name="load_parameters",
                description="Structural design loads (live, dead, snow, wind)",
                patterns=[r"([A-Za-z\s]+load):\s*(\d+\.?\d*)\s*(PSF|PSI|kN|kPa)"],
                extractor_func=ke._extract_load_parameters,
                storage_db="load_parameters.db"
            ),
            KnowledgeExtractor(
                name="decision_trees",
                description="Conditional code compliance logic",
                patterns=[r"[Ii]f\s+([^,]+?)\s*([<>=]+)\s*([^,]+?),\s*then"],
                extractor_func=ke._extract_decision_trees,
                storage_db="decision_trees.db"
            )
        ]
    
    async def create_project(
        self,
        description: str,
        requirements: Optional[Dict[str, Any]] = None
    ) -> ConstructionProjectStateMachine:
        """Initialize a new construction project"""
        import uuid
        
        project_id = str(uuid.uuid4())
        project = ConstructionProjectStateMachine(project_id, description)
        
        if requirements:
            project.location = requirements.get("location")
            project.building_type = requirements.get("building_type")
            project.budget = requirements.get("budget", {})
            project.timeline = requirements.get("timeline", {})
        
        return project
    
    def get_deliverable_types(self) -> List[DeliverableSpec]:
        """List construction deliverables"""
        return [
            DeliverableSpec(
                name="construction_drawings",
                description="Complete construction drawings (plans, elevations, sections, details)",
                file_types=["pdf", "dwg", "dxf"],
                generator_func=None,  # TODO: Implement
                required_knowledge=["design_rules", "code_requirements", "span_tables"]
            ),
            DeliverableSpec(
                name="bill_of_materials",
                description="Complete BOM with quantities and costs",
                file_types=["xlsx", "csv", "json", "pdf"],
                generator_func=None,  # TODO: Implement
                required_knowledge=["materials", "cost_data"]
            ),
            DeliverableSpec(
                name="construction_schedule",
                description="Phase-by-phase construction timeline",
                file_types=["pdf", "json", "xlsx"],
                generator_func=None,  # TODO: Implement
                required_knowledge=["procedures"]
            ),
            DeliverableSpec(
                name="inspection_checklists",
                description="QC checklists for each construction phase",
                file_types=["pdf", "json"],
                generator_func=None,  # TODO: Implement
                required_knowledge=["inspection_criteria", "code_requirements"]
            ),
            DeliverableSpec(
                name="structural_calculations",
                description="Engineering calculations for structural members",
                file_types=["pdf", "xlsx"],
                generator_func=None,  # TODO: Implement
                required_knowledge=["formulas", "span_tables", "load_parameters"]
            ),
            DeliverableSpec(
                name="cost_estimate",
                description="Detailed project cost breakdown",
                file_types=["pdf", "xlsx", "json"],
                generator_func=None,  # TODO: Implement
                required_knowledge=["cost_data", "materials"]
            )
        ]
    
    async def generate_deliverables(
        self,
        project: ProjectStateMachine,
        deliverable_types: List[str],
        output_dir: Path
    ) -> Dict[str, Path]:
        """Generate construction deliverables"""
        from .deliverables_generator import ConstructionDeliverablesGenerator
        
        generator = ConstructionDeliverablesGenerator(self.data_dir)
        generated = {}
        
        for deliv_type in deliverable_types:
            try:
                # Route to appropriate generator
                if deliv_type == "construction_drawings":
                    result = await generator.generate_construction_drawings(project)
                elif deliv_type == "bill_of_materials":
                    result = await generator.generate_bill_of_materials(project)
                elif deliv_type == "construction_schedule":
                    result = await generator.generate_construction_schedule(project)
                elif deliv_type == "inspection_checklists":
                    result = await generator.generate_inspection_checklists(project)
                elif deliv_type == "structural_calculations":
                    result = await generator.generate_structural_calculations(project)
                elif deliv_type == "cost_estimate":
                    result = await generator.generate_cost_estimate(project)
                else:
                    continue
                
                # Save to JSON file
                output_path = output_dir / f"{deliv_type}.json"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)
                
                generated[deliv_type] = output_path
                
            except Exception as e:
                print(f"Error generating {deliv_type}: {e}")
        
        return generated
    
    async def validate_requirements(
        self,
        requirements: Dict[str, Any]
    ) -> ValidationResult:
        """Validate construction project requirements"""
        errors = []
        warnings = []
        suggestions = []
        
        # Check required fields
        required_fields = ["location", "building_type", "size_sqft"]
        for field in required_fields:
            if field not in requirements:
                errors.append(f"Missing required field: {field}")
        
        # Validate location (BC-specific for now)
        if "location" in requirements:
            location = requirements["location"].lower()
            bc_cities = ["vancouver", "victoria", "sechelt", "whistler", "kelowna"]
            if not any(city in location for city in bc_cities):
                warnings.append("Location may be outside BC - code compliance uncertain")
        
        # Validate size
        if "size_sqft" in requirements:
            size = requirements["size_sqft"]
            if size < 100:
                errors.append("Building size too small (< 100 sq ft)")
            elif size > 10000:
                warnings.append("Large building (> 10,000 sq ft) may require additional engineering")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    async def estimate_complexity(
        self,
        project: ProjectStateMachine
    ) -> ComplexityScore:
        """Estimate construction project complexity"""
        # Simple heuristic for now
        # TODO: Implement sophisticated ML-based estimation
        
        factors = {
            "size": 0.3,
            "stories": 0.2,
            "custom_features": 0.2,
            "site_conditions": 0.15,
            "timeline_constraints": 0.15
        }
        
        overall_score = sum(factors.values()) * 50  # Placeholder
        
        return ComplexityScore(
            overall_score=overall_score,
            time_estimate_days=180,  # ~6 months for typical home
            cost_estimate_usd=300000,  # Placeholder
            risk_level="medium",
            factors=factors
        )
    
    def get_knowledge_stats(self) -> Dict[str, int]:
        """Get construction knowledge statistics"""
        stats = {}
        
        # Query each database
        databases = [
            ("span_tables", self.span_tables_db),
            ("procedures", self.procedures_db),
            ("inspection_criteria", self.inspection_criteria_db),
            ("cost_data", self.cost_data_db),
            ("load_parameters", self.load_parameters_db),
            ("decision_trees", self.decision_trees_db)
        ]
        
        for name, db_path in databases:
            if db_path.exists():
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute(f"SELECT COUNT(*) FROM {name}")
                    count = cursor.fetchone()[0]
                    stats[name] = count
                    conn.close()
                except Exception:
                    stats[name] = 0
            else:
                stats[name] = 0
        
        return stats
