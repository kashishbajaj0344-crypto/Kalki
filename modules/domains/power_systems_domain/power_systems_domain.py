"""
Power Systems Domain

Specialized domain for power and energy systems including:
- Battery energy storage systems
- Solar photovoltaic systems
- Grid-tied and off-grid systems
- Electric vehicle power systems
- Energy efficiency optimization
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


class PowerSystemsPhase(str, Enum):
    """Power systems project phases"""
    CONCEPT = "concept"
    REQUIREMENTS = "requirements"
    SYSTEM_DESIGN = "system_design"
    COMPONENT_SELECTION = "component_selection"
    ELECTRICAL_DESIGN = "electrical_design"
    INTEGRATION = "integration"
    TESTING = "testing"
    COMMISSIONING = "commissioning"


class SystemType(str, Enum):
    """Power system types"""
    BATTERY_STORAGE = "battery_storage"
    SOLAR_PV = "solar_pv"
    HYBRID = "hybrid"
    EV_CHARGING = "ev_charging"
    MICROGRID = "microgrid"
    GRID_TIED = "grid_tied"
    OFF_GRID = "off_grid"


class PowerSystemsProjectStateMachine(ProjectStateMachine):
    """State machine for power systems projects"""
    
    def __init__(self, project_id: str, description: str):
        super().__init__(project_id, description, "power_systems")
        self.current_phase = PowerSystemsPhase.CONCEPT
        
        # System specifications
        self.system_type = None
        self.capacity_kwh = 0.0
        self.power_kw = 0.0
        self.voltage_v = 0
        self.efficiency_percent = 0.0
        
        # Components
        self.battery_chemistry = None  # lithium_ion, lifepo4, lead_acid
        self.solar_panels = []
        self.inverter_type = None
        self.charge_controller = None
        
        # Budget tracking
        self.budget = {
            "estimated_total": 0,
            "actual_spent": 0,
            "by_category": {
                "batteries": 0,
                "solar_panels": 0,
                "inverters": 0,
                "bms": 0,
                "installation": 0,
                "commissioning": 0
            }
        }
        
        # Milestones
        self.milestones = {
            PowerSystemsPhase.CONCEPT: [
                {"name": "Define energy requirements", "complete": False},
                {"name": "Select system type", "complete": False},
                {"name": "Establish design criteria", "complete": False}
            ],
            PowerSystemsPhase.REQUIREMENTS: [
                {"name": "Calculate energy budget", "complete": False},
                {"name": "Specify power requirements", "complete": False},
                {"name": "Define safety requirements", "complete": False},
                {"name": "Set efficiency targets", "complete": False}
            ],
            PowerSystemsPhase.SYSTEM_DESIGN: [
                {"name": "Size battery capacity", "complete": False},
                {"name": "Design solar array (if applicable)", "complete": False},
                {"name": "Calculate power distribution", "complete": False},
                {"name": "Plan thermal management", "complete": False}
            ],
            PowerSystemsPhase.COMPONENT_SELECTION: [
                {"name": "Select battery cells/modules", "complete": False},
                {"name": "Choose inverter/charger", "complete": False},
                {"name": "Select BMS", "complete": False},
                {"name": "Pick monitoring system", "complete": False}
            ],
            PowerSystemsPhase.ELECTRICAL_DESIGN: [
                {"name": "Design wiring schematic", "complete": False},
                {"name": "Size conductors and breakers", "complete": False},
                {"name": "Plan grounding system", "complete": False},
                {"name": "Design control system", "complete": False}
            ],
            PowerSystemsPhase.INTEGRATION: [
                {"name": "Install batteries", "complete": False},
                {"name": "Install solar panels (if applicable)", "complete": False},
                {"name": "Wire electrical system", "complete": False},
                {"name": "Configure BMS and monitoring", "complete": False}
            ],
            PowerSystemsPhase.TESTING: [
                {"name": "Test individual components", "complete": False},
                {"name": "Perform system integration test", "complete": False},
                {"name": "Validate safety systems", "complete": False},
                {"name": "Measure efficiency", "complete": False}
            ],
            PowerSystemsPhase.COMMISSIONING: [
                {"name": "Grid interconnection (if applicable)", "complete": False},
                {"name": "Final safety inspection", "complete": False},
                {"name": "Train operators", "complete": False},
                {"name": "Begin monitored operation", "complete": False}
            ]
        }
    
    async def validate_phase_complete(self, phase: PowerSystemsPhase) -> ValidationResult:
        """Validate if phase requirements are met"""
        errors = []
        warnings = []
        suggestions = []
        
        phase_milestones = self.milestones.get(phase, [])
        completed = sum(1 for m in phase_milestones if m["complete"])
        total = len(phase_milestones)
        
        if phase == PowerSystemsPhase.CONCEPT:
            if not self.system_type:
                errors.append("System type not specified")
            if completed < total:
                warnings.append(f"Only {completed}/{total} concept milestones complete")
        
        elif phase == PowerSystemsPhase.REQUIREMENTS:
            if self.capacity_kwh == 0:
                warnings.append("Energy capacity not specified")
            if self.power_kw == 0:
                warnings.append("Power rating not specified")
        
        elif phase == PowerSystemsPhase.SYSTEM_DESIGN:
            if not self.battery_chemistry:
                errors.append("Battery chemistry not selected")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    async def advance_phase(self, next_phase: PowerSystemsPhase) -> bool:
        """Advance to next project phase"""
        self.phase_history.append({
            "from": self.current_phase,
            "to": next_phase,
            "timestamp": datetime.now().isoformat()
        })
        self.current_phase = next_phase
        return True
    
    def get_available_phases(self) -> list:
        """Get all power systems phases"""
        return list(PowerSystemsPhase)
    
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
        """Provide help relevant to current power systems phase"""
        phase_help = {
            PowerSystemsPhase.CONCEPT: "Concept phase: Define energy needs and select system type",
            PowerSystemsPhase.REQUIREMENTS: "Requirements phase: Calculate energy budget, power, safety",
            PowerSystemsPhase.SYSTEM_DESIGN: "System design: Size capacity, design array, thermal management",
            PowerSystemsPhase.COMPONENT_SELECTION: "Component selection: Choose batteries, inverter, BMS, monitoring",
            PowerSystemsPhase.ELECTRICAL_DESIGN: "Electrical design: Wiring schematic, conductors, grounding, control",
            PowerSystemsPhase.INTEGRATION: "Integration: Install batteries, solar, wiring, configure BMS",
            PowerSystemsPhase.TESTING: "Testing: Component tests, integration test, safety validation, efficiency",
            PowerSystemsPhase.COMMISSIONING: "Commissioning: Grid interconnection, inspection, training, monitoring"
        }
        
        progress = self.get_phase_progress()
        
        help_text = f"""
Current Phase: {self.current_phase.value}
Progress: {progress['completed_milestones']}/{progress['total_milestones']} milestones ({progress['percent_complete']:.1f}%)

{phase_help.get(self.current_phase, 'Power systems development in progress')}

Pending Milestones:
"""
        
        for milestone in progress['milestones']:
            if not milestone['complete']:
                help_text += f"  • {milestone['name']}\n"
        
        return help_text.strip()


class PowerSystemsDomain(BaseDomain):
    """Power Systems Domain"""
    
    def __init__(self):
        super().__init__(
            name="power_systems",
            description="Power and energy systems - batteries, solar, grid systems"
        )
    
        # Professional systems integration (lazy initialization)
        self._professional_integration = None
    
    def get_knowledge_extractors(self) -> list:
        """Return power systems knowledge extractors"""
        return [
            KnowledgeExtractor(
                name="battery_tech",
                description="Battery chemistry, BMS, safety",
                patterns=["lithium", "lifepo4", "bms", "cell"],
                extractor_func=None,
                storage_db="batteries"
            ),
            KnowledgeExtractor(
                name="solar_pv",
                description="Solar panels, MPPT, efficiency",
                patterns=["solar", "photovoltaic", "mppt", "panel"],
                extractor_func=None,
                storage_db="solar"
            ),
            KnowledgeExtractor(
                name="power_electronics",
                description="Inverters, converters, chargers",
                patterns=["inverter", "converter", "dc-dc", "ac-dc"],
                extractor_func=None,
                storage_db="electronics"
            ),
            KnowledgeExtractor(
                name="energy_storage",
                description="ESS design, grid integration",
                patterns=["energy storage", "grid", "microgrid"],
                extractor_func=None,
                storage_db="storage"
            )
        ]
    
    async def create_project(
        self,
        description: str,
        requirements: Optional[Dict[str, Any]] = None
    ) -> PowerSystemsProjectStateMachine:
        """Create a new power systems project"""
        import uuid
        project_id = f"power-{uuid.uuid4().hex[:8]}"
        
        project = PowerSystemsProjectStateMachine(project_id, description)
        
        if requirements:
            if "system_type" in requirements:
                project.system_type = SystemType(requirements["system_type"])
            if "capacity_kwh" in requirements:
                project.capacity_kwh = requirements["capacity_kwh"]
            if "power_kw" in requirements:
                project.power_kw = requirements["power_kw"]
            if "battery_chemistry" in requirements:
                project.battery_chemistry = requirements["battery_chemistry"]
            if "budget" in requirements:
                project.budget["estimated_total"] = requirements["budget"]
        
        return project
    
    def get_deliverable_types(self) -> list:
        """List all power systems deliverables"""
        return [
            DeliverableSpec(
                name="energy_budget",
                description="Energy consumption and generation analysis",
                file_types=["json", "xlsx"],
                generator_func=None,
                required_knowledge=[]
            ),
            DeliverableSpec(
                name="system_diagram",
                description="Electrical single-line diagram",
                file_types=["pdf", "dwg"],
                generator_func=None,
                required_knowledge=[]
            ),
            DeliverableSpec(
                name="bom",
                description="Bill of materials with specifications",
                file_types=["json", "xlsx"],
                generator_func=None,
                required_knowledge=[]
            ),
            DeliverableSpec(
                name="safety_analysis",
                description="Safety hazards and mitigation",
                file_types=["json", "pdf"],
                generator_func=None,
                required_knowledge=["battery_tech"]
            )
        ]
    
    async def generate_deliverables(
        self,
        project: PowerSystemsProjectStateMachine,
        deliverable_types: List[str],
        output_dir: Path
    ) -> Dict[str, Path]:
        """Generate power systems deliverables"""
        generated_files = {}
        
        for deliverable_type in deliverable_types:
            if deliverable_type == "energy_budget":
                budget = self._generate_energy_budget(project)
                file_path = output_dir / "energy_budget.json"
                with open(file_path, 'w') as f:
                    import json
                    json.dump(budget, f, indent=2)
                generated_files["energy_budget"] = file_path
        
        return generated_files
    
    def _generate_energy_budget(self, project: PowerSystemsProjectStateMachine) -> Dict[str, Any]:
        """Generate energy budget analysis"""
        return {
            "system": project.description,
            "type": project.system_type.value if project.system_type else "unknown",
            "capacity_kwh": project.capacity_kwh,
            "power_kw": project.power_kw,
            "daily_energy_kwh": project.capacity_kwh * 0.8,  # 80% DOD
            "estimated_cycles": 5000,
            "lifetime_years": 10,
            "efficiency": f"{project.efficiency_percent}%" if project.efficiency_percent > 0 else "92%"
        }
    
    async def validate_requirements(
        self,
        requirements: Dict[str, Any]
    ) -> ValidationResult:
        """Validate power systems project requirements"""
        errors = []
        warnings = []
        suggestions = []
        
        if "system_type" not in requirements:
            errors.append("System type is required")
        
        if "capacity_kwh" not in requirements:
            warnings.append("Energy capacity should be specified")
        
        if "power_kw" not in requirements:
            warnings.append("Power rating should be specified")
        
        # Check power/energy ratio
        if "capacity_kwh" in requirements and "power_kw" in requirements:
            c_rate = requirements["power_kw"] / requirements["capacity_kwh"]
            if c_rate > 1.0:
                warnings.append(f"High C-rate ({c_rate:.1f}C) may reduce battery lifespan")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    async def estimate_complexity(
        self,
        project: PowerSystemsProjectStateMachine
    ) -> ComplexityScore:
        """Estimate power systems project complexity"""
        score = 30.0
        factors = {}
        
        # System type complexity
        type_scores = {
            SystemType.BATTERY_STORAGE: 25,
            SystemType.SOLAR_PV: 20,
            SystemType.HYBRID: 40,
            SystemType.EV_CHARGING: 30,
            SystemType.MICROGRID: 50,
            SystemType.GRID_TIED: 35,
            SystemType.OFF_GRID: 45
        }
        
        if project.system_type:
            type_score = type_scores.get(project.system_type, 30)
            score += type_score
            factors["system_type"] = type_score
        
        # Capacity complexity
        if project.capacity_kwh > 100:
            capacity_score = 20
            score += capacity_score
            factors["large_capacity"] = capacity_score
        
        score = min(score, 100)
        
        # Time estimate
        time_estimate = int(score * 2.5)  # 2.5 days per complexity point
        
        # Cost estimate
        cost_estimate = time_estimate * 550  # $550/day
        
        # Risk level
        if score < 40:
            risk = "low"
        elif score < 70:
            risk = "medium"
        else:
            risk = "high"
        
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
            "battery_tech": 0,
            "solar_pv": 0,
            "power_electronics": 0,
            "energy_storage": 0
        }
    
    async def _get_professional_integration(self):
        """Get or initialize professional integration"""
        if self._professional_integration is None:
            from modules.domains.domain_professional_integration import DomainProfessionalIntegration
            self._professional_integration = DomainProfessionalIntegration(
                domain_name="power_systems"
            )
            await self._professional_integration.initialize()
            
            # Initialize power systems professional roles
            await self._professional_integration.initialize_roles([
                ("ELECTRICAL_ENGINEER", "DESIGN"),
                ("THERMAL_ENGINEER", "ANALYSIS"),
                ("SAFETY_OFFICER", "QUALITY_ASSESSMENT")
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
