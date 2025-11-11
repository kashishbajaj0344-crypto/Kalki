"""
Robotics Domain

Specialized domain for robotics projects including:
- Mobile robots and autonomous vehicles
- Robotic arms and manipulators
- Control systems and kinematics
- Sensor integration and perception
- SLAM and navigation
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


class RoboticsPhase(str, Enum):
    """Robotics project phases"""
    CONCEPT = "concept"
    REQUIREMENTS = "requirements"
    MECHANICAL_DESIGN = "mechanical_design"
    ELECTRONICS_DESIGN = "electronics_design"
    SOFTWARE_DEVELOPMENT = "software_development"
    INTEGRATION = "integration"
    TESTING = "testing"
    CALIBRATION = "calibration"
    DEPLOYMENT = "deployment"


class RobotType(str, Enum):
    """Robot types"""
    MOBILE_ROBOT = "mobile_robot"
    MANIPULATOR = "manipulator"
    HUMANOID = "humanoid"
    DRONE = "drone"
    UNDERWATER = "underwater"
    INDUSTRIAL = "industrial"
    COLLABORATIVE = "collaborative"
    AGRICULTURAL = "agricultural"


class RoboticsProjectStateMachine(ProjectStateMachine):
    """State machine for robotics projects"""
    
    def __init__(self, project_id: str, description: str):
        super().__init__(project_id, description, "robotics")
        self.current_phase = RoboticsPhase.CONCEPT
        
        # Robot specifications
        self.robot_type = None
        self.dof = 0  # Degrees of freedom
        self.payload_kg = 0.0
        self.max_speed_ms = 0.0
        self.operating_environment = None  # indoor, outdoor, hazardous
        
        # Hardware
        self.microcontroller = None  # arduino, raspberry_pi, jetson
        self.sensors = []
        self.actuators = []
        
        # Software
        self.control_system = None  # pid, mpc, rl
        self.navigation_method = None  # slam, gps, vision
        
        # Budget tracking
        self.budget = {
            "estimated_total": 0,
            "actual_spent": 0,
            "by_category": {
                "mechanical": 0,
                "electronics": 0,
                "sensors": 0,
                "actuators": 0,
                "compute": 0,
                "software": 0
            }
        }
        
        # Milestones
        self.milestones = {
            RoboticsPhase.CONCEPT: [
                {"name": "Define robot purpose and requirements", "complete": False},
                {"name": "Select robot type and configuration", "complete": False},
                {"name": "Identify key technical challenges", "complete": False}
            ],
            RoboticsPhase.REQUIREMENTS: [
                {"name": "Specify performance requirements", "complete": False},
                {"name": "Define operating environment", "complete": False},
                {"name": "Set safety requirements", "complete": False},
                {"name": "Establish budget constraints", "complete": False}
            ],
            RoboticsPhase.MECHANICAL_DESIGN: [
                {"name": "Create CAD models", "complete": False},
                {"name": "Perform kinematic analysis", "complete": False},
                {"name": "Select materials and actuators", "complete": False},
                {"name": "Design mounting and enclosures", "complete": False}
            ],
            RoboticsPhase.ELECTRONICS_DESIGN: [
                {"name": "Select microcontroller/SBC", "complete": False},
                {"name": "Design power system", "complete": False},
                {"name": "Select and integrate sensors", "complete": False},
                {"name": "Design PCB/wiring harness", "complete": False}
            ],
            RoboticsPhase.SOFTWARE_DEVELOPMENT: [
                {"name": "Implement control algorithms", "complete": False},
                {"name": "Develop sensor fusion", "complete": False},
                {"name": "Create navigation system", "complete": False},
                {"name": "Build user interface", "complete": False}
            ],
            RoboticsPhase.INTEGRATION: [
                {"name": "Assemble mechanical components", "complete": False},
                {"name": "Install electronics and sensors", "complete": False},
                {"name": "Deploy software stack", "complete": False},
                {"name": "Perform initial testing", "complete": False}
            ],
            RoboticsPhase.TESTING: [
                {"name": "Test individual subsystems", "complete": False},
                {"name": "Perform integration testing", "complete": False},
                {"name": "Conduct performance validation", "complete": False},
                {"name": "Execute safety testing", "complete": False}
            ],
            RoboticsPhase.CALIBRATION: [
                {"name": "Calibrate sensors", "complete": False},
                {"name": "Tune control parameters", "complete": False},
                {"name": "Optimize performance", "complete": False},
                {"name": "Validate accuracy", "complete": False}
            ],
            RoboticsPhase.DEPLOYMENT: [
                {"name": "Prepare deployment environment", "complete": False},
                {"name": "Train operators", "complete": False},
                {"name": "Deploy robot system", "complete": False},
                {"name": "Monitor initial operation", "complete": False}
            ]
        }
    
    async def validate_phase_complete(self, phase: RoboticsPhase) -> ValidationResult:
        """Validate if phase requirements are met"""
        errors = []
        warnings = []
        suggestions = []
        
        phase_milestones = self.milestones.get(phase, [])
        completed = sum(1 for m in phase_milestones if m["complete"])
        total = len(phase_milestones)
        
        if phase == RoboticsPhase.CONCEPT:
            if not self.robot_type:
                errors.append("Robot type not specified")
            if completed < total:
                warnings.append(f"Only {completed}/{total} concept milestones complete")
        
        elif phase == RoboticsPhase.REQUIREMENTS:
            if not self.operating_environment:
                errors.append("Operating environment not defined")
            if self.budget["estimated_total"] == 0:
                warnings.append("Budget not estimated")
        
        elif phase == RoboticsPhase.MECHANICAL_DESIGN:
            if self.dof == 0:
                warnings.append("Degrees of freedom not specified")
        
        elif phase == RoboticsPhase.ELECTRONICS_DESIGN:
            if not self.microcontroller:
                errors.append("Microcontroller not selected")
            if not self.sensors:
                warnings.append("No sensors specified")
        
        elif phase == RoboticsPhase.SOFTWARE_DEVELOPMENT:
            if not self.control_system:
                warnings.append("Control system type not specified")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    async def advance_phase(self, next_phase: RoboticsPhase) -> bool:
        """Advance to next project phase"""
        self.phase_history.append({
            "from": self.current_phase,
            "to": next_phase,
            "timestamp": datetime.now().isoformat()
        })
        self.current_phase = next_phase
        return True
    
    def get_available_phases(self) -> list:
        """Get all robotics phases"""
        return list(RoboticsPhase)
    
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
        """Provide help relevant to current robotics phase"""
        phase_help = {
            RoboticsPhase.CONCEPT: "Concept phase: Define what the robot will do and key requirements",
            RoboticsPhase.REQUIREMENTS: "Requirements phase: Specify technical performance, environment, and safety",
            RoboticsPhase.MECHANICAL_DESIGN: "Mechanical design: Create CAD models, analyze kinematics, select actuators",
            RoboticsPhase.ELECTRONICS_DESIGN: "Electronics design: Select compute platform, sensors, power system",
            RoboticsPhase.SOFTWARE_DEVELOPMENT: "Software development: Implement control, perception, navigation algorithms",
            RoboticsPhase.INTEGRATION: "Integration: Assemble hardware and deploy software stack",
            RoboticsPhase.TESTING: "Testing: Validate subsystems and overall performance",
            RoboticsPhase.CALIBRATION: "Calibration: Tune sensors and control parameters for optimal performance",
            RoboticsPhase.DEPLOYMENT: "Deployment: Prepare environment, train operators, deploy system"
        }
        
        progress = self.get_phase_progress()
        
        help_text = f"""
Current Phase: {self.current_phase.value}
Progress: {progress['completed_milestones']}/{progress['total_milestones']} milestones ({progress['percent_complete']:.1f}%)

{phase_help.get(self.current_phase, 'Robotics development in progress')}

Pending Milestones:
"""
        
        for milestone in progress['milestones']:
            if not milestone['complete']:
                help_text += f"  • {milestone['name']}\n"
        
        return help_text.strip()


class RoboticsDomain(BaseDomain):
    """Robotics Domain"""
    
    def __init__(self):
        super().__init__(
            name="robotics",
            description="Robotics projects - mobile robots, manipulators, autonomous systems"
        )
        
        # Professional systems integration (lazy initialization)
        self._professional_integration = None
    
    def get_knowledge_extractors(self) -> list:
        """Return robotics knowledge extractors"""
        return [
            KnowledgeExtractor(
                name="kinematics",
                description="Forward/inverse kinematics, DH parameters",
                patterns=["kinematics", "dh parameters", "jacobian"],
                extractor_func=None,
                storage_db="kinematics"
            ),
            KnowledgeExtractor(
                name="control_systems",
                description="PID, MPC, adaptive control",
                patterns=["pid", "control", "feedback"],
                extractor_func=None,
                storage_db="control"
            ),
            KnowledgeExtractor(
                name="slam",
                description="SLAM algorithms and mapping",
                patterns=["slam", "mapping", "localization"],
                extractor_func=None,
                storage_db="slam"
            ),
            KnowledgeExtractor(
                name="sensors",
                description="Sensor specifications and integration",
                patterns=["lidar", "imu", "camera", "encoder"],
                extractor_func=None,
                storage_db="sensors"
            )
        ]
    
    async def create_project(
        self,
        description: str,
        requirements: Optional[Dict[str, Any]] = None
    ) -> RoboticsProjectStateMachine:
        """Create a new robotics project"""
        import uuid
        project_id = f"robot-{uuid.uuid4().hex[:8]}"
        
        project = RoboticsProjectStateMachine(project_id, description)
        
        if requirements:
            if "robot_type" in requirements:
                project.robot_type = RobotType(requirements["robot_type"])
            if "dof" in requirements:
                project.dof = requirements["dof"]
            if "payload_kg" in requirements:
                project.payload_kg = requirements["payload_kg"]
            if "microcontroller" in requirements:
                project.microcontroller = requirements["microcontroller"]
            if "budget" in requirements:
                project.budget["estimated_total"] = requirements["budget"]
        
        return project
    
    def get_deliverable_types(self) -> list:
        """List all robotics deliverables"""
        return [
            DeliverableSpec(
                name="cad_models",
                description="3D CAD models and assembly",
                file_types=["stl", "step", "pdf"],
                generator_func=None,
                required_knowledge=["kinematics"]
            ),
            DeliverableSpec(
                name="bom",
                description="Bill of materials with part numbers",
                file_types=["json", "xlsx"],
                generator_func=None,
                required_knowledge=[]
            ),
            DeliverableSpec(
                name="control_code",
                description="Control system implementation",
                file_types=["py", "cpp", "ino"],
                generator_func=None,
                required_knowledge=["control_systems"]
            ),
            DeliverableSpec(
                name="testing_protocol",
                description="Testing and validation procedures",
                file_types=["json", "pdf"],
                generator_func=None,
                required_knowledge=[]
            )
        ]
    
    async def generate_deliverables(
        self,
        project: RoboticsProjectStateMachine,
        deliverable_types: List[str],
        output_dir: Path
    ) -> Dict[str, Path]:
        """Generate robotics deliverables"""
        generated_files = {}
        
        for deliverable_type in deliverable_types:
            if deliverable_type == "bom":
                bom = self._generate_bom(project)
                file_path = output_dir / "robotics_bom.json"
                with open(file_path, 'w') as f:
                    import json
                    json.dump(bom, f, indent=2)
                generated_files["bom"] = file_path
        
        return generated_files
    
    def _generate_bom(self, project: RoboticsProjectStateMachine) -> Dict[str, Any]:
        """Generate bill of materials"""
        bom = {
            "project": project.description,
            "robot_type": project.robot_type.value if project.robot_type else "generic",
            "components": []
        }
        
        # Add common components based on robot type
        if project.microcontroller == "raspberry_pi":
            bom["components"].append({
                "category": "Compute",
                "part": "Raspberry Pi 4B 8GB",
                "quantity": 1,
                "unit_cost": 75.0
            })
        
        return bom
    
    async def _get_professional_integration(self):
        """Get or initialize professional integration"""
        if self._professional_integration is None:
            from modules.domains.domain_professional_integration import DomainProfessionalIntegration
            self._professional_integration = DomainProfessionalIntegration(
                domain_name="robotics"
            )
            await self._professional_integration.initialize()
            
            # Initialize robotics professional roles
            await self._professional_integration.initialize_roles([
                ("MECHANICAL_ENGINEER", "DESIGN"),
                ("CONTROL_ENGINEER", "ANALYSIS"),
                ("SYSTEMS_ENGINEER", "PLANNING")
            ])
        
        return self._professional_integration
    
    @property
    async def team_orchestrator(self):
        """Get professional team orchestrator"""
        integration = await self._get_professional_integration()
        return integration.team_orchestrator
    
    @property
    async def deliverable_generator(self):
        """Get professional deliverable generator"""
        integration = await self._get_professional_integration()
        return integration.deliverable_generator
    
    @property
    async def cross_learning(self):
        """Get cross-domain learning system"""
        integration = await self._get_professional_integration()
        return integration.cross_learning
    
    @property
    async def workflow_executor(self):
        """Get professional workflow executor"""
        integration = await self._get_professional_integration()
        return integration.workflow_executor
    
    @property
    async def quality_framework(self):
        """Get quality assurance framework"""
        integration = await self._get_professional_integration()
        return integration.quality_framework
    
    async def validate_requirements(
        self,
        requirements: Dict[str, Any]
    ) -> ValidationResult:
        """Validate robotics project requirements"""
        errors = []
        warnings = []
        suggestions = []
        
        if "robot_type" not in requirements:
            errors.append("Robot type is required")
        
        if "operating_environment" not in requirements:
            warnings.append("Operating environment should be specified")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    async def estimate_complexity(
        self,
        project: RoboticsProjectStateMachine
    ) -> ComplexityScore:
        """Estimate robotics project complexity"""
        score = 30.0
        factors = {}
        
        # DOF complexity
        if project.dof > 0:
            dof_score = min(project.dof * 5, 30)
            score += dof_score
            factors["dof_complexity"] = dof_score
        
        # Robot type complexity
        type_scores = {
            RobotType.MOBILE_ROBOT: 20,
            RobotType.MANIPULATOR: 30,
            RobotType.HUMANOID: 50,
            RobotType.DRONE: 25,
            RobotType.INDUSTRIAL: 35
        }
        
        if project.robot_type:
            type_score = type_scores.get(project.robot_type, 25)
            score += type_score
            factors["robot_type"] = type_score
        
        score = min(score, 100)
        
        # Time estimate
        time_estimate = int(score * 2)  # 2 days per complexity point
        
        # Cost estimate
        cost_estimate = time_estimate * 500  # $500/day
        
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
            "kinematics": 0,
            "control_algorithms": 0,
            "slam_methods": 0,
            "sensor_specs": 0
        }
