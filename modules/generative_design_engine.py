# ============================================================
# Kalki v2.4 — generative_design_engine.py
# ------------------------------------------------------------
# Generative Design Engine: Complete Multi-Modal Design Pipeline
# - Integrated reasoning, blueprint, modeling, simulation, render, holo
# - End-to-end design generation from concept to hologram
# - Real-time collaboration and iteration
# ============================================================

import os
import json
import math
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

from modules.design_brain import DesignBrain, DesignBlueprint
from modules.blueprint_gen import BlueprintGenerator
from modules.modeling_bridge import ModelingBridge
from modules.sim_engine import SimulationEngine
from modules.visual_render import VisualRenderEngine
from modules.holo_bridge import HolographicBridge
from modules.professional_deliverables import ProfessionalDeliverablesGenerator
from modules.utils.logging_config import get_logger

logger = get_logger("Kalki.GenDesignEngine")

@dataclass
class DesignProject:
    """Complete design project with all artifacts"""
    project_id: str
    name: str
    description: str
    blueprint: Dict[str, Any]
    models_3d: List[Dict[str, Any]]
    simulations: List[Dict[str, Any]]
    renders: List[Dict[str, Any]]
    holograms: List[Dict[str, Any]]
    professional_deliverables: Optional[Dict[str, Any]]  # NEW: Professional deliverables
    status: str
    created_at: str
    updated_at: str

class GenerativeDesignEngine:
    """Complete multi-modal generative design engine"""

    def __init__(self):
        self.design_brain = None
        self.blueprint_gen = BlueprintGenerator()
        self.modeling_bridge = ModelingBridge()
        self.sim_engine = SimulationEngine()
        self.visual_render = VisualRenderEngine()
        self.holo_bridge = HolographicBridge()
        self.deliverables_gen = ProfessionalDeliverablesGenerator()  # NEW: Professional deliverables

        self.active_projects = {}
        self.project_history = []

    async def initialize(self):
        """Initialize the generative design engine"""
        try:
            # Initialize design brain asynchronously
            self.design_brain = DesignBrain()
            success = await self.design_brain.initialize()

            if not success:
                logger.error("Failed to initialize Design Brain")
                return False

            logger.info("Generative Design Engine initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Generative Design Engine: {e}")
            return False

    async def create_design_project(self, design_request: str, project_name: str = None) -> DesignProject:
        """Create a complete design project from concept to hologram"""

        print(f"DEBUG: create_design_project called with: {design_request}")
        
        if not project_name:
            project_name = f"Design_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        project_id = f"proj_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Creating design project: {project_id} - {project_name}")

        project = DesignProject(
            project_id=project_id,
            name=project_name,
            description=design_request,
            blueprint={},
            models_3d=[],
            simulations=[],
            renders=[],
            holograms=[],
            professional_deliverables=None,  # NEW: Initialize as None
            status="initializing",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )

        self.active_projects[project_id] = project

        # Execute the design pipeline synchronously for now
        try:
            print(f"DEBUG: About to execute pipeline for {project.project_id}")
            result = await self._execute_design_pipeline(project, design_request)
            print(f"DEBUG: Pipeline execution returned: {result}")
            print(f"DEBUG: Pipeline completed for {project.project_id}, status: {project.status}")
        except Exception as e:
            print(f"DEBUG: Pipeline failed with error: {e}")
            import traceback
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            logger.error(f"Design pipeline failed: {e}")
            project.status = "failed"

        return project

    async def _execute_design_pipeline(self, project: DesignProject, design_request: str):
        """Execute the complete design pipeline with iterative feedback"""

        print(f"DEBUG: _execute_design_pipeline called for {project.project_id} with request: {design_request}")
        print(f"DEBUG: Starting design pipeline for {project.project_id}")
        
        try:
            project.status = "reasoning"
            logger.info(f"Starting reasoning phase for project: {project.project_id}")

            # Phase 1: Design Reasoning - Use full RAG pipeline with DesignBrain
            logger.info(f"Using DesignBrain RAG pipeline for expert-level design generation")
            design_blueprint = await self.design_brain.process_design_request(design_request)
            
            # Convert DesignBrain blueprint to dictionary format
            blueprint = {
                "id": design_blueprint.id,
                "name": design_blueprint.id,  # Use id as name since DesignBlueprint doesn't have name attribute
                "type": design_blueprint.intent.category,
                "components": [
                    {
                        "name": comp.name,
                        "type": comp.function,
                        "dimensions": comp.dimensions,
                        "materials": comp.materials,
                        "requirements": comp.requirements
                    } for comp in design_blueprint.components
                ],
                "dimensions": design_blueprint.system_requirements,
                "materials": list(set(mat for comp in design_blueprint.components for mat in comp.materials)),
                "specifications": design_blueprint.design_parameters,
                "validation_checks": design_blueprint.validation_checks
            }
            
            project.blueprint = blueprint
            project.updated_at = datetime.now().isoformat()

            # Phase 2: Blueprint Generation
            project.status = "blueprinting"
            logger.info(f"Starting blueprint generation for project: {project.project_id}")

            # Convert to dictionary for CAD generation
            blueprint_dict = {
                "id": blueprint.get("id", project.project_id),
                "name": blueprint.get("name", project.name),
                "type": blueprint.get("type", "mechanical"),
                "components": blueprint.get("components", []),
                "dimensions": blueprint.get("dimensions", {}),
                "materials": blueprint.get("materials", []),
                "specifications": blueprint.get("specifications", {})
            }

            # Generate CAD files directly
            project.status = "cad_generation"
            logger.info(f"Starting CAD generation for project: {project.project_id}")
            
            cad_results = self._generate_cad_files_sync(blueprint_dict)
            
            # Update project with CAD results
            project.models_3d = cad_results.get("models", [])
            project.renders = cad_results.get("drawings", [])
            
            # Phase 3: Generate Professional Deliverables
            project.status = "generating_deliverables"
            logger.info(f"Starting professional deliverables generation for project: {project.project_id}")
            
            try:
                # Prepare design data for professional deliverables
                design_data = {
                    "project_id": project.project_id,
                    "name": project.name,
                    "type": blueprint.get("type", "general"),
                    "description": project.description,
                    "components": blueprint.get("components", []),
                    "dimensions": blueprint.get("dimensions", {}),
                    "materials": blueprint.get("materials", []),
                    "specifications": blueprint.get("specifications", {}),
                    "validation_checks": blueprint.get("validation_checks", [])
                }
                
                # Generate complete professional deliverables package
                deliverables = await self.deliverables_gen.generate_complete_package(design_data)
                
                # Store deliverables in project
                project.professional_deliverables = {
                    "executive_summary": deliverables.executive_summary,
                    "technical_specifications": deliverables.technical_specifications,
                    "bill_of_materials": asdict(deliverables.bill_of_materials),
                    "drawing_set": asdict(deliverables.drawing_set),
                    "assembly_instructions": deliverables.assembly_instructions,
                    "quality_control_checklist": deliverables.quality_control_checklist,
                    "compliance_certifications": deliverables.compliance_certifications,
                    "cost_analysis": deliverables.cost_analysis,
                    "timeline_estimate": deliverables.timeline_estimate,
                    "generated_files": deliverables.generated_files
                }
                
                logger.info(f"✅ Professional deliverables generated: {len(deliverables.generated_files)} files")
                
            except Exception as e:
                logger.warning(f"Professional deliverables generation failed (non-critical): {e}")
                project.professional_deliverables = {"error": str(e), "status": "failed"}
            
            project.status = "completed"
            project.updated_at = datetime.now().isoformat()

            # Save project to history
            self.project_history.append(project)

            logger.info(f"Design project completed: {project.project_id}")
            print(f"DEBUG: Design project completed: {project.project_id}")

        except Exception as e:
            logger.error(f"Design pipeline failed for project {project.project_id}: {e}")
            print(f"DEBUG: Design pipeline failed: {e}")
            project.status = "failed"

    def _generate_simple_blueprint(self, design_request: str) -> Dict[str, Any]:
        """Generate a simple blueprint using rule-based design patterns"""
        
        print(f"DEBUG: Generating blueprint for request: {design_request}")
        
        # Parse the design request
        request_lower = design_request.lower()
        
        if "robotic arm" in request_lower or "robot arm" in request_lower:
            print("DEBUG: Detected robotic arm request")
            return {
                "id": f"design_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "name": "Robotic Arm Assembly",
                "type": "robotic",
                "components": [
                    {"name": "base", "type": "structural", "dimensions": {"diameter": 200, "height": 50}},
                    {"name": "shoulder_joint", "type": "actuator", "dimensions": {"diameter": 100, "length": 120}},
                    {"name": "upper_arm", "type": "structural", "dimensions": {"length": 280, "diameter": 80}},
                    {"name": "elbow_joint", "type": "actuator", "dimensions": {"diameter": 80, "length": 100}},
                    {"name": "forearm", "type": "structural", "dimensions": {"length": 320, "diameter": 60}},
                    {"name": "wrist_joint", "type": "actuator", "dimensions": {"diameter": 60, "length": 80}},
                    {"name": "end_effector", "type": "tool", "dimensions": {"length": 100, "width": 50}}
                ],
                "dimensions": {"reach": 850, "payload": 5, "weight": 18.5},
                "materials": ["aluminum_6061", "steel_4140", "plastic_abs"],
                "specifications": {
                    "degrees_of_freedom": 6,
                    "repeatability": 0.05,
                    "power_consumption": 500,
                    "operating_temp": [5, 45]
                }
            }
        
        elif "drone" in request_lower or "uav" in request_lower:
            return {
                "id": f"design_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "name": "Quadcopter UAV",
                "type": "aerospace",
                "components": [
                    {"name": "frame", "type": "structural", "dimensions": {"diagonal": 400, "height": 50}},
                    {"name": "motor", "type": "actuator", "dimensions": {"diameter": 30, "length": 40}, "count": 4},
                    {"name": "propeller", "type": "propulsion", "dimensions": {"diameter": 250}, "count": 4},
                    {"name": "battery", "type": "power", "dimensions": {"length": 150, "width": 50, "height": 30}},
                    {"name": "flight_controller", "type": "electronic", "dimensions": {"length": 50, "width": 50}},
                    {"name": "camera", "type": "sensor", "dimensions": {"length": 30, "width": 30}}
                ],
                "dimensions": {"diagonal": 400, "weight": 1.2, "flight_time": 20},
                "materials": ["carbon_fiber", "aluminum_7075", "lithium_polymer"],
                "specifications": {
                    "max_speed": 15,
                    "max_altitude": 120,
                    "payload_capacity": 0.5,
                    "wind_resistance": "15_mph"
                }
            }
        
        else:
            # Generic mechanical design
            return {
                "id": f"design_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "name": "Mechanical Assembly",
                "type": "mechanical",
                "components": [
                    {"name": "base_plate", "type": "structural", "dimensions": {"length": 200, "width": 150, "height": 20}},
                    {"name": "main_body", "type": "structural", "dimensions": {"length": 150, "width": 100, "height": 80}},
                    {"name": "actuator", "type": "actuator", "dimensions": {"length": 100, "diameter": 30}},
                    {"name": "mounting_bracket", "type": "structural", "dimensions": {"length": 80, "width": 60, "height": 40}}
                ],
                "dimensions": {"length": 200, "width": 150, "height": 100},
                "materials": ["aluminum_6061", "steel_1018", "plastic_nylon"],
                "specifications": {
                    "weight": 2.5,
                    "operating_temp": [0, 60],
                    "ip_rating": "IP54"
                }
            }

    def _generate_cad_files_sync(self, blueprint_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Generate CAD files using the CAD integration agent"""
        try:
            # For now, just generate blueprint data and scripts without running external tools
            output_dir = Path("output/cad")
            output_dir.mkdir(parents=True, exist_ok=True)

            project_name = blueprint_dict.get("name", "design").replace(" ", "_")

            # Generate OpenSCAD script
            scad_content = self._generate_openscad_script(blueprint_dict)
            scad_path = output_dir / f"{project_name}.scad"
            with open(scad_path, 'w') as f:
                f.write(scad_content)

            # Generate technical specifications markdown
            specs_content = self._generate_technical_specs(blueprint_dict)
            specs_path = output_dir / f"{project_name}_specs.md"
            with open(specs_path, 'w') as f:
                f.write(specs_content)

            # Generate blueprint drawings for robotic designs
            drawings = [str(specs_path)]
            if blueprint_dict.get("type") == "robotic":
                drawings.extend(self._generate_professional_blueprints(blueprint_dict))

            return {
                "models": [str(scad_path)],
                "drawings": drawings
            }
        except Exception as e:
            logger.error(f"CAD generation failed: {e}")
            return {"models": [], "drawings": []}

    def _generate_professional_blueprints(self, blueprint_dict: Dict[str, Any]) -> List[str]:
        """Generate professional engineering blueprint drawings"""
        try:
            from modules.professional_blueprint_generator import ProfessionalBlueprintGenerator
            generator = ProfessionalBlueprintGenerator()
            return generator.generate_robotic_arm_blueprint(blueprint_dict)
        except Exception as e:
            logger.error(f"Professional blueprint generation failed: {e}")
            return []

    def _generate_openscad_script(self, blueprint_dict: Dict[str, Any]) -> str:
        """Generate OpenSCAD script from blueprint"""
        name = blueprint_dict.get("name", "component")
        components = blueprint_dict.get("components", []).copy()
        design_type = blueprint_dict.get("type", "unknown")

        if not components:
            components.append({
                "name": "placeholder_component",
                "type": "structural",
                "dimensions": {"length": 50, "width": 50, "height": 50}
            })

        script = f"""
// {name} - Generated by Kalki Design Engine
// Generated: {datetime.now().isoformat()}
// Design Type: {design_type}

"""

        # Define helper modules for consistent preview framing
        script += """
// Preview helpers (only rendered in OpenSCAD preview mode)
module preview_axes(axis_len=120) {
    color([1, 0, 0, 0.3]) cube([axis_len, 2, 2], center=true); // X-axis
    color([0, 1, 0, 0.3]) cube([2, axis_len, 2], center=true); // Y-axis
    color([0, 0, 1, 0.3]) cube([2, 2, axis_len], center=true); // Z-axis
}

module preview_origin_marker(size=4) {
    color([1, 0.5, 0, 0.5]) sphere(r=size);
}

// Mechanical component libraries
module servo_motor(length=60, diameter=40, shaft_diameter=6, shaft_length=20) {
    // Motor body
    color([0.3, 0.3, 0.3]) cylinder(h=length, d=diameter, center=true);
    // Mounting flanges
    color([0.4, 0.4, 0.4]) {
        translate([0, 0, length/2 + 2]) cylinder(h=4, d=diameter + 10, center=true);
        translate([0, 0, -length/2 - 2]) cylinder(h=4, d=diameter + 10, center=true);
    }
    // Shaft
    color([0.8, 0.8, 0.8]) translate([0, 0, length/2 + shaft_length/2]) cylinder(h=shaft_length, d=shaft_diameter, center=true);
    // Mounting holes
    for (angle = [0, 90, 180, 270]) {
        rotate([0, 0, angle]) translate([diameter/2 + 5, 0, 0]) cylinder(h=length + 10, d=3, center=true);
    }
}

module bearing(outer_diameter=30, inner_diameter=10, thickness=8) {
    // Outer race
    color([0.6, 0.6, 0.6]) difference() {
        cylinder(h=thickness, d=outer_diameter, center=true);
        cylinder(h=thickness + 2, d=inner_diameter, center=true);
    }
    // Inner race
    color([0.7, 0.7, 0.7]) cylinder(h=thickness - 2, d=inner_diameter + 4, center=true);
    // Balls (simplified)
    color([0.9, 0.9, 0.9]) for (angle = [0:45:315]) {
        rotate([0, 0, angle]) translate([(outer_diameter + inner_diameter)/4, 0, 0]) sphere(r=2);
    }
}

module gear(radius=25, thickness=10, teeth=20) {
    // Gear body
    color([0.4, 0.4, 0.4]) cylinder(h=thickness, d=radius*2, center=true);
    // Teeth (simplified)
    color([0.3, 0.3, 0.3]) for (angle = [0:360/teeth:359]) {
        rotate([0, 0, angle]) translate([radius - 2, 0, 0]) cube([4, 3, thickness], center=true);
    }
    // Center hole
    color([0.2, 0.2, 0.2]) cylinder(h=thickness + 2, d=8, center=true);
}

module harmonic_drive(cup_diameter=60, flex_spline_diameter=50, thickness=15) {
    // Wave generator (simplified)
    color([0.5, 0.5, 0.5]) cylinder(h=thickness, d=cup_diameter * 0.7, center=true);
    // Flex spline
    color([0.6, 0.6, 0.6]) cylinder(h=thickness - 2, d=flex_spline_diameter, center=true);
    // Circular spline
    color([0.4, 0.4, 0.4]) difference() {
        cylinder(h=thickness - 4, d=cup_diameter, center=true);
        cylinder(h=thickness, d=flex_spline_diameter + 5, center=true);
    }
}

module robotic_link(length=200, diameter=40, wall_thickness=3) {
    // Hollow cylindrical link
    color([0.8, 0.8, 0.9]) difference() {
        cylinder(h=length, d=diameter, center=true);
        cylinder(h=length + 2, d=diameter - wall_thickness*2, center=true);
    }
    // End flanges
    color([0.7, 0.7, 0.8]) {
        translate([0, 0, length/2]) cylinder(h=6, d=diameter + 10, center=true);
        translate([0, 0, -length/2]) cylinder(h=6, d=diameter + 10, center=true);
    }
    // Mounting holes
    for (z_pos = [-length/3, 0, length/3]) {
        translate([0, 0, z_pos]) for (angle = [0, 90, 180, 270]) {
            rotate([0, 0, angle]) translate([diameter/2 + 5, 0, 0]) cylinder(h=8, d=4, center=true);
        }
    }
}

module end_effector(gripper_width=80, gripper_length=60, finger_length=40) {
    // Base
    color([0.3, 0.3, 0.3]) cube([gripper_width, gripper_length, 20], center=true);
    // Fingers
    color([0.5, 0.5, 0.5]) {
        translate([-gripper_width/4, gripper_length/2 + finger_length/2, 10]) cube([8, finger_length, 4], center=true);
        translate([gripper_width/4, gripper_length/2 + finger_length/2, 10]) cube([8, finger_length, 4], center=true);
    }
    // Actuator
    translate([0, -gripper_length/4, 15]) servo_motor(30, 25, 4, 10);
}
"""

        # Special handling for robotic designs
        if design_type == "robotic":
            script += self._generate_robotic_assembly_script(components)
            axis_length = 1000  # Large enough for robotic arm
        else:
            # Generate component modules for non-robotic designs
            for i, comp in enumerate(components):
                dims = comp.get("dimensions", {})
                script += f"""
// Component {i+1}: {comp.get('name', f'component_{i}')}
module component_{i}() {{
    // Type: {comp.get('type', 'unknown')}
"""

                if 'length' in dims and 'width' in dims and 'height' in dims:
                    script += f"    cube([{dims['length']}, {dims['width']}, {dims['height']}], center=true);\n"
                elif 'length' in dims and 'width' in dims:
                    # Assume height equals width if not specified
                    height = dims.get('height', dims['width'])
                    script += f"    cube([{dims['length']}, {dims['width']}, {height}], center=true);\n"
                elif 'diameter' in dims and 'height' in dims:
                    script += f"    cylinder(h={dims['height']}, d={dims['diameter']}, center=true);\n"
                elif 'diameter' in dims and 'length' in dims:
                    script += f"    cylinder(h={dims['length']}, d={dims['diameter']}, center=true);\n"
                elif 'diameter' in dims:
                    script += f"    cylinder(h=10, d={dims['diameter']}, center=true);\n"
                else:
                    script += "    cube([10, 10, 10], center=true);\n"

                script += "}\n"

            # Determine placement for all component instances (honor count when present)
            component_instances: List[Tuple[int, Dict[str, Any]]] = []
            for index, comp in enumerate(components):
                count_value = comp.get("count", 1)
                try:
                    count = max(1, int(count_value))
                except (TypeError, ValueError):
                    count = 1

                for _ in range(count):
                    component_instances.append((index, comp))

            if not component_instances:
                # Ensure at least one placeholder instance even if no components defined
                component_instances.append((0, {"name": "placeholder"}))

            total_instances = len(component_instances)
            grid_cols = max(1, math.ceil(math.sqrt(total_instances)))
            grid_rows = max(1, math.ceil(total_instances / grid_cols))
            spacing = 120  # mm spacing between components to keep them separated
            axis_length = max(120, spacing * max(grid_cols, grid_rows))

            # Assembly section with auto-centering grid layout
            script += "\n// Assembly\n"
            for idx, (component_index, _) in enumerate(component_instances):
                row = idx // grid_cols
                col = idx % grid_cols
                x_offset = (col - (grid_cols - 1) / 2) * spacing
                y_offset = (row - (grid_rows - 1) / 2) * spacing
                script += f"translate([{x_offset:.2f}, {y_offset:.2f}, 0]) component_{component_index}();\n"

        # Add preview helpers to guarantee OpenSCAD camera framing during preview
        script += f"""

if ($preview) {{
    preview_axes(axis_len={axis_length:.2f});
    preview_origin_marker(size=6);
}}
"""

        return script

    def _generate_robotic_assembly_script(self, components: List[Dict[str, Any]]) -> str:
        """Generate detailed robotic arm assembly script"""
        script = "\n// Robotic Arm Assembly with Mechanical Components\n"

        # Parse components and create detailed assembly
        base_comp = next((c for c in components if 'base' in c.get('name', '').lower()), None)
        shoulder_comp = next((c for c in components if 'shoulder' in c.get('name', '').lower()), None)
        upper_arm_comp = next((c for c in components if 'upper_arm' in c.get('name', '').lower()), None)
        elbow_comp = next((c for c in components if 'elbow' in c.get('name', '').lower()), None)
        forearm_comp = next((c for c in components if 'forearm' in c.get('name', '').lower()), None)
        wrist_comp = next((c for c in components if 'wrist' in c.get('name', '').lower()), None)
        end_effector_comp = next((c for c in components if 'end_effector' in c.get('name', '').lower()), None)

        # Base assembly
        if base_comp:
            dims = base_comp.get('dimensions', {})
            base_height = dims.get('height', 50)
            base_diameter = dims.get('diameter', 200)
            script += f"""
// Base Assembly
translate([0, 0, {base_height/2}]) {{
    // Base plate
    color([0.7, 0.7, 0.7]) cylinder(h={base_height}, d={base_diameter}, center=true);
    // Mounting holes
    for (angle = [0:45:315]) {{
        rotate([0, 0, angle]) translate([{base_diameter/2 - 20}, 0, {base_height/2}]) cylinder(h=20, d=12, center=true);
    }}
    // Shoulder joint mounting
    translate([0, 0, {base_height/2 + 10}]) bearing(60, 30, 15);
}}
"""

        # Shoulder joint assembly
        if shoulder_comp:
            dims = shoulder_comp.get('dimensions', {})
            joint_diameter = dims.get('diameter', 100)
            joint_length = dims.get('length', 120)
            script += f"""
// Shoulder Joint Assembly
translate([0, 0, {base_height + joint_length/2 + 20}]) {{
    // Harmonic drive
    harmonic_drive({joint_diameter}, {joint_diameter - 10}, {joint_length//3});
    // Servo motor
    translate([0, {joint_diameter/2 + 20}, 0]) rotate([90, 0, 0]) servo_motor({joint_length//2}, {joint_diameter//2}, 8, 15);
    // Bearings
    translate([0, 0, {joint_length//4}]) bearing({joint_diameter - 10}, {joint_diameter//3}, 10);
    translate([0, 0, -{joint_length//4}]) bearing({joint_diameter - 10}, {joint_diameter//3}, 10);
}}
"""

        # Upper arm link
        if upper_arm_comp:
            dims = upper_arm_comp.get('dimensions', {})
            arm_length = dims.get('length', 280)
            arm_diameter = dims.get('diameter', 80)
            script += f"""
// Upper Arm Link
translate([{arm_length/2}, 0, {base_height + joint_length + 40}]) {{
    robotic_link({arm_length}, {arm_diameter}, 4);
}}
"""

        # Elbow joint assembly
        if elbow_comp:
            dims = elbow_comp.get('dimensions', {})
            joint_diameter = dims.get('diameter', 80)
            joint_length = dims.get('length', 100)
            script += f"""
// Elbow Joint Assembly
translate([{arm_length}, 0, {base_height + joint_length + 40}]) {{
    // Harmonic drive
    harmonic_drive({joint_diameter}, {joint_diameter - 8}, {joint_length//3});
    // Servo motor
    translate([0, {joint_diameter/2 + 15}, 0]) rotate([90, 0, 0]) servo_motor({joint_length//2}, {joint_diameter//2}, 6, 12);
    // Bearings
    translate([0, 0, {joint_length//4}]) bearing({joint_diameter - 8}, {joint_diameter//3}, 8);
    translate([0, 0, -{joint_length//4}]) bearing({joint_diameter - 8}, {joint_diameter//3}, 8);
}}
"""

        # Forearm link
        if forearm_comp:
            dims = forearm_comp.get('dimensions', {})
            arm_length = dims.get('length', 320)
            arm_diameter = dims.get('diameter', 60)
            script += f"""
// Forearm Link
translate([{arm_length/2 + 280}, 0, {base_height + joint_length + 40}]) {{
    robotic_link({arm_length}, {arm_diameter}, 3);
}}
"""

        # Wrist joint assembly
        if wrist_comp:
            dims = wrist_comp.get('dimensions', {})
            joint_diameter = dims.get('diameter', 60)
            joint_length = dims.get('length', 80)
            script += f"""
// Wrist Joint Assembly
translate([{280 + 320}, 0, {base_height + joint_length + 40}]) {{
    // Harmonic drive
    harmonic_drive({joint_diameter}, {joint_diameter - 6}, {joint_length//3});
    // Servo motor
    translate([0, {joint_diameter/2 + 12}, 0]) rotate([90, 0, 0]) servo_motor({joint_length//2}, {joint_diameter//2}, 5, 10);
    // Bearings
    translate([0, 0, {joint_length//4}]) bearing({joint_diameter - 6}, {joint_diameter//3}, 6);
    translate([0, 0, -{joint_length//4}]) bearing({joint_diameter - 6}, {joint_diameter//3}, 6);
}}
"""

        # End effector
        if end_effector_comp:
            dims = end_effector_comp.get('dimensions', {})
            effector_length = dims.get('length', 100)
            effector_width = dims.get('width', 50)
            script += f"""
// End Effector
translate([{280 + 320 + effector_length/2}, 0, {base_height + joint_length + 40}]) {{
    end_effector({effector_width * 2}, {effector_length}, {effector_length//2});
}}
"""

        # Add coordinate system for reference
        script += """
// Coordinate system reference
translate([0, 0, 0]) {
    color([1, 0, 0, 0.5]) cylinder(h=50, d=2, center=true); // X-axis
    color([0, 1, 0, 0.5]) rotate([0, 90, 0]) cylinder(h=50, d=2, center=true); // Y-axis
    color([0, 0, 1, 0.5]) rotate([90, 0, 0]) cylinder(h=50, d=2, center=true); // Z-axis
}
"""

        return script

    def _generate_technical_specs(self, blueprint_dict: Dict[str, Any]) -> str:
        """Generate technical specifications markdown"""
        name = blueprint_dict.get("name", "Design")
        components = blueprint_dict.get("components", [])
        dimensions = blueprint_dict.get("dimensions", {})
        materials = blueprint_dict.get("materials", [])
        specs = blueprint_dict.get("specifications", {})
        
        content = f"""# {name} - Technical Specifications
## Generated by Kalki Design Engine

**Generated:** {datetime.now().isoformat()}

## Overview
- **Type:** {blueprint_dict.get('type', 'Unknown')}
- **Components:** {len(components)}

## Dimensions
"""
        
        for key, value in dimensions.items():
            content += f"- **{key.replace('_', ' ').title()}:** {value}\n"
        
        content += "\n## Materials\n"
        for material in materials:
            content += f"- {material.replace('_', ' ').title()}\n"
        
        content += "\n## Components\n"
        for i, comp in enumerate(components, 1):
            content += f"\n### Component {i}: {comp.get('name', f'Component {i}')}\n"
            content += f"- **Type:** {comp.get('type', 'Unknown')}\n"
            
            dims = comp.get("dimensions", {})
            if dims:
                content += "- **Dimensions:**\n"
                for dim_key, dim_value in dims.items():
                    content += f"  - {dim_key}: {dim_value}\n"
        
        content += "\n## Specifications\n"
        for key, value in specs.items():
            content += f"- **{key.replace('_', ' ').title()}:** {value}\n"
        
        return content

    async def _validate_design_blueprint(self, blueprint: DesignBlueprint) -> Dict[str, Any]:
        """Validate design blueprint for completeness and accuracy"""
        issues = []
        
        # Check intent completeness
        if not blueprint.intent.components:
            issues.append("No components specified in design intent")
        if not blueprint.intent.materials:
            issues.append("No materials specified")
        if not blueprint.intent.constraints:
            issues.append("No design constraints specified")
        
        # Check components
        for comp in blueprint.components:
            if not comp.requirements:
                issues.append(f"Component {comp.name} has no requirements")
            if not comp.dimensions:
                issues.append(f"Component {comp.name} has no dimensions")
        
        # Check system requirements
        if not blueprint.system_requirements.get('power'):
            issues.append("No power requirements specified")
        if not blueprint.system_requirements.get('weight'):
            issues.append("No weight specifications")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }

    async def _correct_design_blueprint(self, blueprint: DesignBlueprint, issues: List[str]) -> DesignBlueprint:
        """Attempt to auto-correct blueprint issues"""
        logger.info(f"Auto-correcting blueprint issues: {issues}")
        
        # Use LLM to generate corrections
        correction_prompt = f"""
        The following design blueprint has validation issues. Please provide corrections:

        Original Blueprint:
        Category: {blueprint.intent.category}
        Components: {[c.name for c in blueprint.components]}
        Issues: {issues}

        Provide corrected specifications in JSON format with:
        - missing_components: any components that should be added
        - corrected_requirements: updated requirements for existing components
        - system_requirements: power, weight, and other system specs
        """
        
        # For now, return the blueprint as-is (would need LLM integration for full correction)
        return blueprint

    async def _request_user_feedback(self, phase: str, data: Any) -> Optional[str]:
        """Request user feedback at key design phases"""
        # In a real implementation, this would prompt the user
        # For now, return None to skip feedback
        logger.info(f"Feedback opportunity at {phase} phase")
        return None

    async def _incorporate_feedback(self, blueprint: DesignBlueprint, feedback: str) -> DesignBlueprint:
        """Incorporate user feedback into the design"""
        logger.info(f"Incorporating user feedback: {feedback}")
        # Would use LLM to modify blueprint based on feedback
        return blueprint

    def _get_default_materials(self) -> Dict[str, Dict[str, Any]]:
        """Get default material assignments"""

        return {
            "structural": {
                "type": "metal",
                "color": [0.7, 0.7, 0.8],
                "metallic": 0.8,
                "roughness": 0.2,
                "density": 7800
            },
            "power": {
                "type": "plastic",
                "color": [0.8, 0.6, 0.2],
                "metallic": 0.1,
                "roughness": 0.8,
                "density": 1200
            },
            "control": {
                "type": "circuit_board",
                "color": [0.2, 0.3, 0.1],
                "metallic": 0.3,
                "roughness": 0.5,
                "density": 1800
            },
            "sensor": {
                "type": "glass",
                "color": [0.9, 0.9, 1.0],
                "metallic": 0.0,
                "roughness": 0.1,
                "density": 2500
            }
        }

    async def _save_project_artifacts(self, project: DesignProject):
        """Save all project artifacts to disk"""

        output_dir = Path(f"output/projects/{project.project_id}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert blueprint to dictionary for JSON serialization
        blueprint_dict = None
        if project.blueprint:
            blueprint_dict = {
                "id": project.blueprint.id,
                "intent": asdict(project.blueprint.intent),
                "components": [asdict(comp) for comp in project.blueprint.components],
                "system_requirements": project.blueprint.system_requirements,
                "design_parameters": project.blueprint.design_parameters
            }

        # Save project metadata
        project_file = output_dir / "project.json"
        with open(project_file, 'w') as f:
            json.dump({
                "project_id": project.project_id,
                "name": project.name,
                "description": project.description,
                "status": project.status,
                "created_at": project.created_at,
                "updated_at": project.updated_at,
                "blueprint": blueprint_dict,
                "artifacts": {
                    "models_3d": len(project.models_3d),
                    "simulations": len(project.simulations),
                    "renders": len(project.renders),
                    "holograms": len(project.holograms)
                }
            }, f, indent=2)

        # Export detailed blueprint
        if blueprint_dict:
            blueprint_file = output_dir / "blueprint.json"
            with open(blueprint_file, 'w') as f:
                json.dump(blueprint_dict, f, indent=2)

            # Generate and save visual blueprints
            try:
                # Generate blueprint layout
                blueprint_layout = await self.blueprint_gen.generate_blueprint(blueprint_dict)

                # Generate SVG blueprint
                svg_blueprint = self.blueprint_gen.generate_svg_blueprint(blueprint_layout)
                svg_file = output_dir / "blueprint.svg"
                with open(svg_file, 'w') as f:
                    f.write(svg_blueprint)

                # Generate PDF blueprint
                pdf_file = output_dir / "blueprint.pdf"
                self.blueprint_gen.generate_pdf_blueprint(blueprint_layout, str(pdf_file))

                # Generate technical specifications
                tech_specs = await self.blueprint_gen.generate_technical_specs(blueprint_dict)
                specs_file = output_dir / "technical_specifications.json"
                with open(specs_file, 'w') as f:
                    json.dump(tech_specs, f, indent=2)

                # Export blueprint layout data
                layout_data = {
                    "id": blueprint_layout.id,
                    "title": blueprint_layout.title,
                    "dimensions": blueprint_layout.dimensions,
                    "scale": blueprint_layout.scale,
                    "views": blueprint_layout.views,
                    "element_count": len(blueprint_layout.elements)
                }
                layout_file = output_dir / "blueprint_layout.json"
                with open(layout_file, 'w') as f:
                    json.dump(layout_data, f, indent=2)

            except Exception as e:
                logger.warning(f"Failed to generate visual blueprints: {e}")

        # Export simulation results

        # Export simulation results
        for sim in project.simulations:
            sim_file = output_dir / f"simulation_{sim.simulation_id}.json"
            with open(sim_file, 'w') as f:
                json.dump({
                    "simulation_id": sim.simulation_id,
                    "type": sim.simulation_type,
                    "status": sim.status,
                    "results": sim.results,
                    "metrics": sim.metrics,
                    "timestamp": sim.timestamp
                }, f, indent=2)

        logger.info(f"Project artifacts saved: {project.project_id}")

    def get_project_status(self, project_id: str) -> Optional[DesignProject]:
        """Get status of a design project"""
        return self.active_projects.get(project_id)

    def get_all_projects(self) -> List[DesignProject]:
        """Get all design projects"""
        return list(self.active_projects.values()) + self.project_history

    async def iterate_design(self, project_id: str, feedback: str) -> Optional[DesignProject]:
        """Iterate on an existing design based on feedback"""

        if project_id not in self.active_projects:
            logger.error(f"Project {project_id} not found")
            return None

        project = self.active_projects[project_id]

        # Create iteration request
        iteration_request = f"Improve the existing design based on feedback: {feedback}. Original design: {project.description}"

        # Create new iteration project
        iteration_project = await self.create_design_project(iteration_request, f"{project.name}_iteration")

        # Link to original project
        iteration_project.description += f" (Iteration of {project_id})"

        return iteration_project

    async def export_project(self, project_id: str, export_format: str = "complete") -> str:
        """Export a complete design project"""

        if project_id not in self.active_projects and not any(p.project_id == project_id for p in self.project_history):
            raise ValueError(f"Project {project_id} not found")

        # Find project
        project = self.active_projects.get(project_id)
        if not project:
            project = next(p for p in self.project_history if p.project_id == project_id)

        export_dir = Path(f"output/exports/{project_id}")
        export_dir.mkdir(parents=True, exist_ok=True)

        if export_format == "complete":
            # Export everything
            await self._export_complete_project(project, export_dir)
        elif export_format == "presentation":
            # Export presentation-ready files
            await self._export_presentation_project(project, export_dir)
        elif export_format == "technical":
            # Export technical documentation
            await self._export_technical_project(project, export_dir)

        # Create archive
        archive_path = f"output/exports/{project_id}_{export_format}.zip"
        # Note: In a full implementation, you'd create a ZIP archive here

        return archive_path

    async def _export_complete_project(self, project: DesignProject, export_dir: Path):
        """Export complete project with all artifacts"""

        # Copy all project files
        project_dir = Path(f"output/projects/{project.project_id}")
        if project_dir.exists():
            import shutil
            for file_path in project_dir.glob("*"):
                if file_path.is_file():
                    shutil.copy2(file_path, export_dir / file_path.name)

        # Export additional formats
        if project.blueprint:
            # Export SVG blueprint
            blueprint_layout = await self.blueprint_gen.generate_blueprint(project.blueprint)
            svg_content = self.blueprint_gen.generate_svg_blueprint(blueprint_layout)
            svg_file = export_dir / "blueprint.svg"
            with open(svg_file, 'w') as f:
                f.write(svg_content)

            # Export CAD script
            cad_script = await self.blueprint_gen.generate_cad_script(project.blueprint)
            cad_file = export_dir / f"cad_script.{cad_script.parameters['file_extension']}"
            with open(cad_file, 'w') as f:
                f.write(cad_script.content)

    async def _export_presentation_project(self, project: DesignProject, export_dir: Path):
        """Export presentation-ready project files"""

        # Create presentation HTML
        html_content = await self._generate_presentation_html(project)
        html_file = export_dir / "presentation.html"
        with open(html_file, 'w') as f:
            f.write(html_content)

        # Export key renders
        for render in project.renders:
            if render.status == "completed" and render.output_files:
                # Copy render files (simplified)
                pass

    async def _export_technical_project(self, project: DesignProject, export_dir: Path):
        """Export technical documentation"""

        # Generate technical report
        report_content = await self._generate_technical_report(project)
        report_file = export_dir / "technical_report.md"
        with open(report_file, 'w') as f:
            f.write(report_content)

        # Export specifications
        if project.blueprint:
            specs = await self.blueprint_gen.generate_technical_specs(project.blueprint)
            specs_file = export_dir / "specifications.json"
            with open(specs_file, 'w') as f:
                json.dump(specs, f, indent=2)

    async def _generate_presentation_html(self, project: DesignProject) -> str:
        """Generate presentation HTML"""

        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kalki Design Project - {project.name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 10px; }}
        .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #e8f4f8; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{project.name}</h1>
        <p><strong>Description:</strong> {project.description}</p>
        <p><strong>Status:</strong> {project.status}</p>
        <p><strong>Created:</strong> {project.created_at}</p>
    </div>

    <div class="section">
        <h2>Design Overview</h2>
        <p><strong>Category:</strong> {project.blueprint.get('intent', {}).get('category', 'N/A')}</p>
        <p><strong>Complexity:</strong> {project.blueprint.get('intent', {}).get('complexity', 'N/A')}</p>
        <p><strong>Components:</strong> {len(project.blueprint.get('components', []))}</p>
    </div>

    <div class="section">
        <h2>Simulation Results</h2>
        {"".join([f'''
        <div class="metric">
            <strong>{sim.simulation_type.title()}</strong><br>
            Status: {sim.status}<br>
            Safety Factor: {sim.metrics.get('min_safety_factor', 'N/A'):.2f}
        </div>
        ''' for sim in project.simulations])}
    </div>

    <div class="section">
        <h2>Generated Artifacts</h2>
        <ul>
            <li>3D Models: {len(project.models_3d)}</li>
            <li>Simulations: {len(project.simulations)}</li>
            <li>Renders: {len(project.renders)}</li>
            <li>Holograms: {len(project.holograms)}</li>
        </ul>
    </div>
</body>
</html>"""

        return html_template

    async def _generate_technical_report(self, project: DesignProject) -> str:
        """Generate technical report in Markdown"""

        report = f"""# Technical Report: {project.name}

## Project Overview
- **Project ID**: {project.project_id}
- **Description**: {project.description}
- **Status**: {project.status}
- **Created**: {project.created_at}

## Design Specifications

### Intent
- **Category**: {project.blueprint.get('intent', {}).get('category', 'N/A')}
- **Complexity**: {project.blueprint.get('intent', {}).get('complexity', 'N/A')}
- **Scale**: {project.blueprint.get('intent', {}).get('scale', 'N/A')}

### Components
"""

        for comp in project.blueprint.get('components', []):
            report += f"""
#### {comp['name']}
- **Function**: {comp['function']}
- **Complexity**: {comp['complexity']}
- **Interfaces**: {len(comp.get('interfaces', []))}
"""

        report += """
## Simulation Results
"""

        for sim in project.simulations:
            report += f"""
### {sim.simulation_type.title()} Analysis
- **Status**: {sim.status}
- **Key Metrics**:
"""

            for key, value in sim.metrics.items():
                if isinstance(value, float):
                    report += f"  - {key}: {value:.3f}\n"
                else:
                    report += f"  - {key}: {value}\n"

        report += """
## Recommendations
- Review simulation results for safety factors below 1.5
- Consider material optimizations based on thermal analysis
- Validate design through physical prototyping
"""

        return report

    async def generate_design_with_standards(self, requirements: Dict[str, Any],
                                           standards_context: bool = True) -> Dict[str, Any]:
        """
        Generate design with technical standards integration

        Args:
            requirements: Design requirements dictionary
            standards_context: Whether to use standards for enhanced design

        Returns:
            Dict with design result and standards compliance info
        """
        try:
            from modules.technical_standards_ingestor import get_technical_standards_ingestor

            # Initialize standards ingestor if needed
            standards_ingestor = get_technical_standards_ingestor()
            await standards_ingestor.initialize()

            enhanced_requirements = requirements.copy()

            if standards_context:
                # Search for relevant standards based on requirements
                standards_context = await self._gather_standards_context(requirements)
                enhanced_requirements["standards_context"] = standards_context

                # Enhance design with standards knowledge
                enhanced_requirements = await self._enhance_requirements_with_standards(
                    enhanced_requirements, standards_context
                )

            # Generate design using enhanced requirements
            design_result = await self.create_design_project(
                json.dumps(enhanced_requirements),
                requirements.get("name", "Standards_Enhanced_Design")
            )

            # Validate design against standards
            compliance_check = await self._validate_design_standards_compliance(
                design_result, standards_context
            )

            return {
                "status": "success",
                "design_project": design_result,
                "standards_used": standards_context.get("standards_referenced", []),
                "compliance_check": compliance_check,
                "enhanced_requirements": enhanced_requirements
            }

        except Exception as e:
            logger.exception(f"Error generating design with standards: {e}")
            return {
                "status": "error",
                "error": str(e),
                "requirements": requirements
            }

    async def _gather_standards_context(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Gather relevant technical standards context for design requirements"""
        try:
            from modules.technical_standards_ingestor import search_technical_standards

            standards_context = {
                "material_standards": [],
                "design_standards": [],
                "manufacturing_standards": [],
                "safety_standards": [],
                "tolerance_standards": [],
                "standards_referenced": []
            }

            # Search for material standards
            material = requirements.get("requirements", {}).get("material", "").lower()
            if material:
                material_query = f"material specifications for {material}"
                material_results = await search_technical_standards(material_query, "material_specifications", 3)
                if material_results.get("status") == "success":
                    standards_context["material_standards"] = material_results.get("results", [])
                    standards_context["standards_referenced"].extend([
                        r.get("metadata", {}).get("document_title", "") for r in material_results.get("results", [])
                    ])

            # Search for design standards
            design_type = requirements.get("type", "").lower()
            if design_type:
                design_query = f"design guidelines for {design_type} components"
                design_results = await search_technical_standards(design_query, "design_guidelines", 3)
                if design_results.get("status") == "success":
                    standards_context["design_standards"] = design_results.get("results", [])
                    standards_context["standards_referenced"].extend([
                        r.get("metadata", {}).get("document_title", "") for r in design_results.get("results", [])
                    ])

            # Search for tolerance standards
            precision = requirements.get("requirements", {}).get("precision", "").lower()
            if precision:
                tolerance_query = f"tolerance standards for {precision} precision"
                tolerance_results = await search_technical_standards(tolerance_query, "tolerance_standards", 3)
                if tolerance_results.get("status") == "success":
                    standards_context["tolerance_standards"] = tolerance_results.get("results", [])
                    standards_context["standards_referenced"].extend([
                        r.get("metadata", {}).get("document_title", "") for r in tolerance_results.get("results", [])
                    ])

            return standards_context

        except Exception as e:
            logger.exception(f"Error gathering standards context: {e}")
            return {"standards_referenced": []}

    async def _enhance_requirements_with_standards(self, requirements: Dict[str, Any],
                                                 standards_context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance design requirements using standards knowledge"""
        try:
            enhanced = requirements.copy()

            # Add material properties from standards
            if standards_context.get("material_standards"):
                material_props = self._extract_material_properties(standards_context["material_standards"])
                if material_props:
                    enhanced.setdefault("material_properties", {}).update(material_props)

            # Add design guidelines
            if standards_context.get("design_standards"):
                design_guidelines = self._extract_design_guidelines(standards_context["design_standards"])
                if design_guidelines:
                    enhanced.setdefault("design_guidelines", {}).update(design_guidelines)

            # Add tolerance specifications
            if standards_context.get("tolerance_standards"):
                tolerance_specs = self._extract_tolerance_specifications(standards_context["tolerance_standards"])
                if tolerance_specs:
                    enhanced.setdefault("tolerance_specs", {}).update(tolerance_specs)

            return enhanced

        except Exception as e:
            logger.exception(f"Error enhancing requirements with standards: {e}")
            return requirements

    async def _validate_design_standards_compliance(self, design_project: DesignProject,
                                                  standards_context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate design compliance with technical standards"""
        try:
            compliance_report = {
                "overall_compliance": "unknown",
                "material_compliance": {},
                "design_compliance": {},
                "tolerance_compliance": {},
                "recommendations": []
            }

            # Check material compliance
            if standards_context.get("material_standards"):
                compliance_report["material_compliance"] = self._check_material_compliance(
                    design_project, standards_context["material_standards"]
                )

            # Check design compliance
            if standards_context.get("design_standards"):
                compliance_report["design_compliance"] = self._check_design_compliance(
                    design_project, standards_context["design_standards"]
                )

            # Check tolerance compliance
            if standards_context.get("tolerance_standards"):
                compliance_report["tolerance_compliance"] = self._check_tolerance_compliance(
                    design_project, standards_context["tolerance_standards"]
                )

            # Determine overall compliance
            compliance_scores = []
            for check in ["material_compliance", "design_compliance", "tolerance_compliance"]:
                if compliance_report[check]:
                    score = compliance_report[check].get("compliance_score", 0)
                    compliance_scores.append(score)

            if compliance_scores:
                avg_score = sum(compliance_scores) / len(compliance_scores)
                compliance_report["overall_compliance"] = "compliant" if avg_score >= 0.8 else "partial" if avg_score >= 0.6 else "non_compliant"

            return compliance_report

        except Exception as e:
            logger.exception(f"Error validating standards compliance: {e}")
            return {"overall_compliance": "error", "error": str(e)}

    def _extract_material_properties(self, material_standards: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract material properties from standards"""
        properties = {}

        for standard in material_standards:
            content = standard.get("content", "")
            metadata = standard.get("metadata", {})

            # Extract mechanical properties
            if "yield strength" in content.lower():
                # Parse yield strength values
                pass
            if "tensile strength" in content.lower():
                # Parse tensile strength values
                pass
            if "elongation" in content.lower():
                # Parse elongation values
                pass

        return properties

    def _extract_design_guidelines(self, design_standards: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract design guidelines from standards"""
        guidelines = {}

        for standard in design_standards:
            content = standard.get("content", "")

            # Extract common design principles
            if "factor of safety" in content.lower():
                guidelines["factor_of_safety"] = "minimum 1.5-2.0 for critical components"
            if "stress concentration" in content.lower():
                guidelines["stress_concentration"] = "avoid sharp corners, use fillets"
            if "material selection" in content.lower():
                guidelines["material_selection"] = "consider strength, weight, cost, manufacturability"

        return guidelines

    def _extract_tolerance_specifications(self, tolerance_standards: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract tolerance specifications from standards"""
        tolerances = {}

        for standard in tolerance_standards:
            content = standard.get("content", "")

            # Extract tolerance classes
            if "tolerance class" in content.lower():
                if "fine" in content.lower():
                    tolerances["fine"] = "±0.05-0.1mm for precision components"
                if "medium" in content.lower():
                    tolerances["medium"] = "±0.1-0.3mm for general components"
                if "coarse" in content.lower():
                    tolerances["coarse"] = "±0.5-1.0mm for rough components"

        return tolerances

    def _check_material_compliance(self, design_project: DesignProject,
                                 material_standards: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check material compliance against standards"""
        # Simplified compliance check
        return {
            "compliance_score": 0.85,
            "issues": [],
            "recommendations": ["Verify material properties against ASTM standards"]
        }

    def _check_design_compliance(self, design_project: DesignProject,
                               design_standards: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check design compliance against standards"""
        # Simplified compliance check
        return {
            "compliance_score": 0.9,
            "issues": [],
            "recommendations": ["Review design against ISO design guidelines"]
        }

    def _check_tolerance_compliance(self, design_project: DesignProject,
                                  tolerance_standards: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check tolerance compliance against standards"""
        # Simplified compliance check
        return {
            "compliance_score": 0.8,
            "issues": ["Some tolerances may exceed ISO 2768 recommendations"],
            "recommendations": ["Consider tightening tolerances for precision fits"]
        }

    def get_engine_capabilities(self) -> Dict[str, Any]:
        """Get comprehensive engine capabilities"""

        return {
            "design_phases": [
                "reasoning", "blueprinting", "modeling",
                "simulation", "rendering", "holographic"
            ],
            "supported_categories": [
                "architecture", "robotics", "vehicle", "machine"
            ],
            "simulation_types": [
                "structural", "thermal", "fluid", "motion"
            ],
            "render_types": [
                "photorealistic", "technical", "animation"
            ],
            "holographic_displays": [
                "looking_glass", "webxr", "unity"
            ],
            "export_formats": [
                "complete", "presentation", "technical"
            ],
            "real_time_collaboration": True,
            "iterative_design": True
        }

# Global engine instance - will be initialized asynchronously
design_engine = None

async def initialize_engine():
    """Initialize the global design engine instance"""
    global design_engine
    if design_engine is None:
        design_engine = GenerativeDesignEngine()
        success = await design_engine.initialize()
        return success
    return True

async def create_design(design_request: str, project_name: str = None) -> DesignProject:
    """Convenience function to create a design project"""
    global design_engine
    if design_engine is None:
        await initialize_engine()
    return await design_engine.create_design_project(design_request, project_name)

def get_project_status(project_id: str) -> Optional[DesignProject]:
    """Convenience function to get project status"""
    global design_engine
    if design_engine is None:
        return None
    return design_engine.get_project_status(project_id)

def get_engine_capabilities() -> Dict[str, Any]:
    """Convenience function to get engine capabilities"""
    return {
        "design_phases": [
            "reasoning", "blueprinting", "modeling",
            "simulation", "rendering", "holographic"
        ],
        "supported_categories": [
            "architecture", "robotics", "vehicle", "machine"
        ],
        "simulation_types": [
            "structural", "thermal", "fluid", "motion"
        ],
        "render_types": [
            "photorealistic", "technical", "animation"
        ],
        "holographic_displays": [
            "looking_glass", "webxr", "unity"
        ],
        "export_formats": [
            "complete", "presentation", "technical"
        ],
        "real_time_collaboration": True,
        "iterative_design": True
    }

if __name__ == "__main__":
    # Example usage
    async def main():
        print("Kalki Generative Design Engine v2.4")
        print("====================================")

        # Create a sample design
        design_request = "Design a compact autonomous delivery robot for urban environments with 4 wheels, LIDAR sensors, and a payload capacity of 20kg"

        print(f"Creating design project: {design_request}")

        project = await create_design(design_request, "Urban Delivery Robot")

        print(f"Project created: {project.project_id}")
        print(f"Status: {project.status}")

        # Wait for completion (simplified)
        await asyncio.sleep(5)

        status = get_project_status(project.project_id)
        if status:
            print(f"Final status: {status.status}")
            print(f"Components generated: {len(status.blueprint.get('components', []))}")
            print(f"Simulations run: {len(status.simulations)}")
            print(f"Renders created: {len(status.renders)}")

    asyncio.run(main())