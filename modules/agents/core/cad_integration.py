#!/usr/bin/env python3
"""
CADIntegrationAgent - CAD modeling and file generation for robotics
Extends Kalki with 3D modeling, CAD file export, and manufacturing preparation.
"""

import asyncio
import logging
import numpy as np
import os
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json

# Import CAD libraries
try:
    import cadquery as cq
    CADQUERY_AVAILABLE = True
except ImportError:
    CADQUERY_AVAILABLE = False

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

from ..base_agent import BaseAgent, AgentCapability, AgentStatus
from ..safety.guard import SafetyGuard


class CADIntegrationAgent(BaseAgent):
    """
    CADIntegrationAgent provides CAD modeling and file generation capabilities.
    Supports 3D modeling, STL export, and manufacturing preparation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        capabilities = [
            AgentCapability.CAD_MODELING,
            AgentCapability.FILE_EXPORT,
            AgentCapability.MANUFACTURING_PREP
        ]

        super().__init__(
            name="CADIntegrationAgent",
            capabilities=capabilities,
            description="CAD modeling and manufacturing file generation",
            config=config
        )

        # CAD workspace
        self.workspace_dir = Path("data/cad_models")
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # Model templates
        self.templates = {
            'robot_link': self._create_link_template,
            'robot_joint': self._create_joint_template,
            'end_effector': self._create_end_effector_template,
            'mounting_plate': self._create_mounting_plate_template
        }

        # Material properties
        self.materials = {
            'aluminum': {'density': 2700, 'youngs_modulus': 69e9, 'yield_strength': 276e6},
            'steel': {'density': 7850, 'youngs_modulus': 200e9, 'yield_strength': 250e6},
            'plastic': {'density': 1200, 'youngs_modulus': 2e9, 'yield_strength': 50e6},
            'titanium': {'density': 4500, 'youngs_modulus': 110e9, 'yield_strength': 880e6}
        }

        # Safety controls
        self.safety_guard = SafetyGuard()

        self.logger.info("CADIntegrationAgent initialized")

    def _create_link_template(self) -> str:
        """Create URDF link template"""
        return """<link name="{name}">
    <visual>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <geometry>
            <cylinder length="{length}" radius="{radius}"/>
        </geometry>
        <material name="blue">
            <color rgba="0 0 0.8 1.0"/>
        </material>
    </visual>
    <collision>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <geometry>
            <cylinder length="{length}" radius="{radius}"/>
        </geometry>
    </collision>
    <inertial>
        <mass value="{mass}"/>
        <inertia ixx="{ixx}" ixy="0" ixz="0" iyy="{iyy}" iyz="0" izz="{izz}"/>
    </inertial>
</link>"""

    def _create_joint_template(self) -> str:
        """Create URDF joint template"""
        return """<joint name="{name}" type="{type}">
    <parent link="{parent}"/>
    <child link="{child}"/>
    <origin xyz="{x} {y} {z}" rpy="{roll} {pitch} {yaw}"/>
    <axis xyz="{axis_x} {axis_y} {axis_z}"/>
    <limit lower="{lower}" upper="{upper}" effort="{effort}" velocity="{velocity}"/>
    <dynamics damping="{damping}" friction="{friction}"/>
</joint>"""

    def _create_end_effector_template(self) -> str:
        """Create end effector template"""
        return """<link name="{name}">
    <visual>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <geometry>
            <box size="{width} {height} {depth}"/>
        </geometry>
        <material name="red">
            <color rgba="0.8 0 0 1.0"/>
        </material>
    </visual>
    <collision>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <geometry>
            <box size="{width} {height} {depth}"/>
        </geometry>
    </collision>
    <inertial>
        <mass value="{mass}"/>
        <inertia ixx="{ixx}" ixy="0" ixz="0" iyy="{iyy}" iyz="0" izz="{izz}"/>
    </inertial>
</link>"""

    def _create_mounting_plate_template(self) -> str:
        """Create mounting plate template"""
        return """<link name="{name}">
    <visual>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <geometry>
            <box size="{width} {length} {thickness}"/>
        </geometry>
        <material name="gray">
            <color rgba="0.5 0.5 0.5 1.0"/>
        </material>
    </visual>
    <collision>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <geometry>
            <box size="{width} {length} {thickness}"/>
        </geometry>
    </collision>
    <inertial>
        <mass value="{mass}"/>
        <inertia ixx="{ixx}" ixy="0" ixz="0" iyy="{iyy}" iyz="0" izz="{izz}"/>
    </inertial>
</link>"""

    async def initialize(self) -> bool:
        """Initialize the CAD integration agent"""
        try:
            self.status = AgentStatus.READY
            self.logger.info("CADIntegrationAgent initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize CADIntegrationAgent: {e}")
            self.status = AgentStatus.ERROR
            return False

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a CAD integration task"""
        try:
            self.increment_task_count()
            action = task.get('action', '')

            if action == 'generate_robot_cad':
                result = await self.generate_robot_cad(task.get('robot_config', {}))
            elif action == 'generate_custom_part':
                result = await self.generate_custom_part(task.get('specifications', {}))
            elif action == 'analyze_manufacturing':
                result = await self.analyze_manufacturing(task.get('components', {}), task.get('material', 'aluminum'))
            else:
                result = {'error': f'Unknown action: {action}'}

            return {
                'status': 'success',
                'result': result
            }

        except Exception as e:
            self.increment_error_count()
            self.logger.error(f"Task execution failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    async def shutdown(self) -> bool:
        """Shutdown the CAD integration agent"""
        try:
            self.status = AgentStatus.TERMINATED
            self.logger.info("CADIntegrationAgent shutdown successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to shutdown CADIntegrationAgent: {e}")
            return False

    async def generate_robot_cad(self, robot_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate complete CAD model for robot

        Args:
            robot_config: Robot configuration from RoboticsSimulationAgent

        Returns:
            Dict with CAD files and analysis
        """
        try:
            if not CADQUERY_AVAILABLE:
                return {'error': 'CAD libraries not available'}

            # Extract robot parameters
            links = robot_config.get('links', [])
            joints = robot_config.get('joints', [])
            material = robot_config.get('material', 'aluminum')

            # Generate individual components
            components = {}
            for i, link in enumerate(links):
                component_name = f"link_{i+1}"
                components[component_name] = await self.generate_link_cad(link, material)

            for i, joint in enumerate(joints):
                component_name = f"joint_{i+1}"
                components[component_name] = await self.generate_joint_cad(joint, material)

            # Generate assembly
            assembly = await self.generate_assembly(components, robot_config)

            # Export files
            export_files = await self.export_cad_files(assembly, robot_config.get('name', 'robot'))

            # Manufacturing analysis
            manufacturing_analysis = await self.analyze_manufacturing(components, material)

            return {
                'components': components,
                'assembly': assembly,
                'export_files': export_files,
                'manufacturing_analysis': manufacturing_analysis,
                'material': material,
                'total_mass': sum(comp.get('mass', 0) for comp in components.values())
            }

        except Exception as e:
            self.logger.error(f"CAD generation failed: {e}")
            return {'error': str(e)}

    async def generate_link_cad(self, link_config: Dict[str, Any], material: str = 'aluminum') -> Dict[str, Any]:
        """Generate CAD model for robot link"""
        try:
            if not CADQUERY_AVAILABLE:
                return {'error': 'CAD libraries not available'}

            length = link_config.get('length', 0.2)
            diameter = link_config.get('diameter', 0.05)
            wall_thickness = link_config.get('wall_thickness', 0.005)

            # Create hollow cylindrical link
            outer = cq.Workplane("XY").circle(diameter/2).extrude(length)
            inner = cq.Workplane("XY").circle(diameter/2 - wall_thickness).extrude(length)
            link = outer.cut(inner)

            # Add mounting features
            link = link.faces(">Z").workplane().circle(diameter/4).cutThruAll()
            link = link.faces("<Z").workplane().circle(diameter/4).cutThruAll()

            # Calculate properties
            volume = self._calculate_volume(link)
            mass = volume * self.materials[material]['density']

            # Export STL
            filename = f"link_{link_config.get('name', 'unnamed')}.stl"
            filepath = self.workspace_dir / filename
            cq.exporters.export(link, str(filepath), 'STL')

            return {
                'cad_model': link,
                'volume': volume,
                'mass': mass,
                'material': material,
                'dimensions': {'length': length, 'diameter': diameter, 'wall_thickness': wall_thickness},
                'stl_file': str(filepath),
                'bounding_box': self._get_bounding_box(link)
            }

        except Exception as e:
            self.logger.error(f"Link CAD generation failed: {e}")
            return {'error': str(e)}

    async def generate_joint_cad(self, joint_config: Dict[str, Any], material: str = 'steel') -> Dict[str, Any]:
        """Generate CAD model for robot joint"""
        try:
            if not CADQUERY_AVAILABLE:
                return {'error': 'CAD libraries not available'}

            joint_type = joint_config.get('joint_type', 'revolute')
            size = joint_config.get('size', 0.08)

            if joint_type == 'revolute':
                # Create revolute joint housing
                housing = cq.Workplane("XY").rect(size, size).extrude(size/2)
                shaft_hole = cq.Workplane("XY").circle(size/6).extrude(size/2)
                housing = housing.cut(shaft_hole)

                # Add mounting holes
                housing = housing.faces(">Z").workplane().circle(size/8).cutThruAll()
                housing = housing.faces(">Z").workplane().rect(size*0.8, size*0.8).vertices().circle(size/12).cutThruAll()

            elif joint_type == 'prismatic':
                # Create prismatic joint
                housing = cq.Workplane("XY").rect(size, size).extrude(size)
                slider = cq.Workplane("XY").rect(size*0.8, size*0.6).extrude(size*0.8)
                housing = housing.cut(slider.translate((0, 0, size*0.1)))

            else:
                # Fixed joint - simple bracket
                housing = cq.Workplane("XY").rect(size, size/2).extrude(size/4)

            # Calculate properties
            volume = self._calculate_volume(housing)
            mass = volume * self.materials[material]['density']

            # Export STL
            filename = f"joint_{joint_config.get('name', 'unnamed')}.stl"
            filepath = self.workspace_dir / filename
            cq.exporters.export(housing, str(filepath), 'STL')

            return {
                'cad_model': housing,
                'volume': volume,
                'mass': mass,
                'material': material,
                'joint_type': joint_type,
                'dimensions': {'size': size},
                'stl_file': str(filepath),
                'bounding_box': self._get_bounding_box(housing)
            }

        except Exception as e:
            self.logger.error(f"Joint CAD generation failed: {e}")
            return {'error': str(e)}

    async def generate_assembly(self, components: Dict[str, Any], robot_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate robot assembly"""
        try:
            if not CADQUERY_AVAILABLE:
                return {'error': 'CAD libraries not available'}

            assembly = cq.Assembly()

            # Position components
            current_position = np.array([0, 0, 0])
            for i, (name, component) in enumerate(components.items()):
                if 'cad_model' in component:
                    # Position component along robot arm
                    position = current_position
                    assembly.add(component['cad_model'], name=name, loc=cq.Location(cq.Vector(*position)))

                    # Update position for next component
                    if 'link' in name:
                        length = component.get('dimensions', {}).get('length', 0.2)
                        current_position += np.array([length, 0, 0])

            return {
                'assembly': assembly,
                'component_count': len(components),
                'total_components': list(components.keys())
            }

        except Exception as e:
            self.logger.error(f"Assembly generation failed: {e}")
            return {'error': str(e)}

    async def export_cad_files(self, assembly: Dict[str, Any], robot_name: str) -> Dict[str, Any]:
        """Export CAD files in multiple formats"""
        try:
            export_dir = self.workspace_dir / robot_name
            export_dir.mkdir(exist_ok=True)

            exported_files = {}

            # Export assembly STL
            stl_file = export_dir / f"{robot_name}_assembly.stl"
            if 'assembly' in assembly and hasattr(assembly['assembly'], 'toCompound'):
                cq.exporters.export(assembly['assembly'].toCompound(), str(stl_file), 'STL')
                exported_files['stl'] = str(stl_file)

            # Export STEP file (if supported)
            try:
                step_file = export_dir / f"{robot_name}_assembly.step"
                if 'assembly' in assembly and hasattr(assembly['assembly'], 'toCompound'):
                    cq.exporters.export(assembly['assembly'].toCompound(), str(step_file), 'STEP')
                    exported_files['step'] = str(step_file)
            except:
                self.logger.warning("STEP export not supported")

            # Export component info as JSON
            info_file = export_dir / f"{robot_name}_info.json"
            component_info = {
                'robot_name': robot_name,
                'export_date': str(asyncio.get_event_loop().time()),
                'components': list(assembly.get('total_components', [])),
                'files': exported_files
            }

            with open(info_file, 'w') as f:
                json.dump(component_info, f, indent=2)

            exported_files['info'] = str(info_file)

            return exported_files

        except Exception as e:
            self.logger.error(f"CAD export failed: {e}")
            return {'error': str(e)}

    async def analyze_manufacturing(self, components: Dict[str, Any], material: str) -> Dict[str, Any]:
        """Analyze manufacturing requirements and costs"""
        try:
            total_volume = 0
            total_mass = 0
            manufacturing_processes = set()
            material_cost = 0

            for component in components.values():
                if 'volume' in component:
                    total_volume += component['volume']
                if 'mass' in component:
                    total_mass += component['mass']

                # Determine manufacturing processes
                if 'link' in component.get('name', '').lower():
                    manufacturing_processes.add('cnc_machining')
                    manufacturing_processes.add('drilling')
                elif 'joint' in component.get('name', '').lower():
                    manufacturing_processes.add('cnc_machining')
                    manufacturing_processes.add('threading')

            # Estimate costs (rough approximations)
            material_cost_per_kg = {
                'aluminum': 5.0,
                'steel': 3.0,
                'plastic': 2.0,
                'titanium': 50.0
            }.get(material, 5.0)

            machining_cost_per_hour = 50.0  # USD
            estimated_machining_hours = total_volume * 1000  # Rough estimate

            total_cost = (total_mass * material_cost_per_kg) + (estimated_machining_hours * machining_cost_per_hour)

            return {
                'total_volume_m3': total_volume,
                'total_mass_kg': total_mass,
                'material': material,
                'manufacturing_processes': list(manufacturing_processes),
                'estimated_cost_usd': total_cost,
                'machining_hours': estimated_machining_hours,
                'material_cost': total_mass * material_cost_per_kg,
                'labor_cost': estimated_machining_hours * machining_cost_per_hour
            }

        except Exception as e:
            self.logger.error(f"Manufacturing analysis failed: {e}")
            return {'error': str(e)}

    def _calculate_volume(self, cad_model) -> float:
        """Calculate volume of CAD model in m³"""
        try:
            if hasattr(cad_model, 'val'):
                # CadQuery object
                return cad_model.val().Volume() / 1e9  # Convert mm³ to m³
            return 0.0
        except:
            return 0.0

    def _get_bounding_box(self, cad_model) -> Dict[str, List[float]]:
        """Get bounding box of CAD model"""
        try:
            if hasattr(cad_model, 'val'):
                bbox = cad_model.val().BoundingBox()
                return {
                    'min': [bbox.xmin/1000, bbox.ymin/1000, bbox.zmin/1000],  # Convert to meters
                    'max': [bbox.xmax/1000, bbox.ymax/1000, bbox.zmax/1000]
                }
            return {'min': [0, 0, 0], 'max': [0, 0, 0]}
        except:
            return {'min': [0, 0, 0], 'max': [0, 0, 0]}

    async def generate_custom_part(self, specifications: Dict[str, Any]) -> Dict[str, Any]:
        """Generate custom CAD part from specifications"""
        try:
            if not CADQUERY_AVAILABLE:
                return {'error': 'CAD libraries not available'}

            part_type = specifications.get('type', 'block')
            dimensions = specifications.get('dimensions', {})
            material = specifications.get('material', 'aluminum')

            if part_type == 'block':
                width = dimensions.get('width', 0.1)
                height = dimensions.get('height', 0.1)
                depth = dimensions.get('depth', 0.1)

                part = cq.Workplane("XY").rect(width, height).extrude(depth)

            elif part_type == 'cylinder':
                diameter = dimensions.get('diameter', 0.1)
                height = dimensions.get('height', 0.1)

                part = cq.Workplane("XY").circle(diameter/2).extrude(height)

            elif part_type == 'sphere':
                diameter = dimensions.get('diameter', 0.1)

                part = cq.Workplane("XY").sphere(diameter/2)

            else:
                return {'error': f'Unsupported part type: {part_type}'}

            # Calculate properties
            volume = self._calculate_volume(part)
            mass = volume * self.materials[material]['density']

            # Export
            filename = f"custom_{part_type}_{hash(str(specifications))}.stl"
            filepath = self.workspace_dir / filename
            cq.exporters.export(part, str(filepath), 'STL')

            return {
                'cad_model': part,
                'part_type': part_type,
                'dimensions': dimensions,
                'volume': volume,
                'mass': mass,
                'material': material,
                'stl_file': str(filepath),
                'bounding_box': self._get_bounding_box(part)
            }

        except Exception as e:
            self.logger.error(f"Custom part generation failed: {e}")
            return {'error': str(e)}

    async def optimize_for_manufacturing(self, cad_model: Any, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize CAD model for manufacturing"""
        try:
            optimizations = []

            # Check for thin walls
            min_wall_thickness = constraints.get('min_wall_thickness', 0.002)
            # Note: Actual wall thickness analysis would require mesh analysis

            # Check for overhangs
            max_overhang_angle = constraints.get('max_overhang_angle', 45)
            # Note: Overhang analysis would require advanced geometric analysis

            # Suggest material changes
            material_suggestions = []
            if constraints.get('cost_optimization'):
                material_suggestions.append('Consider switching to aluminum for cost reduction')

            if constraints.get('weight_optimization'):
                material_suggestions.append('Consider carbon fiber for weight reduction')

            optimizations.extend(material_suggestions)

            return {
                'optimizations_applied': optimizations,
                'manufacturing_constraints': constraints,
                'feasibility_score': 0.85  # Placeholder
            }

        except Exception as e:
            self.logger.error(f"Manufacturing optimization failed: {e}")
            return {'error': str(e)}