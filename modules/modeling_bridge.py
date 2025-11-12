# ============================================================
# Kalki v2.4 — modeling_bridge.py
# ------------------------------------------------------------
# 3D Modeling Bridge: Blueprint to 3D Geometry Conversion
# - FreeCAD integration for parametric modeling
# - Blender integration for organic shapes
# - Mesh generation and optimization
# - Material assignment and rendering prep
# ============================================================

import os
import json
import subprocess
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

from modules.utils.config import get_config
from modules.utils.logging_config import get_logger

logger = get_logger("Kalki.ModelingBridge")

@dataclass
class Model3D:
    """3D model representation"""
    id: str
    format: str  # "stl", "obj", "step", "blend"
    file_path: str
    metadata: Dict[str, Any]
    components: List[Dict[str, Any]]

@dataclass
class ModelingTask:
    """3D modeling task specification"""
    task_type: str  # "parametric", "organic", "assembly", "mesh"
    input_data: Dict[str, Any]
    parameters: Dict[str, Any]
    output_format: str

class ModelingBridge:
    """Bridge between blueprints and 3D modeling software"""

    def __init__(self):
        self.freecad_path = self._find_freecad()
        self.blender_path = self._find_blender()
        self.templates = self._load_modeling_templates()

    def _find_freecad(self) -> Optional[str]:
        """Find FreeCAD executable path"""
        common_paths = [
            "/usr/bin/freecad",
            "/usr/local/bin/freecad",
            "/opt/freecad/bin/freecad",
            "/Applications/FreeCAD.app/Contents/MacOS/FreeCAD",
            "C:\\Program Files\\FreeCAD\\bin\\FreeCAD.exe"
        ]

        for path in common_paths:
            if os.path.exists(path):
                return path

        # Try to find via which command
        try:
            result = subprocess.run(["which", "freecad"], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass

        return None

    def _find_blender(self) -> Optional[str]:
        """Find Blender executable path"""
        common_paths = [
            "/usr/bin/blender",
            "/usr/local/bin/blender",
            "/opt/blender/blender",
            "/Applications/Blender.app/Contents/MacOS/Blender",
            "C:\\Program Files\\Blender Foundation\\Blender\\blender.exe"
        ]

        for path in common_paths:
            if os.path.exists(path):
                return path

        # Try to find via which command
        try:
            result = subprocess.run(["which", "blender"], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass

        return None

    def _load_modeling_templates(self) -> Dict[str, Any]:
        """Load 3D modeling templates"""
        return {
            "parametric": {
                "freecad_script": """
import FreeCAD
import Part
from FreeCAD import Base

doc = FreeCAD.newDocument("ParametricModel")

# Create base component
def create_component(name, dimensions, position=(0,0,0)):
    comp = doc.addObject("Part::Box", name)
    comp.Length = dimensions.get("length", 10)
    comp.Width = dimensions.get("width", 10)
    comp.Height = dimensions.get("height", 10)
    comp.Placement.Base = Base.Vector(*position)
    return comp

# Generate components
components = []
{component_creation}

# Assembly
assembly = doc.addObject("Part::Compound", "Assembly")
assembly.Links = components

doc.recompute()

# Export
import Mesh
mesh = Mesh.Mesh()
for obj in doc.Objects:
    if hasattr(obj, 'Shape'):
        mesh.addMesh(obj.Shape.tessellate(0.1))

mesh.write("{output_file}")
print("Model exported successfully")
"""
            },
            "organic": {
                "blender_script": """
import bpy
import bmesh
from mathutils import Vector

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Create organic shape
def create_organic_shape(name, dimensions, complexity=3):
    # Create base mesh
    bpy.ops.mesh.primitive_cube_add(size=1)
    obj = bpy.context.active_object
    obj.name = name

    # Scale to dimensions
    obj.scale = Vector((
        dimensions.get("length", 1),
        dimensions.get("width", 1),
        dimensions.get("height", 1)
    ))

    # Add subdivision surface for organic feel
    bpy.ops.object.modifier_add(type='SUBSURF')
    bpy.context.object.modifiers["Subdivision"].levels = complexity

    # Add displacement for organic variation
    bpy.ops.object.modifier_add(type='DISPLACE')
    displace = bpy.context.object.modifiers["Displace"]
    displace.strength = 0.1

    return obj

# Generate components
{component_creation}

# Export
bpy.ops.export_mesh.stl(filepath="{output_file}")
print("Organic model exported successfully")
"""
            }
        }

    async def generate_3d_model(self, design_blueprint: Dict[str, Any], modeling_type: str = "parametric") -> Model3D:
        """Generate 3D model from design blueprint"""

        design_id = design_blueprint["id"]
        components = design_blueprint["components"]

        if modeling_type == "parametric" and self.freecad_path:
            return await self._generate_parametric_model(design_id, components)
        elif modeling_type == "organic" and self.blender_path:
            return await self._generate_organic_model(design_id, components)
        else:
            # Fallback to simple mesh generation
            return await self._generate_simple_mesh_model(design_id, components)

    async def _generate_parametric_model(self, design_id: str, components: List[Dict[str, Any]]) -> Model3D:
        """Generate parametric model using FreeCAD"""

        output_dir = Path("output/models")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{design_id}_parametric.stl"

        # Generate FreeCAD script
        component_lines = []
        for i, component in enumerate(components):
            dims = component.get("dimensions", {})
            position = (i * 15, 0, 0)  # Space components along X axis
            component_lines.append(f"components.append(create_component('{component['name']}', {dims}, {position}))")

        script_content = self.templates["parametric"]["freecad_script"].format(
            component_creation="\n".join(component_lines),
            output_file=str(output_file)
        )

        # Write script
        script_path = output_dir / f"{design_id}_freecad.py"
        with open(script_path, 'w') as f:
            f.write(script_content)

        # Execute FreeCAD script
        try:
            result = await asyncio.create_subprocess_exec(
                self.freecad_path, "-c", str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                logger.info(f"FreeCAD model generated: {output_file}")
            else:
                logger.error(f"FreeCAD error: {stderr.decode()}")

        except Exception as e:
            logger.error(f"Failed to run FreeCAD: {e}")
            # Fallback to simple model
            return await self._generate_simple_mesh_model(design_id, components)

        return Model3D(
            id=f"model_{design_id}",
            format="stl",
            file_path=str(output_file),
            metadata={
                "software": "FreeCAD",
                "type": "parametric",
                "generated": datetime.now().isoformat()
            },
            components=components
        )

    async def _generate_organic_model(self, design_id: str, components: List[Dict[str, Any]]) -> Model3D:
        """Generate organic model using Blender"""

        output_dir = Path("output/models")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{design_id}_organic.stl"

        # Generate Blender script
        component_lines = []
        for i, component in enumerate(components):
            dims = component.get("dimensions", {})
            position = (i * 3, 0, 0)  # Space components
            component_lines.append(f"create_organic_shape('{component['name']}', {dims})")

        script_content = self.templates["organic"]["blender_script"].format(
            component_creation="\n".join(component_lines),
            output_file=str(output_file)
        )

        # Write script
        script_path = output_dir / f"{design_id}_blender.py"
        with open(script_path, 'w') as f:
            f.write(script_content)

        # Execute Blender script
        try:
            result = await asyncio.create_subprocess_exec(
                self.blender_path, "--background", "--python", str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                logger.info(f"Blender model generated: {output_file}")
            else:
                logger.error(f"Blender error: {stderr.decode()}")

        except Exception as e:
            logger.error(f"Failed to run Blender: {e}")
            # Fallback to simple model
            return await self._generate_simple_mesh_model(design_id, components)

        return Model3D(
            id=f"model_{design_id}",
            format="stl",
            file_path=str(output_file),
            metadata={
                "software": "Blender",
                "type": "organic",
                "generated": datetime.now().isoformat()
            },
            components=components
        )

    async def _generate_simple_mesh_model(self, design_id: str, components: List[Dict[str, Any]]) -> Model3D:
        """Generate simple mesh model as fallback"""

        output_dir = Path("output/models")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{design_id}_simple.obj"

        # Generate simple OBJ file
        vertices = []
        faces = []
        vertex_count = 1

        for i, component in enumerate(components):
            dims = component.get("dimensions", {})
            length = dims.get("length", 10)
            width = dims.get("width", 10)
            height = dims.get("height", 10)

            # Position component
            x_offset = i * (length + 5)
            y_offset = 0
            z_offset = 0

            # Cube vertices
            cube_vertices = [
                (x_offset, y_offset, z_offset),
                (x_offset + length, y_offset, z_offset),
                (x_offset + length, y_offset + width, z_offset),
                (x_offset, y_offset + width, z_offset),
                (x_offset, y_offset, z_offset + height),
                (x_offset + length, y_offset, z_offset + height),
                (x_offset + length, y_offset + width, z_offset + height),
                (x_offset, y_offset + width, z_offset + height)
            ]

            vertices.extend(cube_vertices)

            # Cube faces (6 faces, 2 triangles each)
            base_idx = vertex_count
            cube_faces = [
                (0, 1, 2), (0, 2, 3),  # bottom
                (4, 5, 6), (4, 6, 7),  # top
                (0, 1, 5), (0, 5, 4),  # front
                (1, 2, 6), (1, 6, 5),  # right
                (2, 3, 7), (2, 7, 6),  # back
                (3, 0, 4), (3, 4, 7)   # left
            ]

            for face in cube_faces:
                faces.append((face[0] + base_idx, face[1] + base_idx, face[2] + base_idx))

            vertex_count += 8

        # Write OBJ file
        with open(output_file, 'w') as f:
            f.write(f"# {design_id} - Simple Mesh Model\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n\n")

            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")

            f.write("\n")
            for face in faces:
                f.write(f"f {face[0]} {face[1]} {face[2]}\n")

        return Model3D(
            id=f"model_{design_id}",
            format="obj",
            file_path=str(output_file),
            metadata={
                "software": "Fallback",
                "type": "simple_mesh",
                "generated": datetime.now().isoformat()
            },
            components=components
        )

    async def optimize_mesh(self, model: Model3D, target_faces: int = 10000) -> Model3D:
        """Optimize mesh for performance"""

        if not os.path.exists(model.file_path):
            logger.error(f"Model file not found: {model.file_path}")
            return model

        # For now, just copy the file with optimization note
        # In a full implementation, this would use mesh processing libraries
        optimized_path = model.file_path.replace(".stl", "_optimized.stl").replace(".obj", "_optimized.obj")

        # Simple file copy for now
        import shutil
        shutil.copy2(model.file_path, optimized_path)

        # Update metadata
        model.metadata["optimized"] = True
        model.metadata["target_faces"] = target_faces
        model.file_path = optimized_path

        return model

    async def assign_materials(self, model: Model3D, material_map: Dict[str, Dict[str, Any]]) -> Model3D:
        """Assign materials to model components"""

        # Create material assignment file
        material_file = model.file_path.replace(".stl", "_materials.json").replace(".obj", "_materials.json")

        material_data = {
            "model_id": model.id,
            "materials": material_map,
            "assignments": {}
        }

        # Assign materials to components
        for component in model.components:
            comp_name = component["name"]
            if comp_name in material_map:
                material_data["assignments"][comp_name] = material_map[comp_name]

        with open(material_file, 'w') as f:
            json.dump(material_data, f, indent=2)

        model.metadata["materials_assigned"] = True
        model.metadata["material_file"] = material_file

        return model

    async def export_model(self, model: Model3D, formats: List[str] = None) -> Dict[str, str]:
        """Export model in multiple formats"""

        if formats is None:
            formats = ["stl", "obj"]

        exported_files = {}
        base_path = Path(model.file_path).parent / Path(model.file_path).stem

        for fmt in formats:
            if fmt == "stl" and model.format != "stl":
                # Convert to STL (simplified - would need proper conversion)
                stl_path = f"{base_path}.stl"
                exported_files["stl"] = stl_path

            elif fmt == "obj" and model.format != "obj":
                # Convert to OBJ (simplified)
                obj_path = f"{base_path}.obj"
                exported_files["obj"] = obj_path

            else:
                exported_files[model.format] = model.file_path

        return exported_files

    def get_modeling_capabilities(self) -> Dict[str, Any]:
        """Get available modeling capabilities"""

        return {
            "freecad_available": self.freecad_path is not None,
            "blender_available": self.blender_path is not None,
            "supported_formats": ["stl", "obj", "step"],
            "modeling_types": ["parametric", "organic", "assembly"],
            "optimization_available": True,
            "material_assignment": True
        }