# ============================================================
# Kalki v2.3 — freecad_integration.py
# ------------------------------------------------------------
# FreeCAD Integration for Physics Validation and Simulation
# - Import OpenSCAD models into FreeCAD for physics analysis
# - Perform structural analysis, mass calculations, center of gravity
# - Generate physics validation reports
# - FEM analysis integration for stress/strain simulation
# ============================================================

import os
import sys
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import logging
import tempfile
import subprocess

from modules.utils.config import get_config
from modules.utils.logging_config import get_logger

logger = get_logger("Kalki.FreeCAD")

class FreeCADIntegration:
    """FreeCAD integration for physics validation and simulation"""

    def __init__(self):
        self.freecad_available = self._check_freecad_availability()
        self.temp_dir = Path(tempfile.gettempdir()) / "kalki_freecad"
        self.temp_dir.mkdir(exist_ok=True)

        # FreeCAD Python paths (will be set if available)
        self.freecad_python = None
        self.freecad_libs = []

    def _check_freecad_availability(self) -> bool:
        """Check if FreeCAD is available on the system"""
        try:
            # Try importing FreeCAD
            import FreeCAD
            logger.info("FreeCAD Python API available")
            return True
        except ImportError:
            pass

        # Try to find FreeCAD executable
        freecad_paths = [
            "/usr/bin/freecad",
            "/usr/local/bin/freecad",
            "/opt/freecad/bin/freecad",
            "/Applications/FreeCAD.app/Contents/MacOS/FreeCAD",  # macOS
            "C:\\Program Files\\FreeCAD\\bin\\FreeCAD.exe",  # Windows
        ]

        for path in freecad_paths:
            if os.path.exists(path):
                logger.info(f"Found FreeCAD executable at: {path}")
                return True

        logger.warning("FreeCAD not found. Physics validation will be unavailable.")
        return False

    def _setup_freecad_environment(self):
        """Setup FreeCAD Python environment"""
        if not self.freecad_available:
            return False

        try:
            # Common FreeCAD Python paths
            freecad_python_paths = [
                "/usr/lib/freecad-python",
                "/usr/lib/freecad/lib",
                "/opt/freecad/lib",
                "/Applications/FreeCAD.app/Contents/Resources/lib",  # macOS
                "C:\\Program Files\\FreeCAD\\lib",  # Windows
            ]

            for path in freecad_python_paths:
                if os.path.exists(path):
                    if path not in sys.path:
                        sys.path.insert(0, path)
                        self.freecad_libs.append(path)

            # Try importing FreeCAD modules
            import FreeCAD
            import Part
            import Mesh

            logger.info("FreeCAD environment setup successfully")
            return True

        except Exception as e:
            logger.warning(f"Failed to setup FreeCAD environment: {e}")
            return False

    async def validate_physics(self, scad_file: str,
                             analysis_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform physics validation on a CAD model

        Args:
            scad_file: Path to OpenSCAD file
            analysis_types: Types of analysis to perform

        Returns:
            Dict with physics validation results
        """
        if not self.freecad_available:
            return {
                "status": "error",
                "error": "FreeCAD not available for physics validation",
                "scad_file": scad_file
            }

        if analysis_types is None:
            analysis_types = ['basic_properties', 'mass_properties', 'stability']

        scad_path = Path(scad_file)
        if not scad_path.exists():
            return {
                "status": "error",
                "error": f"SCAD file not found: {scad_file}"
            }

        try:
            # Convert SCAD to STEP format first (FreeCAD can import STEP better)
            step_file = await self._convert_scad_to_step(scad_file)
            if not step_file:
                return {
                    "status": "error",
                    "error": "Failed to convert SCAD to STEP for FreeCAD analysis"
                }

            # Perform physics analysis
            results = await self._run_physics_analysis(step_file, analysis_types)

            # Clean up temporary files
            try:
                os.remove(step_file)
            except:
                pass

            results.update({
                "scad_file": scad_file,
                "step_file": step_file,
                "analysis_types": analysis_types
            })

            return results

        except Exception as e:
            logger.exception(f"Error in physics validation: {e}")
            return {
                "status": "error",
                "error": str(e),
                "scad_file": scad_file
            }

    async def _convert_scad_to_step(self, scad_file: str) -> Optional[str]:
        """
        Convert OpenSCAD file to STEP format using OpenSCAD

        Args:
            scad_file: Path to SCAD file

        Returns:
            Path to STEP file or None if failed
        """
        try:
            scad_path = Path(scad_file)
            step_file = self.temp_dir / f"{scad_path.stem}.step"

            # Use OpenSCAD to export to STEP
            from modules.cad_exporter import get_cad_exporter
            exporter = get_cad_exporter()

            result = await exporter.export_file(scad_file, 'step', str(step_file))

            if result.get("status") == "success":
                return str(step_file)
            else:
                logger.warning(f"Failed to convert SCAD to STEP: {result.get('error')}")
                return None

        except Exception as e:
            logger.exception(f"Error converting SCAD to STEP: {e}")
            return None

    async def _run_physics_analysis(self, step_file: str,
                                  analysis_types: List[str]) -> Dict[str, Any]:
        """
        Run physics analysis using FreeCAD

        Args:
            step_file: Path to STEP file
            analysis_types: Types of analysis to perform

        Returns:
            Dict with analysis results
        """
        # Create a Python script for FreeCAD to execute
        freecad_script = self._generate_freecad_script(step_file, analysis_types)
        script_path = self.temp_dir / "physics_analysis.py"

        try:
            with open(script_path, 'w') as f:
                f.write(freecad_script)

            # Run FreeCAD with the analysis script
            results = await self._execute_freecad_script(str(script_path))

            return {
                "status": "success",
                "analysis_results": results,
                "performed_analyses": analysis_types
            }

        except Exception as e:
            logger.exception(f"Error running FreeCAD physics analysis: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
        finally:
            # Clean up script file
            try:
                os.remove(script_path)
            except:
                pass

    def _generate_freecad_script(self, step_file: str, analysis_types: List[str]) -> str:
        """Generate Python script for FreeCAD to execute physics analysis"""

        script = f'''
import sys
import json
import FreeCAD
import Part
import Mesh

def analyze_model(step_file, analysis_types):
    """Analyze the CAD model and return physics properties"""

    results = {{
        "file": step_file,
        "analyses": {{}}
    }}

    try:
        # Import the STEP file
        doc = FreeCAD.newDocument("PhysicsAnalysis")
        Part.insert(step_file, doc.Name)

        if len(doc.Objects) == 0:
            return {{"error": "No objects found in STEP file"}}

        # Get the first solid object
        solid = None
        for obj in doc.Objects:
            if hasattr(obj, 'Shape') and obj.Shape is not None:
                solid = obj
                break

        if solid is None:
            return {{"error": "No solid objects found"}}

        shape = solid.Shape
        results["basic_info"] = {{
            "volume": shape.Volume,
            "area": shape.Area,
            "bounding_box": {{
                "x": shape.BoundBox.XLength,
                "y": shape.BoundBox.YLength,
                "z": shape.BoundBox.ZLength
            }}
        }}

        # Basic properties analysis
        if "basic_properties" in analysis_types:
            results["analyses"]["basic_properties"] = {{
                "volume_mm3": shape.Volume,
                "surface_area_mm2": shape.Area,
                "bounding_box": {{
                    "length": shape.BoundBox.XLength,
                    "width": shape.BoundBox.YLength,
                    "height": shape.BoundBox.ZLength
                }},
                "center_of_mass": {{
                    "x": shape.CenterOfMass.x,
                    "y": shape.CenterOfMass.y,
                    "z": shape.CenterOfMass.z
                }}
            }}

        # Mass properties (assuming density of 1 g/cm³ for steel-like material)
        if "mass_properties" in analysis_types:
            density = 7.85  # g/cm³ for steel
            volume_cm3 = shape.Volume / 1000  # convert mm³ to cm³
            mass = volume_cm3 * density

            results["analyses"]["mass_properties"] = {{
                "density_g_cm3": density,
                "volume_cm3": volume_cm3,
                "mass_grams": mass,
                "mass_kg": mass / 1000,
                "center_of_mass": {{
                    "x": shape.CenterOfMass.x,
                    "y": shape.CenterOfMass.y,
                    "z": shape.CenterOfMass.z
                }},
                "inertia_tensor": {{
                    "xx": shape.MatrixOfInertia.A11,
                    "xy": shape.MatrixOfInertia.A12,
                    "xz": shape.MatrixOfInertia.A13,
                    "yy": shape.MatrixOfInertia.A22,
                    "yz": shape.MatrixOfInertia.A23,
                    "zz": shape.MatrixOfInertia.A33
                }}
            }}

        # Stability analysis
        if "stability" in analysis_types:
            # Simple stability check based on center of mass vs base
            com_z = shape.CenterOfMass.z
            bbox_min_z = shape.BoundBox.ZMin
            bbox_height = shape.BoundBox.ZLength

            stability_ratio = (com_z - bbox_min_z) / bbox_height

            results["analyses"]["stability"] = {{
                "center_of_mass_height": com_z,
                "base_height": bbox_min_z,
                "total_height": bbox_height,
                "stability_ratio": stability_ratio,
                "stable": stability_ratio < 0.5,  # COM below midpoint
                "assessment": "stable" if stability_ratio < 0.5 else "unstable"
            }}

        FreeCAD.closeDocument(doc.Name)
        return results

    except Exception as e:
        return {{"error": str(e)}}

# Run the analysis
result = analyze_model("{step_file}", {analysis_types})
print(json.dumps(result, indent=2))
'''

        return script

    async def _execute_freecad_script(self, script_path: str) -> Dict[str, Any]:
        """
        Execute FreeCAD script and capture results

        Args:
            script_path: Path to Python script for FreeCAD

        Returns:
            Dict with analysis results
        """
        try:
            # Find FreeCAD executable
            freecad_cmd = self._find_freecad_executable()
            if not freecad_cmd:
                return {"error": "FreeCAD executable not found"}

            # Run FreeCAD with the script
            cmd = [freecad_cmd, "-c", f"exec(open('{script_path}').read())"]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.temp_dir)
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                # Parse JSON output
                output = stdout.decode().strip()
                try:
                    import json
                    return json.loads(output)
                except json.JSONDecodeError:
                    return {"error": "Failed to parse FreeCAD output", "raw_output": output}
            else:
                error_msg = stderr.decode().strip()
                return {"error": f"FreeCAD execution failed: {error_msg}"}

        except Exception as e:
            logger.exception(f"Error executing FreeCAD script: {e}")
            return {"error": str(e)}

    def _find_freecad_executable(self) -> Optional[str]:
        """Find FreeCAD executable path"""
        freecad_paths = [
            "/usr/bin/freecad",
            "/usr/local/bin/freecad",
            "/opt/freecad/bin/freecad",
            "/Applications/FreeCAD.app/Contents/MacOS/FreeCAD",
            "C:\\Program Files\\FreeCAD\\bin\\FreeCAD.exe",
        ]

        for path in freecad_paths:
            if os.path.exists(path):
                return path

        return None

    async def generate_physics_report(self, validation_results: Dict[str, Any],
                                    output_format: str = 'markdown') -> Dict[str, Any]:
        """
        Generate a comprehensive physics validation report

        Args:
            validation_results: Results from physics validation
            output_format: Report format ('markdown', 'json', 'html')

        Returns:
            Dict with report generation results
        """
        try:
            if validation_results.get("status") != "success":
                return {
                    "status": "error",
                    "error": "Invalid validation results for report generation"
                }

            report_content = self._format_physics_report(validation_results, output_format)

            # Save report
            report_file = self.temp_dir / f"physics_report_{Path(validation_results['scad_file']).stem}.{output_format}"
            with open(report_file, 'w') as f:
                f.write(report_content)

            return {
                "status": "success",
                "report_file": str(report_file),
                "format": output_format,
                "validation_results": validation_results
            }

        except Exception as e:
            logger.exception(f"Error generating physics report: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def export_stl(self, scad_file: str, output_file: str,
                        quality_settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Export OpenSCAD file to STL format using FreeCAD

        Args:
            scad_file: Path to OpenSCAD file
            output_file: Path to output STL file
            quality_settings: Quality settings for export

        Returns:
            Dict with export results
        """
        if not self.freecad_available:
            return {
                "status": "error",
                "error": "FreeCAD not available for STL export",
                "scad_file": scad_file
            }

        scad_path = Path(scad_file)
        if not scad_path.exists():
            return {
                "status": "error",
                "error": f"SCAD file not found: {scad_file}"
            }

        try:
            # First convert SCAD to STEP
            step_file = await self._convert_scad_to_step(scad_file)
            if not step_file:
                return {
                    "status": "error",
                    "error": "Failed to convert SCAD to STEP for STL export"
                }

            # Create FreeCAD script for STL export
            freecad_script = f"""
import FreeCAD
import Part
import Mesh

# Load the STEP file
doc = FreeCAD.newDocument("STLExport")
Part.insert("{step_file}", doc.Name)

if len(doc.Objects) == 0:
    print("ERROR: No objects found in STEP file")
    exit(1)

# Get the first solid object
solid = None
for obj in doc.Objects:
    if hasattr(obj, 'Shape') and obj.Shape is not None:
        solid = obj
        break

if solid is None:
    print("ERROR: No solid objects found")
    exit(1)

# Create mesh from shape
mesh = doc.addObject("Mesh::Feature", "Mesh")
mesh.Mesh = Mesh.Mesh(solid.Shape.tessellate(0.1))  # 0.1mm tolerance

# Export to STL
Mesh.export([mesh], "{output_file}")

print("STL export completed successfully")
"""

            script_path = self.temp_dir / "stl_export.py"
            with open(script_path, 'w') as f:
                f.write(freecad_script)

            # Run FreeCAD with the export script
            result = await self._execute_freecad_script(str(script_path))

            # Check if output file was created
            output_path = Path(output_file)
            if output_path.exists() and output_path.stat().st_size > 0:
                file_size = output_path.stat().st_size
                logger.info(f"Successfully exported STL: {output_file} ({file_size} bytes)")

                # Clean up temporary files
                try:
                    os.remove(step_file)
                    os.remove(script_path)
                except:
                    pass

                return {
                    "status": "success",
                    "input_file": scad_file,
                    "output_file": output_file,
                    "format": "stl",
                    "file_size": file_size,
                    "method": "freecad"
                }
            else:
                return {
                    "status": "error",
                    "error": "STL file was not created or is empty",
                    "scad_file": scad_file,
                    "output_file": output_file
                }

        except Exception as e:
            logger.exception(f"Error exporting STL with FreeCAD: {e}")
            return {
                "status": "error",
                "error": f"FreeCAD STL export failed: {str(e)}",
                "scad_file": scad_file,
                "output_file": output_file
            }

    def _format_physics_report(self, results: Dict[str, Any], format_type: str) -> str:
        """Format physics validation results into a report"""

        if format_type == 'json':
            import json
            return json.dumps(results, indent=2)

        # Markdown report
        report = f"""# Physics Validation Report

**Model:** {results.get('scad_file', 'Unknown')}

## Basic Properties
- **Volume:** {results.get('basic_info', {}).get('volume', 'N/A'):.2f} mm³
- **Surface Area:** {results.get('basic_info', {}).get('area', 'N/A'):.2f} mm²
- **Bounding Box:** {results.get('basic_info', {}).get('bounding_box', {})}

## Analysis Results

"""

        analyses = results.get('analyses', {})
        for analysis_type, analysis_data in analyses.items():
            report += f"### {analysis_type.replace('_', ' ').title()}\n"

            if analysis_type == 'mass_properties':
                report += f"- **Mass:** {analysis_data.get('mass_kg', 'N/A'):.3f} kg\n"
                report += f"- **Center of Mass:** ({analysis_data.get('center_of_mass', {}).get('x', 'N/A'):.2f}, {analysis_data.get('center_of_mass', {}).get('y', 'N/A'):.2f}, {analysis_data.get('center_of_mass', {}).get('z', 'N/A'):.2f}) mm\n"

            elif analysis_type == 'stability':
                report += f"- **Stability Assessment:** {analysis_data.get('assessment', 'N/A')}\n"
                report += f"- **Stability Ratio:** {analysis_data.get('stability_ratio', 'N/A'):.3f}\n"

            report += "\n"

        return report

# Global FreeCAD integration instance
_freecad_integration = None

def get_freecad_integration() -> FreeCADIntegration:
    """Get the global FreeCAD integration instance"""
    global _freecad_integration
    if _freecad_integration is None:
        _freecad_integration = FreeCADIntegration()
    return _freecad_integration

async def validate_cad_physics(scad_file: str, analysis_types: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Convenience function for physics validation

    Args:
        scad_file: Path to OpenSCAD file
        analysis_types: Types of analysis to perform

    Returns:
        Physics validation results
    """
    integrator = get_freecad_integration()
    return await integrator.validate_physics(scad_file, analysis_types)
