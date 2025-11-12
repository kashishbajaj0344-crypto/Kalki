# ============================================================
# Kalki v2.3 — cad_drawings.py
# ------------------------------------------------------------
# CAD Drawings Module for 2D Projections and Schematics
# - Generate 2D SVG/DXF projections from 3D OpenSCAD models
# - Create technical drawings and schematics
# - Multiple view generation (top, front, side, isometric)
# - Dimension annotations and measurements
# ============================================================

import os
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import math
import re

from modules.utils.config import get_config
from modules.utils.logging_config import get_logger
from modules.cad_exporter import get_cad_exporter

logger = get_logger("Kalki.CADDrawings")

class CADDrawingGenerator:
    """Generate 2D technical drawings and projections from 3D CAD models"""

    def __init__(self):
        self.cad_exporter = get_cad_exporter()
        self.output_dir = Path("output/drawings")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Standard views for technical drawings
        self.standard_views = {
            'front': {'rotation': [0, 0, 0], 'description': 'Front View'},
            'top': {'rotation': [90, 0, 0], 'description': 'Top View'},
            'side': {'rotation': [0, 90, 0], 'description': 'Right Side View'},
            'isometric': {'rotation': [35.264, 45, 0], 'description': 'Isometric View'},
            'rear': {'rotation': [0, 180, 0], 'description': 'Rear View'},
            'bottom': {'rotation': [-90, 0, 0], 'description': 'Bottom View'},
            'left': {'rotation': [0, -90, 0], 'description': 'Left Side View'}
        }

    async def generate_2d_projection(self, scad_file: str, view: str = 'front',
                                   output_format: str = 'svg',
                                   dimensions: bool = True) -> Dict[str, Any]:
        """
        Generate a 2D projection from a 3D OpenSCAD model

        Args:
            scad_file: Path to OpenSCAD file
            view: View type ('front', 'top', 'side', 'isometric', etc.)
            output_format: Output format ('svg', 'dxf', 'png', 'pdf')
            dimensions: Whether to include dimension annotations

        Returns:
            Dict with projection results
        """
        if view not in self.standard_views:
            return {
                "status": "error",
                "error": f"Unknown view: {view}. Available: {list(self.standard_views.keys())}"
            }

        scad_path = Path(scad_file)
        if not scad_path.exists():
            return {
                "status": "error",
                "error": f"SCAD file not found: {scad_file}"
            }

        # Generate output filename
        output_name = f"{scad_path.stem}_{view}.{output_format}"
        output_path = self.output_dir / output_name

        try:
            # Create modified SCAD file with projection
            projection_scad = await self._create_projection_scad(scad_file, view, dimensions)

            if not projection_scad:
                return {
                    "status": "error",
                    "error": "Failed to create projection SCAD file"
                }

            # Export using OpenSCAD
            result = await self.cad_exporter.export_file(
                projection_scad, output_format, str(output_path)
            )

            if result["status"] == "success":
                # Add view metadata
                result.update({
                    "view": view,
                    "view_description": self.standard_views[view]['description'],
                    "projection_type": "2d_technical_drawing",
                    "dimensions_included": dimensions
                })

                logger.info(f"Generated {view} projection: {output_path}")
            else:
                logger.error(f"Failed to generate {view} projection for {scad_file}")

            # Clean up temporary projection file
            try:
                os.remove(projection_scad)
            except:
                pass

            return result

        except Exception as e:
            logger.exception(f"Error generating 2D projection: {e}")
            return {
                "status": "error",
                "error": str(e),
                "scad_file": scad_file,
                "view": view
            }

    async def generate_technical_drawing_set(self, scad_file: str,
                                           views: Optional[List[str]] = None,
                                           output_format: str = 'svg',
                                           include_dimensions: bool = True) -> Dict[str, Any]:
        """
        Generate a complete set of technical drawings with multiple views

        Args:
            scad_file: Path to OpenSCAD file
            views: List of views to generate (default: ['front', 'top', 'side', 'isometric'])
            output_format: Output format
            include_dimensions: Whether to include dimensions

        Returns:
            Dict with complete drawing set results
        """
        if views is None:
            views = ['front', 'top', 'side', 'isometric']

        results = []
        successful_views = []
        failed_views = []

        for view in views:
            result = await self.generate_2d_projection(
                scad_file, view, output_format, include_dimensions
            )
            results.append(result)

            if result.get("status") == "success":
                successful_views.append(view)
            else:
                failed_views.append(view)

        # Create summary
        return {
            "status": "completed",
            "scad_file": scad_file,
            "total_views": len(views),
            "successful_views": len(successful_views),
            "failed_views": len(failed_views),
            "output_format": output_format,
            "results": results,
            "drawing_set": {
                "title": f"Technical Drawing Set - {Path(scad_file).stem}",
                "views": successful_views,
                "format": output_format,
                "dimensions": include_dimensions
            }
        }

    async def _create_projection_scad(self, scad_file: str, view: str,
                                    include_dimensions: bool) -> Optional[str]:
        """
        Create a modified SCAD file with 2D projection for technical drawing

        Args:
            scad_file: Original SCAD file
            view: View to project
            include_dimensions: Whether to add dimension annotations

        Returns:
            Path to modified SCAD file or None if failed
        """
        try:
            # Read original SCAD file
            with open(scad_file, 'r') as f:
                scad_content = f.read()

            # Get view rotation
            view_config = self.standard_views[view]
            rotation = view_config['rotation']

            # Create projection SCAD
            projection_scad = f"""
// Auto-generated projection for {view} view
// Original file: {scad_file}

module original_model() {{
{scad_content}
}}

// Apply rotation for {view} view
rotate([{rotation[0]}, {rotation[1]}, {rotation[2]}]) {{
    original_model();
}}

// Add dimension annotations if requested
if ({str(include_dimensions).lower()}) {{
    // Basic dimension lines (simplified)
    // In a full implementation, this would analyze the model geometry
    // and add proper dimension annotations
}}
"""

            # Save projection file
            projection_path = Path(scad_file).parent / f"{Path(scad_file).stem}_{view}_projection.scad"
            with open(projection_path, 'w') as f:
                f.write(projection_scad)

            return str(projection_path)

        except Exception as e:
            logger.exception(f"Error creating projection SCAD: {e}")
            return None

    async def generate_schematic(self, scad_file: str, schematic_type: str = 'assembly',
                               output_format: str = 'svg') -> Dict[str, Any]:
        """
        Generate schematic diagrams from CAD models

        Args:
            scad_file: Path to OpenSCAD file
            schematic_type: Type of schematic ('assembly', 'wiring', 'pneumatic', etc.)
            output_format: Output format

        Returns:
            Dict with schematic generation results
        """
        try:
            scad_path = Path(scad_file)
            output_name = f"{scad_path.stem}_{schematic_type}_schematic.{output_format}"
            output_path = self.output_dir / output_name

            # For now, create a basic schematic representation
            # In a full implementation, this would analyze the model structure
            schematic_content = await self._create_schematic_content(scad_file, schematic_type)

            if schematic_content:
                # Save schematic as SVG
                svg_content = self._generate_svg_schematic(schematic_content, schematic_type)
                with open(output_path, 'w') as f:
                    f.write(svg_content)

                return {
                    "status": "success",
                    "scad_file": scad_file,
                    "output_file": str(output_path),
                    "schematic_type": schematic_type,
                    "format": output_format
                }
            else:
                return {
                    "status": "error",
                    "error": "Failed to generate schematic content",
                    "scad_file": scad_file
                }

        except Exception as e:
            logger.exception(f"Error generating schematic: {e}")
            return {
                "status": "error",
                "error": str(e),
                "scad_file": scad_file
            }

    async def _create_schematic_content(self, scad_file: str, schematic_type: str) -> Optional[Dict[str, Any]]:
        """
        Analyze SCAD file and create schematic content

        Args:
            scad_file: Path to SCAD file
            schematic_type: Type of schematic

        Returns:
            Dict with schematic elements or None
        """
        try:
            with open(scad_file, 'r') as f:
                content = f.read()

            # Basic analysis - count common components
            components = {
                'cubes': len(re.findall(r'cube\s*\(', content)),
                'cylinders': len(re.findall(r'cylinder\s*\(', content)),
                'spheres': len(re.findall(r'sphere\s*\(', content)),
                'modules': len(re.findall(r'module\s+\w+', content)),
                'unions': len(re.findall(r'union\s*\(\s*\{', content)),
                'differences': len(re.findall(r'difference\s*\(\s*\{', content))
            }

            return {
                'components': components,
                'total_elements': sum(components.values()),
                'complexity': 'high' if sum(components.values()) > 10 else 'medium' if sum(components.values()) > 5 else 'low'
            }

        except Exception as e:
            logger.exception(f"Error analyzing SCAD for schematic: {e}")
            return None

    def _generate_svg_schematic(self, content: Dict[str, Any], schematic_type: str) -> str:
        """
        Generate SVG representation of schematic

        Args:
            content: Schematic content data
            schematic_type: Type of schematic

        Returns:
            SVG string
        """
        components = content.get('components', {})

        svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="400" height="300" xmlns="http://www.w3.org/2000/svg">
  <title>{schematic_type.title()} Schematic</title>

  <!-- Background -->
  <rect width="400" height="300" fill="#f8f9fa" stroke="#dee2e6" stroke-width="1"/>

  <!-- Title -->
  <text x="200" y="30" text-anchor="middle" font-family="Arial" font-size="16" font-weight="bold">
    {schematic_type.title()} Schematic
  </text>

  <!-- Component counts -->
  <text x="20" y="60" font-family="Arial" font-size="12">Components:</text>
'''

        y_pos = 80
        for component, count in components.items():
            if count > 0:
                svg += f'''
  <text x="30" y="{y_pos}" font-family="Arial" font-size="10">{component.title()}: {count}</text>'''
                y_pos += 15

        svg += '''
  <!-- Simple schematic representation -->
  <circle cx="200" cy="180" r="40" fill="#e9ecef" stroke="#6c757d" stroke-width="2"/>
  <text x="200" y="185" text-anchor="middle" font-family="Arial" font-size="10">CAD Model</text>

  <!-- Connection lines (simplified) -->
  <line x1="160" y1="180" x2="120" y2="150" stroke="#6c757d" stroke-width="1"/>
  <line x1="240" y1="180" x2="280" y2="150" stroke="#6c757d" stroke-width="1"/>
  <line x1="200" y1="140" x2="200" y2="100" stroke="#6c757d" stroke-width="1"/>
</svg>'''

        return svg

    def get_available_views(self) -> List[str]:
        """Get list of available view types"""
        return list(self.standard_views.keys())

    def get_view_description(self, view: str) -> str:
        """Get description for a view type"""
        return self.standard_views.get(view, {}).get('description', 'Unknown view')

# Global drawing generator instance
_drawing_generator = None

def get_drawing_generator() -> CADDrawingGenerator:
    """Get the global CAD drawing generator instance"""
    global _drawing_generator
    if _drawing_generator is None:
        _drawing_generator = CADDrawingGenerator()
    return _drawing_generator

async def generate_technical_drawing(scad_file: str, views: Optional[List[str]] = None,
                                  output_format: str = 'svg') -> Dict[str, Any]:
    """
    Convenience function to generate technical drawings

    Args:
        scad_file: Path to OpenSCAD file
        views: List of views to generate
        output_format: Output format

    Returns:
        Drawing generation results
    """
    generator = get_drawing_generator()
    return await generator.generate_technical_drawing_set(scad_file, views, output_format)

async def generate_svg_projection(scad_file: str, view: str = 'front') -> Dict[str, Any]:
    """
    Convenience function to generate SVG projection

    Args:
        scad_file: Path to OpenSCAD file
        view: View type

    Returns:
        Projection result
    """
    generator = get_drawing_generator()
    return await generator.generate_2d_projection(scad_file, view, 'svg')