# ============================================================
# Kalki v2.4 — blueprint_gen.py
# ------------------------------------------------------------
# Blueprint Generator: Technical Drawing & CAD Layout Engine
# - SVG schematic generation
# - CAD script creation (OpenSCAD, FreeCAD)
# - Parametric blueprint layouts
# - Technical specification tables
# ============================================================

import os
import json
import math
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.pdfgen import canvas
from reportlab.graphics import renderPDF
from reportlab.graphics.shapes import Drawing, Rect, Circle, Line, String

from modules.utils.config import get_config
from modules.utils.logging_config import get_logger

logger = get_logger("Kalki.BlueprintGen")

@dataclass
class BlueprintElement:
    """Individual element in a blueprint"""
    id: str
    element_type: str  # "rectangle", "circle", "line", "text", "dimension"
    x: float
    y: float
    properties: Dict[str, Any]
    label: Optional[str] = None

@dataclass
class BlueprintLayout:
    """Complete blueprint layout"""
    id: str
    title: str
    elements: List[BlueprintElement]
    dimensions: Dict[str, float]  # width, height
    scale: str
    views: List[str]  # "top", "side", "front", "isometric"

@dataclass
class CADScript:
    """CAD script for 3D modeling"""
    format: str  # "openscad", "freecad", "python"
    content: str
    parameters: Dict[str, Any]

class BlueprintGenerator:
    """Generates technical drawings and CAD layouts"""

    def __init__(self):
        self.templates = self._load_blueprint_templates()
        self.cad_templates = self._load_cad_templates()

    def _load_blueprint_templates(self) -> Dict[str, Any]:
        """Load blueprint templates for different design categories"""
        return {
            "architecture": {
                "views": ["floor_plan", "elevation", "section"],
                "elements": ["walls", "doors", "windows", "utilities"],
                "scale": "1:100"
            },
            "robotics": {
                "views": ["assembly", "detail", "wiring"],
                "elements": ["chassis", "actuators", "sensors", "controller"],
                "scale": "1:10"
            },
            "vehicle": {
                "views": ["side", "top", "front", "detail"],
                "elements": ["frame", "engine", "wheels", "controls"],
                "scale": "1:50"
            },
            "machine": {
                "views": ["assembly", "parts", "dimensions"],
                "elements": ["housing", "mechanism", "power_unit", "controls"],
                "scale": "1:20"
            }
        }

    def _load_cad_templates(self) -> Dict[str, Any]:
        """Load CAD script templates"""
        return {
            "openscad": {
                "basic_shape": """
module {name}() {{
    {shape_type}({dimensions});
}}

{name}();
""",
                "assembly": """
// {title}
// Generated: {timestamp}

{modules}

translate([{position}]) {main_module}();
"""
            },
            "freecad": {
                "basic_part": """
# {title}
# Generated: {timestamp}

import FreeCAD
import Part

doc = FreeCAD.newDocument("{name}")

# Create basic shape
box = doc.addObject("Part::Box", "{name}")
box.Length = {length}
box.Width = {width}
box.Height = {height}

doc.recompute()
"""
            }
        }

    async def generate_blueprint(self, design_blueprint: Dict[str, Any]) -> BlueprintLayout:
        """Generate a complete blueprint layout from design blueprint"""

        design_id = design_blueprint["id"]
        intent = design_blueprint["intent"]
        components = design_blueprint["components"]

        # Determine blueprint type based on category
        category = intent["category"]
        template = self.templates.get(category, self.templates["machine"])

        # Generate layout elements
        elements = []
        element_id = 0

        # Title block
        elements.append(BlueprintElement(
            id=f"elem_{element_id}",
            element_type="text",
            x=50,
            y=50,
            properties={"text": f"{category.upper()} DESIGN - {design_id}", "size": 16, "bold": True},
            label="title"
        ))
        element_id += 1

        # Scale indicator
        elements.append(BlueprintElement(
            id=f"elem_{element_id}",
            element_type="text",
            x=500,
            y=50,
            properties={"text": f"SCALE: {template['scale']}", "size": 12},
            label="scale"
        ))
        element_id += 1

        # Generate component layouts
        y_offset = 100
        for i, component in enumerate(components):
            component_elements = self._generate_component_layout(component, 100, y_offset, element_id)
            elements.extend(component_elements)
            element_id += len(component_elements)
            y_offset += 150

        # Dimension lines and annotations
        dimension_elements = self._add_dimensions(elements, design_blueprint["design_parameters"])
        elements.extend(dimension_elements)

        layout = BlueprintLayout(
            id=f"bp_{design_id}",
            title=f"{category.title()} Design Blueprint",
            elements=elements,
            dimensions={"width": 800, "height": y_offset + 100},
            scale=template["scale"],
            views=template["views"]
        )

        return layout

    def _generate_component_layout(self, component: Dict[str, Any], x: float, y: float, start_id: int) -> List[BlueprintElement]:
        """Generate layout elements for a single component"""

        elements = []
        element_id = start_id

        # Component label
        elements.append(BlueprintElement(
            id=f"elem_{element_id}",
            element_type="text",
            x=x,
            y=y,
            properties={"text": component["name"].upper(), "size": 14, "bold": True},
            label=f"component_{component['name']}"
        ))
        element_id += 1

        # Component bounding box
        dims = component.get("dimensions", {})
        width = dims.get("length", 100)
        height = dims.get("width", 50)

        elements.append(BlueprintElement(
            id=f"elem_{element_id}",
            element_type="rectangle",
            x=x + 50,
            y=y + 20,
            properties={
                "width": width,
                "height": height,
                "stroke": "black",
                "stroke_width": 2,
                "fill": "none"
            },
            label=f"bbox_{component['name']}"
        ))
        element_id += 1

        # Component function text
        elements.append(BlueprintElement(
            id=f"elem_{element_id}",
            element_type="text",
            x=x + 60,
            y=y + 35,
            properties={"text": component["function"][:50] + "...", "size": 10},
            label=f"func_{component['name']}"
        ))
        element_id += 1

        # Interface points
        interfaces = component.get("interfaces", [])
        for i, interface in enumerate(interfaces[:3]):  # Limit to 3 interfaces
            elements.append(BlueprintElement(
                id=f"elem_{element_id}",
                element_type="circle",
                x=x + 70 + (i * 30),
                y=y + height + 30,
                properties={"radius": 3, "fill": "blue"},
                label=f"interface_{component['name']}_{i}"
            ))
            element_id += 1

            elements.append(BlueprintElement(
                id=f"elem_{element_id}",
                element_type="text",
                x=x + 65 + (i * 30),
                y=y + height + 45,
                properties={"text": interface[:10], "size": 8},
                label=f"interface_label_{component['name']}_{i}"
            ))
            element_id += 1

        return elements

    def _add_dimensions(self, elements: List[BlueprintElement], design_params: Dict[str, Any]) -> List[BlueprintElement]:
        """Add dimension lines and annotations"""

        dimension_elements = []
        overall_dims = design_params.get("overall_dimensions", {})

        # Overall dimension lines
        width = overall_dims.get("width", 100)
        height = overall_dims.get("height", 100)

        # Horizontal dimension line (width)
        dimension_elements.append(BlueprintElement(
            id="dim_width",
            element_type="line",
            x=100,
            y=height + 50,
            properties={
                "x2": 100 + width,
                "y2": height + 50,
                "stroke": "red",
                "stroke_width": 1
            },
            label="width_dimension"
        ))

        # Vertical dimension line (height)
        dimension_elements.append(BlueprintElement(
            id="dim_height",
            element_type="line",
            x=width + 20,
            y=50,
            properties={
                "x2": width + 20,
                "y2": 50 + height,
                "stroke": "red",
                "stroke_width": 1
            },
            label="height_dimension"
        ))

        # Dimension text
        dimension_elements.append(BlueprintElement(
            id="dim_width_text",
            element_type="text",
            x=100 + width/2,
            y=height + 35,
            properties={"text": f"W: {width:.1f} units", "size": 10, "color": "red"},
            label="width_text"
        ))

        dimension_elements.append(BlueprintElement(
            id="dim_height_text",
            element_type="text",
            x=width + 35,
            y=50 + height/2,
            properties={"text": f"H: {height:.1f} units", "size": 10, "color": "red"},
            label="height_text"
        ))

        return dimension_elements

    def generate_svg_blueprint(self, layout: BlueprintLayout) -> str:
        """Generate SVG representation of the blueprint"""

        svg_elements = []

        # SVG header
        svg_elements.append(f'<svg width="{layout.dimensions["width"]}" height="{layout.dimensions["height"]}" xmlns="http://www.w3.org/2000/svg">')

        # Background
        svg_elements.append('<rect width="100%" height="100%" fill="white" stroke="black" stroke-width="1"/>')

        # Generate SVG for each element
        for element in layout.elements:
            svg_code = self._element_to_svg(element)
            if svg_code:
                svg_elements.append(svg_code)

        # Close SVG
        svg_elements.append('</svg>')

        return '\n'.join(svg_elements)

    def _element_to_svg(self, element: BlueprintElement) -> Optional[str]:
        """Convert blueprint element to SVG"""

        if element.element_type == "rectangle":
            props = element.properties
            return f'<rect x="{element.x}" y="{element.y}" width="{props["width"]}" height="{props["height"]}" stroke="{props.get("stroke", "black")}" stroke-width="{props.get("stroke_width", 1)}" fill="{props.get("fill", "none")}"/>'

        elif element.element_type == "circle":
            props = element.properties
            return f'<circle cx="{element.x}" cy="{element.y}" r="{props["radius"]}" fill="{props.get("fill", "black")}"/>'

        elif element.element_type == "line":
            props = element.properties
            return f'<line x1="{element.x}" y1="{element.y}" x2="{props["x2"]}" y2="{props["y2"]}" stroke="{props.get("stroke", "black")}" stroke-width="{props.get("stroke_width", 1)}"/>'

        elif element.element_type == "text":
            props = element.properties
            font_weight = "bold" if props.get("bold", False) else "normal"
            return f'<text x="{element.x}" y="{element.y}" font-family="Arial" font-size="{props.get("size", 12)}" font-weight="{font_weight}" fill="{props.get("color", "black")}">{props["text"]}</text>'

        return None

    def generate_pdf_blueprint(self, layout: BlueprintLayout, output_path: str) -> str:
        """Generate professional PDF blueprint from layout"""

        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        styles = getSampleStyleSheet()

        # Create custom styles
        title_style = ParagraphStyle(
            'BlueprintTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # Center
        )

        section_style = ParagraphStyle(
            'Section',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=15
        )

        normal_style = styles['Normal']

        # Build PDF content
        content = []

        # Title
        content.append(Paragraph(layout.title, title_style))
        content.append(Spacer(1, 12))

        # Project info table
        project_data = [
            ['Project ID:', layout.id],
            ['Scale:', layout.scale],
            ['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Views:', ', '.join(layout.views)]
        ]

        project_table = Table(project_data, colWidths=[2*inch, 4*inch])
        project_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        content.append(project_table)
        content.append(Spacer(1, 20))

        # Components section
        content.append(Paragraph("Design Components", section_style))

        # Extract component data from layout elements
        components_data = []
        component_elements = [elem for elem in layout.elements if elem.label and elem.label.startswith('component_')]

        for elem in component_elements:
            component_name = elem.label.replace('component_', '').upper()
            components_data.append([component_name, ''])

        if components_data:
            comp_table = Table([['Component', 'Description']] + components_data, colWidths=[2*inch, 4*inch])
            comp_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ]))
            content.append(comp_table)
        content.append(Spacer(1, 20))

        # Technical specifications
        content.append(Paragraph("Technical Specifications", section_style))

        # Extract dimensions from layout
        specs_data = [
            ['Overall Width:', f"{layout.dimensions['width']} units"],
            ['Overall Height:', f"{layout.dimensions['height']} units"],
            ['Drawing Scale:', layout.scale],
            ['Total Components:', str(len(component_elements))]
        ]

        specs_table = Table(specs_data, colWidths=[2*inch, 4*inch])
        specs_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        content.append(specs_table)
        content.append(Spacer(1, 20))

        # Notes section
        content.append(Paragraph("Engineering Notes", section_style))
        notes = [
            "• All dimensions are in millimeters unless otherwise specified",
            "• Material specifications to be determined based on application requirements",
            "• Interface connections require standard fasteners and connectors",
            "• Design subject to final engineering review and testing",
            "• Manufacturing tolerances: ±0.1mm for critical dimensions"
        ]

        for note in notes:
            content.append(Paragraph(note, normal_style))
            content.append(Spacer(1, 6))

        # Build PDF
        doc.build(content)
        return output_path

    async def generate_technical_specs(self, design_blueprint: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive technical specifications"""

        components = design_blueprint["components"]
        design_params = design_blueprint["design_parameters"]

        specs = {
            "general": {
                "project_id": design_blueprint["id"],
                "design_category": design_blueprint["intent"]["category"],
                "complexity_level": design_blueprint["intent"]["complexity"],
                "total_components": len(components)
            },
            "components": [],
            "materials": [],
            "dimensions": design_params.get("overall_dimensions", {}),
            "performance": {},
            "compliance": []
        }

        # Component specifications
        for comp in components:
            comp_spec = {
                "name": comp["name"],
                "function": comp["function"],
                "dimensions": comp.get("dimensions", {}),
                "materials": comp.get("materials", []),
                "interfaces": comp.get("interfaces", []),
                "requirements": comp.get("requirements", [])
            }
            specs["components"].append(comp_spec)

            # Collect unique materials
            for material in comp.get("materials", []):
                if material not in specs["materials"]:
                    specs["materials"].append(material)

        # Performance requirements
        specs["performance"] = {
            "operational_temperature": "-10°C to 50°C",
            "power_requirements": "To be determined",
            "environmental_rating": "IP54 (dust/water resistant)",
            "expected_lifecycle": "5 years minimum"
        }

        # Compliance standards
        specs["compliance"] = [
            "ISO 9001: Quality Management",
            "CE Marking: European safety standards",
            "RoHS: Restriction of hazardous substances",
            "REACH: Chemical safety requirements"
        ]

        return specs

    async def generate_cad_script(self, design_blueprint: Dict[str, Any], format: str = "openscad") -> CADScript:
        """Generate CAD script for 3D modeling"""

        design_id = design_blueprint["id"]
        components = design_blueprint["components"]
        design_params = design_blueprint["design_parameters"]

        if format == "openscad":
            return self._generate_openscad_script(design_id, components, design_params)
        elif format == "freecad":
            return self._generate_freecad_script(design_id, components, design_params)
        else:
            raise ValueError(f"Unsupported CAD format: {format}")

    def _generate_openscad_script(self, design_id: str, components: List[Dict[str, Any]], design_params: Dict[str, Any]) -> CADScript:
        """Generate OpenSCAD script"""

        script_lines = [
            f"// {design_id} - Generated CAD Script",
            f"// Generated: {datetime.now().isoformat()}",
            "",
            "// Design Parameters"
        ]

        # Add design parameters
        for key, value in design_params.get("overall_dimensions", {}).items():
            script_lines.append(f"{key.upper()} = {value};")

        script_lines.extend([
            "",
            "// Component Modules"
        ])

        # Generate modules for each component
        for component in components:
            dims = component.get("dimensions", {})
            length = dims.get("length", 10)
            width = dims.get("width", 10)
            height = dims.get("height", 10)

            module_name = component["name"].replace(" ", "_").lower()

            script_lines.extend([
                f"module {module_name}() {{",
                f"    cube([{length}, {width}, {height}]);",
                f"}}",
                ""
            ])

        # Assembly
        script_lines.extend([
            "// Assembly",
            "module assembly() {"
        ])

        y_offset = 0
        for component in components:
            module_name = component["name"].replace(" ", "_").lower()
            dims = component.get("dimensions", {})
            height = dims.get("height", 10)

            script_lines.append(f"    translate([0, {y_offset}, 0]) {module_name}();")
            y_offset += height + 5

        script_lines.extend([
            "}",
            "",
            "// Render assembly",
            "assembly();"
        ])

        content = "\n".join(script_lines)

        return CADScript(
            format="openscad",
            content=content,
            parameters={
                "file_extension": ".scad",
                "software": "OpenSCAD",
                "render_command": "openscad -o output.stl input.scad"
            }
        )

    def _generate_freecad_script(self, design_id: str, components: List[Dict[str, Any]], design_params: Dict[str, Any]) -> CADScript:
        """Generate FreeCAD Python script"""

        script_lines = [
            f"# {design_id} - FreeCAD CAD Script",
            f"# Generated: {datetime.now().isoformat()}",
            "",
            "import FreeCAD",
            "import Part",
            "",
            f"doc = FreeCAD.newDocument('{design_id}')",
            ""
        ]

        # Create components
        for i, component in enumerate(components):
            dims = component.get("dimensions", {})
            length = dims.get("length", 10)
            width = dims.get("width", 10)
            height = dims.get("height", 10)

            script_lines.extend([
                f"# Create {component['name']}",
                f"comp_{i} = doc.addObject('Part::Box', '{component['name']}')",
                f"comp_{i}.Length = {length}",
                f"comp_{i}.Width = {width}",
                f"comp_{i}.Height = {height}",
                f"comp_{i}.Placement.Base = FreeCAD.Vector(0, {i * 15}, 0)",
                ""
            ])

        script_lines.extend([
            "doc.recompute()",
            "",
            f"FreeCAD.Console.PrintMessage('CAD model generated: {design_id}\\n')",
            "",
            "# Export to STL",
            f"import Mesh",
            f"mesh = Mesh.Mesh()",
            f"for obj in doc.Objects:",
            f"    if obj.TypeId == 'Part::Feature':",
            f"        mesh.addMesh(obj.Shape.tessellate(0.1))",
            f"mesh.write('output.stl')",
            "",
            "FreeCAD.Console.PrintMessage('Model exported to output.stl\\n')"
        ])

        content = "\n".join(script_lines)

        return CADScript(
            format="freecad",
            content=content,
            parameters={
                "file_extension": ".py",
                "software": "FreeCAD",
                "run_command": "freecad -c script.py"
            }
        )

    async def generate_technical_specs(self, design_blueprint: Dict[str, Any]) -> Dict[str, Any]:
        """Generate technical specifications table"""

        components = design_blueprint["components"]
        system_reqs = design_blueprint["system_requirements"]
        design_params = design_blueprint["design_parameters"]

        specs = {
            "title": f"Technical Specifications - {design_blueprint['id']}",
            "sections": [
                {
                    "title": "System Overview",
                    "items": [
                        {"parameter": "Category", "value": design_blueprint["intent"]["category"]},
                        {"parameter": "Complexity", "value": design_blueprint["intent"]["complexity"]},
                        {"parameter": "Scale", "value": design_blueprint["intent"]["scale"]},
                        {"parameter": "Total Components", "value": len(components)}
                    ]
                },
                {
                    "title": "Physical Specifications",
                    "items": [
                        {"parameter": "Total Weight", "value": f"{system_reqs['total_weight_kg']:.1f} kg"},
                        {"parameter": "Power Consumption", "value": f"{system_reqs['total_power_watts']:.0f} W"},
                        {"parameter": "Dimensions (L×W×H)", "value": f"{design_params['overall_dimensions']['length']:.1f} × {design_params['overall_dimensions']['width']:.1f} × {design_params['overall_dimensions']['height']:.1f} units"}
                    ]
                },
                {
                    "title": "Component Details",
                    "items": []
                }
            ]
        }

        # Add component details
        for component in components:
            specs["sections"][2]["items"].append({
                "parameter": component["name"],
                "value": f"{component['function']} ({component['complexity']})"
            })

        return specs

    def export_blueprint(self, layout: BlueprintLayout, output_dir: str = "output") -> Dict[str, str]:
        """Export blueprint in multiple formats"""

        Path(output_dir).mkdir(exist_ok=True)

        files = {}

        # SVG export
        svg_content = self.generate_svg_blueprint(layout)
        svg_path = f"{output_dir}/{layout.id}.svg"
        with open(svg_path, 'w') as f:
            f.write(svg_content)
        files["svg"] = svg_path

        # JSON export for layout data
        json_path = f"{output_dir}/{layout.id}.json"
        with open(json_path, 'w') as f:
            json.dump({
                "id": layout.id,
                "title": layout.title,
                "dimensions": layout.dimensions,
                "scale": layout.scale,
                "views": layout.views,
                "elements": [{"id": e.id, "type": e.element_type, "x": e.x, "y": e.y, "properties": e.properties, "label": e.label} for e in layout.elements]
            }, f, indent=2)
        files["json"] = json_path

        return files