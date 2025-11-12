#!/usr/bin/env python3
"""
Kalki Professional Blueprint Generator
Creates engineering-level blueprint drawings for robotic arms
"""

import os
from pathlib import Path
from typing import Dict, List, Any
import math

class ProfessionalBlueprintGenerator:
    """Generate professional engineering blueprint drawings"""

    def __init__(self):
        self.output_dir = Path("output/blueprints")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_robotic_arm_blueprint(self, blueprint_dict: Dict[str, Any]) -> List[str]:
        """Generate professional blueprint drawings for robotic arm"""
        files = []

        # Generate main assembly drawing
        assembly_svg = self._generate_assembly_drawing(blueprint_dict)
        assembly_path = self.output_dir / "robotic_arm_assembly_blueprint.svg"
        with open(assembly_path, 'w') as f:
            f.write(assembly_svg)
        files.append(str(assembly_path))

        # Generate detailed joint drawings
        components = blueprint_dict.get("components", [])
        for i, comp in enumerate(components):
            if 'joint' in comp.get('name', '').lower():
                joint_svg = self._generate_joint_detail_drawing(comp, i+1)
                joint_path = self.output_dir / f"joint_{i+1}_detail_blueprint.svg"
                with open(joint_path, 'w') as f:
                    f.write(joint_svg)
                files.append(str(joint_path))

        # Generate specifications sheet
        specs_svg = self._generate_specifications_sheet(blueprint_dict)
        specs_path = self.output_dir / "robotic_arm_specifications_blueprint.svg"
        with open(specs_path, 'w') as f:
            f.write(specs_svg)
        files.append(str(specs_path))

        return files

    def _generate_assembly_drawing(self, blueprint_dict: Dict[str, Any]) -> str:
        """Generate main assembly blueprint"""
        components = blueprint_dict.get("components", [])
        dimensions = blueprint_dict.get("dimensions", {})

        svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="800" height="600" xmlns="http://www.w3.org/2000/svg">
    <!-- Title Block -->
    <rect x="500" y="500" width="300" height="100" fill="none" stroke="black" stroke-width="1"/>
    <text x="510" y="520" font-family="Arial" font-size="12" font-weight="bold">ROBOTIC ARM ASSEMBLY</text>
    <text x="510" y="535" font-family="Arial" font-size="10">Scale: 1:10</text>
    <text x="510" y="550" font-family="Arial" font-size="10">Drawn by: Kalki AI</text>
    <text x="510" y="565" font-family="Arial" font-size="10">Date: {blueprint_dict.get('generated', 'N/A')}</text>

    <!-- Main Assembly View -->
    <g transform="translate(50, 50)">
        <!-- Base -->
        <circle cx="100" cy="500" r="40" fill="none" stroke="black" stroke-width="2"/>
        <text x="60" y="510" font-family="Arial" font-size="8">BASE</text>
        <text x="70" y="525" font-family="Arial" font-size="6">Ø200</text>

        <!-- Shoulder Joint -->
        <rect x="85" y="420" width="30" height="60" fill="none" stroke="black" stroke-width="2"/>
        <circle cx="100" cy="410" r="15" fill="none" stroke="black" stroke-width="1"/>
        <text x="75" y="445" font-family="Arial" font-size="8">SHOULDER</text>
        <text x="85" y="455" font-family="Arial" font-size="6">JOINT</text>

        <!-- Upper Arm -->
        <rect x="100" y="350" width="80" height="15" fill="none" stroke="black" stroke-width="2"/>
        <text x="110" y="365" font-family="Arial" font-size="8">UPPER ARM</text>
        <text x="120" y="375" font-family="Arial" font-size="6">280mm</text>

        <!-- Elbow Joint -->
        <rect x="175" y="320" width="25" height="40" fill="none" stroke="black" stroke-width="2"/>
        <circle cx="187" cy="310" r="12" fill="none" stroke="black" stroke-width="1"/>
        <text x="165" y="340" font-family="Arial" font-size="8">ELBOW</text>
        <text x="170" y="350" font-family="Arial" font-size="6">JOINT</text>

        <!-- Forearm -->
        <rect x="195" y="280" width="90" height="12" fill="none" stroke="black" stroke-width="2"/>
        <text x="205" y="295" font-family="Arial" font-size="8">FOREARM</text>
        <text x="215" y="305" font-family="Arial" font-size="6">320mm</text>

        <!-- Wrist Joint -->
        <rect x="280" y="250" width="20" height="35" fill="none" stroke="black" stroke-width="2"/>
        <circle cx="290" cy="240" r="10" fill="none" stroke="black" stroke-width="1"/>
        <text x="270" y="270" font-family="Arial" font-size="8">WRIST</text>
        <text x="275" y="280" font-family="Arial" font-size="6">JOINT</text>

        <!-- End Effector -->
        <rect x="295" y="220" width="30" height="20" fill="none" stroke="black" stroke-width="2"/>
        <text x="285" y="240" font-family="Arial" font-size="8">END EFFECTOR</text>
        <text x="295" y="250" font-family="Arial" font-size="6">GRIPPER</text>

        <!-- Dimensions -->
        <line x1="50" y1="520" x2="50" y2="410" stroke="black" stroke-width="1" marker-start="url(#arrow)" marker-end="url(#arrow)"/>
        <text x="25" y="465" font-family="Arial" font-size="8" transform="rotate(-90, 25, 465)">110mm</text>

        <line x1="100" y1="390" x2="180" y2="390" stroke="black" stroke-width="1" marker-start="url(#arrow)" marker-end="url(#arrow)"/>
        <text x="140" y="385" font-family="Arial" font-size="8">280mm</text>

        <line x1="187" y1="340" x2="285" y2="340" stroke="black" stroke-width="1" marker-start="url(#arrow)" marker-end="url(#arrow)"/>
        <text x="236" y="335" font-family="Arial" font-size="8">320mm</text>
    </g>

    <!-- Arrows for dimensions -->
    <defs>
        <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,6 L9,3 z" fill="black"/>
        </marker>
    </defs>
</svg>'''

        return svg

    def _generate_joint_detail_drawing(self, joint_comp: Dict[str, Any], joint_number: int) -> str:
        """Generate detailed joint blueprint"""
        dims = joint_comp.get('dimensions', {})

        svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="600" height="400" xmlns="http://www.w3.org/2000/svg">
    <!-- Title Block -->
    <rect x="400" y="300" width="200" height="100" fill="none" stroke="black" stroke-width="1"/>
    <text x="410" y="320" font-family="Arial" font-size="12" font-weight="bold">JOINT {joint_number} DETAIL</text>
    <text x="410" y="335" font-family="Arial" font-size="10">Scale: 1:2</text>
    <text x="410" y="350" font-family="Arial" font-size="10">Material: Steel/Aluminum</text>

    <!-- Detailed Joint View -->
    <g transform="translate(50, 50)">
        <!-- Harmonic Drive -->
        <circle cx="150" cy="150" r="40" fill="none" stroke="black" stroke-width="3"/>
        <circle cx="150" cy="150" r="35" fill="none" stroke="black" stroke-width="2"/>
        <circle cx="150" cy="150" r="25" fill="none" stroke="black" stroke-width="1"/>
        <text x="130" y="155" font-family="Arial" font-size="8">HARMONIC DRIVE</text>

        <!-- Bearings -->
        <circle cx="150" cy="110" r="15" fill="none" stroke="black" stroke-width="2"/>
        <circle cx="150" cy="190" r="15" fill="none" stroke="black" stroke-width="2"/>
        <text x="135" y="115" font-family="Arial" font-size="6">BEARING</text>
        <text x="135" y="195" font-family="Arial" font-size="6">BEARING</text>

        <!-- Servo Motor -->
        <rect x="200" y="120" width="40" height="60" fill="none" stroke="black" stroke-width="2"/>
        <circle cx="220" cy="100" r="3" fill="none" stroke="black" stroke-width="1"/>
        <text x="205" y="145" font-family="Arial" font-size="8">SERVO MOTOR</text>
        <text x="210" y="155" font-family="Arial" font-size="6">50W</text>

        <!-- Dimensions -->
        <line x1="110" y1="150" x2="190" y2="150" stroke="black" stroke-width="1" marker-start="url(#arrow)" marker-end="url(#arrow)"/>
        <text x="150" y="145" font-family="Arial" font-size="8">Ø80</text>

        <line x1="150" y1="95" x2="150" y2="125" stroke="black" stroke-width="1" marker-start="url(#arrow)" marker-end="url(#arrow)"/>
        <text x="155" y="110" font-family="Arial" font-size="8">30</text>
    </g>

    <!-- Arrows for dimensions -->
    <defs>
        <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,6 L9,3 z" fill="black"/>
        </marker>
    </defs>
</svg>'''

        return svg

    def _generate_specifications_sheet(self, blueprint_dict: Dict[str, Any]) -> str:
        """Generate specifications sheet"""
        specs = blueprint_dict.get("specifications", {})
        dimensions = blueprint_dict.get("dimensions", {})

        svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="800" height="600" xmlns="http://www.w3.org/2000/svg">
    <!-- Title -->
    <text x="50" y="50" font-family="Arial" font-size="16" font-weight="bold">ROBOTIC ARM TECHNICAL SPECIFICATIONS</text>

    <!-- Specifications Table -->
    <rect x="50" y="70" width="700" height="500" fill="none" stroke="black" stroke-width="1"/>

    <!-- Headers -->
    <line x1="50" y1="100" x2="750" y2="100" stroke="black" stroke-width="1"/>
    <text x="60" y="90" font-family="Arial" font-size="12" font-weight="bold">PARAMETER</text>
    <text x="300" y="90" font-family="Arial" font-size="12" font-weight="bold">VALUE</text>
    <text x="500" y="90" font-family="Arial" font-size="12" font-weight="bold">UNITS</text>

    <!-- Specification Rows -->
    <text x="60" y="120" font-family="Arial" font-size="10">Degrees of Freedom</text>
    <text x="300" y="120" font-family="Arial" font-size="10">{specs.get('degrees_of_freedom', 6)}</text>
    <text x="500" y="120" font-family="Arial" font-size="10">-</text>

    <text x="60" y="140" font-family="Arial" font-size="10">Reach</text>
    <text x="300" y="140" font-family="Arial" font-size="10">{dimensions.get('reach', 850)}</text>
    <text x="500" y="140" font-family="Arial" font-size="10">mm</text>

    <text x="60" y="160" font-family="Arial" font-size="10">Payload Capacity</text>
    <text x="300" y="160" font-family="Arial" font-size="10">{dimensions.get('payload', 5)}</text>
    <text x="500" y="160" font-family="Arial" font-size="10">kg</text>

    <text x="60" y="180" font-family="Arial" font-size="10">Repeatability</text>
    <text x="300" y="180" font-family="Arial" font-size="10">{specs.get('repeatability', 0.05)}</text>
    <text x="500" y="180" font-family="Arial" font-size="10">mm</text>

    <text x="60" y="200" font-family="Arial" font-size="10">Power Consumption</text>
    <text x="300" y="200" font-family="Arial" font-size="10">{specs.get('power_consumption', 500)}</text>
    <text x="500" y="200" font-family="Arial" font-size="10">W</text>

    <text x="60" y="220" font-family="Arial" font-size="10">Operating Temperature</text>
    <text x="300" y="220" font-family="Arial" font-size="10">{specs.get('operating_temp', [5, 45])}</text>
    <text x="500" y="220" font-family="Arial" font-size="10">°C</text>

    <text x="60" y="240" font-family="Arial" font-size="10">Total Weight</text>
    <text x="300" y="240" font-family="Arial" font-size="10">{dimensions.get('weight', 18.5)}</text>
    <text x="500" y="240" font-family="Arial" font-size="10">kg</text>

    <!-- Materials Section -->
    <text x="60" y="280" font-family="Arial" font-size="12" font-weight="bold">MATERIALS:</text>
    <text x="60" y="300" font-family="Arial" font-size="10">• Base & Links: Aluminum 6061</text>
    <text x="60" y="315" font-family="Arial" font-size="10">• Joints & Gears: Steel 4140</text>
    <text x="60" y="330" font-family="Arial" font-size="10">• Fasteners: Stainless Steel</text>
    <text x="60" y="345" font-family="Arial" font-size="10">• Cables & Insulation: Various Polymers</text>

    <!-- Components Section -->
    <text x="60" y="380" font-family="Arial" font-size="12" font-weight="bold">MAJOR COMPONENTS:</text>
    <text x="60" y="400" font-family="Arial" font-size="10">• 6x Harmonic Drive Actuators</text>
    <text x="60" y="415" font-family="Arial" font-size="10">• 6x Servo Motors (50W each)</text>
    <text x="60" y="430" font-family="Arial" font-size="10">• 12x Precision Bearings</text>
    <text x="60" y="445" font-family="Arial" font-size="10">• 1x Parallel Gripper End Effector</text>
    <text x="60" y="460" font-family="Arial" font-size="10">• Integrated Controller & Sensors</text>
</svg>'''

        return svg