#!/usr/bin/env python3
"""
Test Professional Deliverables System
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from modules.professional_deliverables import ProfessionalDeliverablesGenerator

async def test_deliverables():
    """Test generating professional deliverables for a robotic arm"""
    
    print("🎯 Testing Professional Deliverables System")
    print("=" * 60)
    
    # Sample robotic arm design data
    design_data = {
        "project_id": "robot_test_001",
        "name": "6-DOF Industrial Robotic Arm",
        "type": "robotic",
        "description": "High-precision 6-axis robotic manipulator for industrial assembly tasks",
        "components": [
            {
                "name": "Base Platform",
                "type": "structural",
                "dimensions": {"length": 200, "width": 200, "height": 50},
                "materials": ["aluminum_6061"],
                "count": 1
            },
            {
                "name": "Shoulder Joint",
                "type": "actuator",
                "dimensions": {"diameter": 100, "length": 120, "height": 100},
                "materials": ["steel_4140"],
                "count": 1
            },
            {
                "name": "Upper Arm Link",
                "type": "structural",
                "dimensions": {"length": 300, "diameter": 60, "height": 40},
                "materials": ["aluminum_6061"],
                "count": 1
            },
            {
                "name": "Elbow Joint",
                "type": "actuator",
                "dimensions": {"diameter": 80, "length": 100, "height": 80},
                "materials": ["steel_4140"],
                "count": 1
            },
            {
                "name": "Forearm Link",
                "type": "structural",
                "dimensions": {"length": 250, "diameter": 50, "height": 35},
                "materials": ["aluminum_6061"],
                "count": 1
            },
            {
                "name": "Wrist Joint",
                "type": "actuator",
                "dimensions": {"diameter": 60, "length": 80, "height": 60},
                "materials": ["steel_4140"],
                "count": 1
            },
            {
                "name": "End Effector",
                "type": "tool",
                "dimensions": {"length": 100, "width": 80, "height": 60},
                "materials": ["aluminum_6061"],
                "count": 1
            }
        ],
        "dimensions": {
            "reach_m": 0.85,
            "payload_kg": 5.0,
            "precision_mm": 0.05,
            "total_mass_kg": 18.5,
            "degrees_of_freedom": 6
        },
        "materials": ["aluminum_6061", "steel_4140", "stainless_steel_304"],
        "specifications": {
            "degrees_of_freedom": 6,
            "workspace_radius_m": 0.85,
            "payload_capacity_kg": 5.0,
            "positional_accuracy_mm": 0.05,
            "repeatability_mm": 0.025,
            "max_speed_m_s": 1.5,
            "operating_voltage": "24V DC",
            "power_consumption_w": 500,
            "estimated_labor_hours": 90
        }
    }
    
    # Generate deliverables
    generator = ProfessionalDeliverablesGenerator()
    
    print("\n📦 Generating professional deliverables package...")
    deliverables = await generator.generate_complete_package(design_data)
    
    print("\n✅ DELIVERABLES GENERATED SUCCESSFULLY\n")
    
    # Display summary
    print("📋 DELIVERABLES SUMMARY")
    print("=" * 60)
    print(f"Project ID: {deliverables.project_id}")
    print(f"Project Name: {deliverables.project_name}")
    print(f"\n📊 Bill of Materials: {len(deliverables.bill_of_materials.items)} items")
    print(f"💰 Total Cost Estimate: ${deliverables.bill_of_materials.total_cost_estimate:,.2f}")
    print(f"⚖️  Total Weight: {deliverables.bill_of_materials.total_weight:.2f} kg")
    print(f"📅 Lead Time: {deliverables.bill_of_materials.lead_time_days} days")
    
    print(f"\n📝 Assembly Instructions: {len(deliverables.assembly_instructions)} steps")
    print(f"✓ Quality Control Checks: {len(deliverables.quality_control_checklist)} items")
    print(f"📜 Compliance Certifications: {len(deliverables.compliance_certifications)} standards")
    
    print(f"\n💵 Cost Analysis:")
    cost = deliverables.cost_analysis['cost_breakdown']
    print(f"   Materials: ${cost['materials']:,.2f}")
    print(f"   Labor: ${cost['labor']:,.2f}")
    print(f"   Overhead: ${cost['overhead']:,.2f}")
    print(f"   Contingency: ${cost['contingency']:,.2f}")
    print(f"   TOTAL: ${cost['total']:,.2f}")
    
    print(f"\n⏱️  Project Timeline:")
    timeline = deliverables.timeline_estimate
    print(f"   Total Duration: {timeline['total_duration_weeks']} weeks ({timeline['total_duration_days']} days)")
    print(f"   Estimated Completion: {timeline['estimated_completion']}")
    
    print(f"\n📁 Generated Files ({len(deliverables.generated_files)}):")
    for file_path in deliverables.generated_files:
        file_name = Path(file_path).name
        print(f"   ✓ {file_name}")
    
    print(f"\n📍 Location: output/deliverables/{deliverables.project_id}/")
    
    # Display executive summary excerpt
    print("\n" + "=" * 60)
    print("📄 EXECUTIVE SUMMARY (Excerpt)")
    print("=" * 60)
    summary_lines = deliverables.executive_summary.split('\n')[:20]
    print('\n'.join(summary_lines))
    print("... (see full document for complete details)")
    
    print("\n" + "=" * 60)
    print("✨ PROFESSIONAL DELIVERABLES PACKAGE COMPLETE")
    print("=" * 60)
    print("\n🎉 Ready for construction/manufacturing!")
    print(f"\n💡 Open the deliverables folder to review all documents:")
    print(f"   {Path('output/deliverables') / deliverables.project_id}")

if __name__ == "__main__":
    asyncio.run(test_deliverables())
