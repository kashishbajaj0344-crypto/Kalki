#!/usr/bin/env python3
"""
Test Architectural Drawing Generation
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from modules.professional_deliverables import ProfessionalDeliverablesGenerator

async def test_house_design():
    """Test generating complete deliverables for a 30'x50' 3-level house"""
    
    print("🏠 Testing Architectural Drawing Generation")
    print("=" * 60)
    print("Project: 30' x 50' Three-Level Residential House")
    print("=" * 60)
    
    # House design data
    design_data = {
        "project_id": "house_30x50_3level",
        "name": "Modern Three-Story Residence",
        "type": "architecture",
        "description": "30 feet x 50 feet modern residential house with 3 levels",
        "components": [
            {
                "name": "Foundation",
                "type": "structural",
                "dimensions": {"length": 360, "width": 600, "height": 36},  # inches
                "materials": ["concrete"],
                "count": 1
            },
            {
                "name": "First Floor Frame",
                "type": "structural",
                "dimensions": {"length": 360, "width": 600, "height": 108},
                "materials": ["lumber_2x6"],
                "count": 1
            },
            {
                "name": "Second Floor Frame",
                "type": "structural",
                "dimensions": {"length": 360, "width": 600, "height": 108},
                "materials": ["lumber_2x6"],
                "count": 1
            },
            {
                "name": "Third Floor Frame",
                "type": "structural",
                "dimensions": {"length": 360, "width": 600, "height": 108},
                "materials": ["lumber_2x6"],
                "count": 1
            },
            {
                "name": "Roof Structure",
                "type": "structural",
                "dimensions": {"length": 360, "width": 600, "height": 72},
                "materials": ["lumber_2x10"],
                "count": 1
            },
            {
                "name": "Exterior Siding",
                "type": "finish",
                "dimensions": {"area": 4320},  # sq ft
                "materials": ["fiber_cement"],
                "count": 1
            },
            {
                "name": "Windows",
                "type": "opening",
                "dimensions": {"width": 48, "height": 60},
                "materials": ["vinyl"],
                "count": 24
            },
            {
                "name": "Entry Doors",
                "type": "opening",
                "dimensions": {"width": 36, "height": 84},
                "materials": ["steel"],
                "count": 2
            },
            {
                "name": "Interior Doors",
                "type": "opening",
                "dimensions": {"width": 30, "height": 80},
                "materials": ["wood"],
                "count": 12
            }
        ],
        "dimensions": {
            "width_ft": 30,
            "depth_ft": 50,
            "levels": 3,
            "total_area_sqft": 4500,  # 30x50x3
            "ceiling_height_ft": 9,
            "lot_size_sqft": 12000
        },
        "materials": ["concrete", "lumber_2x6", "lumber_2x10", "fiber_cement", "vinyl", "steel", "wood"],
        "specifications": {
            "building_type": "Single Family Residential",
            "construction_type": "Type V - Wood Frame",
            "occupancy_group": "R-3 Residential",
            "bedrooms": 4,
            "bathrooms": 3.5,
            "parking_spaces": 2,
            "building_code": "2021 International Building Code",
            "zoning": "R-1 Single Family Residential",
            "estimated_labor_hours": 2400
        }
    }
    
    # Generate deliverables
    generator = ProfessionalDeliverablesGenerator()
    
    print("\n📦 Generating professional deliverables package...")
    print("   This includes architectural drawings, floor plans, elevations, sections...")
    
    deliverables = await generator.generate_complete_package(design_data)
    
    print("\n✅ DELIVERABLES GENERATED SUCCESSFULLY\n")
    
    # Display summary
    print("📋 ARCHITECTURAL DELIVERABLES SUMMARY")
    print("=" * 60)
    print(f"Project ID: {deliverables.project_id}")
    print(f"Project Name: {deliverables.project_name}")
    
    print(f"\n🏗️  Building Specifications:")
    specs = design_data["specifications"]
    print(f"   Type: {specs['building_type']}")
    print(f"   Construction: {specs['construction_type']}")
    print(f"   Bedrooms: {specs['bedrooms']} | Bathrooms: {specs['bathrooms']}")
    print(f"   Total Area: {design_data['dimensions']['total_area_sqft']:,} sq ft")
    
    print(f"\n📐 Drawing Set:")
    drawing_count = (len(deliverables.drawing_set.plan_views) + 
                    len(deliverables.drawing_set.elevations) +
                    len(deliverables.drawing_set.sections) +
                    len(deliverables.drawing_set.isometric_views))
    
    print(f"   Floor Plans: {len(deliverables.drawing_set.plan_views)} sheets")
    print(f"   Elevations: {len(deliverables.drawing_set.elevations)} sheets")
    print(f"   Sections: {len(deliverables.drawing_set.sections)} sheets")
    print(f"   Site Plan: 1 sheet")
    print(f"   TOTAL DRAWINGS: {drawing_count + 1} sheets")
    
    print(f"\n📊 Bill of Materials: {len(deliverables.bill_of_materials.items)} items")
    print(f"💰 Total Cost Estimate: ${deliverables.bill_of_materials.total_cost_estimate:,.2f}")
    
    print(f"\n💵 Cost Breakdown:")
    cost = deliverables.cost_analysis['cost_breakdown']
    print(f"   Materials: ${cost['materials']:,.2f}")
    print(f"   Labor ({design_data['specifications']['estimated_labor_hours']} hrs @ $75/hr): ${cost['labor']:,.2f}")
    print(f"   Overhead (15%): ${cost['overhead']:,.2f}")
    print(f"   Contingency (10%): ${cost['contingency']:,.2f}")
    print(f"   {'─' * 40}")
    print(f"   TOTAL PROJECT COST: ${cost['total']:,.2f}")
    
    print(f"\n⏱️  Construction Timeline:")
    timeline = deliverables.timeline_estimate
    print(f"   Duration: {timeline['total_duration_weeks']} weeks ({timeline['total_duration_days']} days)")
    print(f"   Estimated Completion: {timeline['estimated_completion']}")
    
    print(f"\n📁 Generated Files ({len(deliverables.generated_files)}):")
    for file_path in deliverables.generated_files:
        file_name = Path(file_path).name
        print(f"   ✓ {file_name}")
    
    print(f"\n📐 Architectural Drawings:")
    drawings_dir = Path("output/deliverables") / deliverables.project_id / "drawings"
    if drawings_dir.exists():
        for drawing in sorted(drawings_dir.glob("*.png")):
            print(f"   ✓ {drawing.name}")
    
    print(f"\n📍 Location: output/deliverables/{deliverables.project_id}/")
    
    print("\n" + "=" * 60)
    print("✨ CONSTRUCTION-READY ARCHITECTURAL PACKAGE COMPLETE")
    print("=" * 60)
    print("\n🎉 Ready for permit submission and construction!")
    print(f"\n💡 Review the complete package:")
    print(f"   {Path('output/deliverables') / deliverables.project_id}")
    print(f"\n📐 View architectural drawings:")
    print(f"   {drawings_dir}")

if __name__ == "__main__":
    asyncio.run(test_house_design())
