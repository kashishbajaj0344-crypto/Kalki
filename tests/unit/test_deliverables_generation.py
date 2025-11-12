"""
Test Construction Deliverables Generation
"""

import asyncio
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from modules.domains.construction_domain.construction_domain import (
    ConstructionDomain,
    ConstructionProjectStateMachine
)


async def test_deliverables():
    """Test all deliverables generation"""
    
    print("=" * 80)
    print("Testing Construction Deliverables Generation")
    print("=" * 80)
    
    # Create domain and project
    domain = ConstructionDomain()
    
    project = ConstructionProjectStateMachine(
        "test-deliv-123",
        "Build 3-story home in Vancouver, BC"
    )
    
    # Set project details
    project.location = "Vancouver, BC"
    project.building_type = "residential_multi_story"
    project.size_sqft = 2500
    project.stories = 3
    project.budget["estimated_total"] = 800000
    
    print(f"\nProject: {project.description}")
    print(f"Size: {project.size_sqft} sq ft, {project.stories} stories")
    print(f"Location: {project.location}")
    
    # Test each deliverable type
    deliverable_types = [
        "construction_drawings",
        "bill_of_materials",
        "construction_schedule",
        "inspection_checklists",
        "structural_calculations",
        "cost_estimate"
    ]
    
    output_dir = Path("output/test_deliverables")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'=' * 80}")
    print("Generating Deliverables")
    print("=" * 80)
    
    generated = await domain.generate_deliverables(
        project,
        deliverable_types,
        output_dir
    )
    
    print(f"\n✅ Generated {len(generated)} deliverables:")
    for deliv_type, path in generated.items():
        print(f"  • {deliv_type}: {path}")
    
    # Display sample content from each
    print(f"\n{'=' * 80}")
    print("Deliverable Summaries")
    print("=" * 80)
    
    for deliv_type, path in generated.items():
        print(f"\n--- {deliv_type.upper().replace('_', ' ')} ---")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        if deliv_type == "construction_drawings":
            print(f"  Total Sheets: {data['total_sheets']}")
            print(f"  Drawing Sets: {len(data['drawings'])}")
            print(f"  Project: {data['project_info']['project_name']}")
            
        elif deliv_type == "bill_of_materials":
            print(f"  Total Items: {data['total_items']}")
            print(f"  Categories: {len(data['categories'])}")
            print(f"  Materials Cost: ${data['cost_summary']['materials_subtotal']:,.2f}")
            print(f"  Grand Total: ${data['cost_summary']['grand_total']:,.2f} {data['currency']}")
            print(f"  Cost per sq ft: ${data['cost_summary']['cost_per_sqft']:.2f}")
            
            print(f"\n  Top 5 Most Expensive Items:")
            sorted_items = sorted(data['items'], key=lambda x: x['total_cost'], reverse=True)[:5]
            for i, item in enumerate(sorted_items, 1):
                print(f"    {i}. {item['item']}: ${item['total_cost']:,.2f}")
            
        elif deliv_type == "construction_schedule":
            print(f"  Total Phases: {len(data['phases'])}")
            print(f"  Project Duration: {data['project_duration_days']} days ({data['project_duration_months']} months)")
            print(f"  Start Date: {data['start_date']}")
            print(f"  Completion Date: {data['completion_date']}")
            print(f"  Total Inspections: {data['total_inspections']}")
            
            print(f"\n  Phase Timeline:")
            for phase in data['phases'][:5]:  # Show first 5
                print(f"    • {phase['phase']}: {phase['duration_days']} days ({phase['start_date']} to {phase['end_date']})")
            
        elif deliv_type == "inspection_checklists":
            print(f"  Total Checklists: {data['total_checklists']}")
            print(f"  Total Items: {data['total_items']}")
            print(f"  Critical Items: {data['critical_items']}")
            print(f"  Jurisdiction: {data['jurisdiction']}")
            
            print(f"\n  Checklists:")
            for name, checklist in list(data['checklists'].items())[:3]:  # Show first 3
                critical = sum(1 for item in checklist['items'] if item['critical'])
                print(f"    • {name}: {len(checklist['items'])} items ({critical} critical)")
            
        elif deliv_type == "structural_calculations":
            print(f"  Project Size: {data['project_info']['size_sqft']} sq ft, {data['project_info']['stories']} stories")
            print(f"  Location: {data['project_info']['location']}")
            print(f"\n  Load Parameters:")
            for key, value in data['loads'].items():
                print(f"    • {key}: {value}")
            
            print(f"\n  Typical Spans:")
            print(f"    • Floor Joists: {data['spans']['floor_joists']['member_size']} @ {data['spans']['floor_joists']['span_ft']}ft")
            print(f"    • Roof Rafters: {data['spans']['roof_rafters']['member_size']} @ {data['spans']['roof_rafters']['span_ft']}ft")
            
        elif deliv_type == "cost_estimate":
            print(f"  Construction Costs: ${data['construction_costs']['grand_total']:,.2f}")
            print(f"  Additional Costs: ${data['cost_summary']['additional_costs_total']:,.2f}")
            print(f"  Grand Total: ${data['cost_summary']['grand_total']:,.2f} {data['currency']}")
            print(f"  Cost per sq ft: ${data['cost_summary']['cost_per_sqft']:.2f}")
            
            print(f"\n  Payment Schedule:")
            for milestone, amount in data['payment_schedule'].items():
                print(f"    • {milestone}: ${amount:,.2f}")
    
    print(f"\n{'=' * 80}")
    print("✅ All deliverables generated successfully!")
    print(f"Output directory: {output_dir.absolute()}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_deliverables())
