#!/usr/bin/env python3
"""
KALKI v2.5 Extraction Test
Test new enhanced extraction methods on sample construction text
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from modules.hybrid_learning_system import KnowledgeExtractor

# Sample construction text with v2.5 extraction targets
SAMPLE_TEXT = """
BC BUILDING CODE PART 9 - SPAN TABLES

Floor Joists - SPF No. 2 Grade
2x6 @ 16" O.C. - 12' 3" span - 40 PSF live load
2x8 @ 16" O.C. - 15' 6" span - 40 PSF live load  
2x10 @ 16" O.C. - 19' 0" span - 40 PSF live load
2x12 @ 16" O.C. - 23' 0" span - 40 PSF live load

FOUNDATION INSTALLATION PROCEDURE

Step 1: Excavate to depth of undisturbed soil
Step 2: Install vapor barrier over compacted gravel
Step 3: Install rebar grid per structural drawings
Step 4: Pour concrete and vibrate to eliminate voids
Step 5: Cure concrete for minimum 7 days

INSPECTION REQUIREMENTS

Inspect foundation for cracks greater than 1/4 inch.
Inspect framing for proper nailing schedule compliance.
Footings shall be inspected before concrete placement.

MATERIAL COSTS (2024 BC Prices)

2x4 SPF studs: $4.50/ea
2x6 SPF studs: $7.25/ea
Framing labor: $45/hr
Concrete 3000 PSI: $140/cubic yard
Rebar #4: $12.50/20ft

STRUCTURAL LOADS

Residential floor live load: 40 PSF
Residential floor dead load: 10 PSF
Roof snow load: 35 PSF
Wind load: 25 PSF

CODE REQUIREMENTS

If building height > 35 feet, then sprinkler system required.
If occupancy > 60 people, then 2 exits required.
When seismic zone is high, then special detailing required.
"""

def test_v25_extraction():
    """Test KALKI v2.5 enhanced extraction"""
    
    print("🧪 KALKI v2.5 Extraction Test")
    print("=" * 60)
    
    # Initialize extractor
    extractor = KnowledgeExtractor(knowledge_db_path="data/knowledge_test/")
    
    # Run extraction
    print("\n📄 Processing sample construction text...")
    results = extractor.extract_from_pdf("test_sample.pdf", SAMPLE_TEXT)
    
    # Display results
    print(f"\n✅ Extraction Results:")
    print(f"   Formulas: {results['formulas']}")
    print(f"   Materials: {results['materials']}")
    print(f"   Design Rules: {results['rules']}")
    print(f"   Code Requirements: {results['codes']}")
    print(f"\n   ⭐ v2.5 Enhanced:")
    print(f"   Span Tables: {results['span_tables']}")
    print(f"   Procedures: {results['procedures']}")
    print(f"   Inspection Criteria: {results['inspection_criteria']}")
    print(f"   Cost Data: {results['cost_data']}")
    print(f"   Load Parameters: {results['load_parameters']}")
    print(f"   Decision Trees: {results['decision_trees']}")
    
    # Query and display some results
    print(f"\n📊 Sample Queries:")
    
    # Query span tables
    span_tables = extractor.query_span_tables(member_type="joist")
    if span_tables:
        print(f"\n   Span Tables Found: {len(span_tables)}")
        for st in span_tables[:2]:
            print(f"   - {st['member_size']} @ {st['spacing']}: "
                  f"{st['span_feet']}'{st['span_inches']}\" "
                  f"({st['load_value']} {st['load_unit']})")
    
    # Query procedures
    procedures = extractor.query_procedures(category="foundation")
    if procedures:
        print(f"\n   Foundation Procedures Found: {len(procedures)}")
        for proc in procedures[:3]:
            print(f"   {proc['step_number']}. {proc['step_description']}")
    
    # Query inspection criteria
    inspections = extractor.query_inspection_criteria(inspection_type="foundation_inspection")
    if inspections:
        print(f"\n   Foundation Inspections Found: {len(inspections)}")
        for insp in inspections[:2]:
            print(f"   - {insp['component']}: {insp['criteria_description']}")
    
    # Query cost data
    costs = extractor.query_cost_data(item_category="material")
    if costs:
        print(f"\n   Material Costs Found: {len(costs)}")
        for cost in costs[:3]:
            print(f"   - {cost['item_name']}: ${cost['unit_cost']}/{cost['unit']}")
    
    # Query load parameters
    loads = extractor.query_load_parameters(building_type="residential")
    if loads:
        print(f"\n   Residential Loads Found: {len(loads)}")
        for load in loads[:3]:
            print(f"   - {load['load_name']}: {load['load_value']} {load['load_unit']}")
    
    # Query decision trees
    decisions = extractor.query_decision_trees()
    if decisions:
        print(f"\n   Decision Trees Found: {len(decisions)}")
        for dec in decisions[:2]:
            print(f"   - IF {dec['condition']} {dec['condition_operator']} {dec['condition_value']}")
            print(f"     THEN {dec['then_action']}")
    
    # Get statistics
    stats = extractor.get_statistics()
    print(f"\n📈 Knowledge Base Statistics:")
    print(f"   Total Items: {sum(stats.values())}")
    for key, value in stats.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n✅ v2.5 extraction test complete!")
    print(f"\n💡 Next steps:")
    print(f"   1. Download PDFs from academia.edu (see download list)")
    print(f"   2. Run: kalki learn ingest <pdf_path>")
    print(f"   3. Verify accuracy: kalki learn stats-v25")

if __name__ == "__main__":
    test_v25_extraction()
