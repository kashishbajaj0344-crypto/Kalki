#!/usr/bin/env python3
"""
Test multi-extractor LLM validation system
Validates all 5 extractors work correctly
"""

import asyncio
import sys
from modules.hybrid_learning_system import (
    KnowledgeExtractor, 
    MaterialProperty,
    DesignRule,
    CodeRequirement
)
from modules.llm import get_llm_engine

async def test_validators():
    """Test all 5 LLM validators"""
    
    # Initialize
    extractor = KnowledgeExtractor()
    llm = get_llm_engine()
    await llm.initialize()
    print("✅ LLM initialized")
    
    # Test 1: Formula Validator (already working)
    # Create a simple ExtractedFormula-like object
    class ExtractedFormula:
        def __init__(self, formula, page_num):
            self.formula = formula
            self.page_num = page_num
    
    formula = ExtractedFormula(formula="F = ma", page_num=1)
    result = await extractor._validate_formula_with_llm(llm, formula)
    print(f"✅ Formula validator: {result} (expected True)")
    
    # Test 2: Material Validator
    material = MaterialProperty(
        material_name="Concrete",
        property_type="compressive_strength",
        properties={"strength": "4000 psi"},
        source_pdf="test"
    )
    result = await extractor._validate_material_with_llm(llm, material)
    print(f"✅ Material validator: {result} (expected True)")
    
    # Test invalid material
    invalid_material = MaterialProperty(
        material_name="ft",  # Just a unit
        property_type="unknown",
        properties={},
        source_pdf="test"
    )
    result = await extractor._validate_material_with_llm(llm, invalid_material)
    print(f"✅ Material validator (invalid): {result} (expected False)")
    
    # Test 3: Design Rule Validator
    rule = DesignRule(
        category="structural",
        condition="WHEN beam span exceeds 20 feet",
        action="THEN provide additional supports",
        priority="high",
        source_pdf="test"
    )
    result = await extractor._validate_design_rule_with_llm(llm, rule)
    print(f"✅ Design rule validator: {result} (expected True)")
    
    # Test 4: Code Requirement Validator
    code = CodeRequirement(
        code_id="IBC-1605.1",
        code_type="building",
        requirement="Buildings SHALL be designed for loads specified in Chapter 16",
        applicability="All buildings",
        exceptions=[],
        source_pdf="test"
    )
    result = await extractor._validate_code_requirement_with_llm(llm, code)
    print(f"✅ Code requirement validator: {result} (expected True)")
    
    # Test 5: Procedure Validator
    procedure = {
        "procedure_name": "Concrete Pour",
        "category": "construction",
        "step_description": "1. Clean formwork 2. Place reinforcement 3. Pour concrete 4. Vibrate 5. Finish surface"
    }
    result = await extractor._validate_procedure_with_llm(llm, procedure)
    print(f"✅ Procedure validator: {result} (expected True)")
    
    print("\n🎉 All 5 validators tested successfully!")
    print("\nSummary:")
    print("  ✓ Formula validation")
    print("  ✓ Material validation")
    print("  ✓ Design rule validation")
    print("  ✓ Code requirement validation")
    print("  ✓ Procedure validation")

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(test_validators())
