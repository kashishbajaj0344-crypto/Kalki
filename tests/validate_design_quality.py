#!/usr/bin/env python3
"""
Design Quality Validation Script
Tests the RAG pipeline to ensure generated designs meet professional engineering standards.
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from modules.generative_design_engine import GenerativeDesignEngine
from modules.design_brain import DesignBrain

class DesignQualityValidator:
    """Validates that generated designs meet professional engineering standards."""

    def __init__(self):
        self.engine = GenerativeDesignEngine()
        self.design_brain = DesignBrain()

    async def initialize(self):
        """Initialize the validator components"""
        self.engine = GenerativeDesignEngine()
        await self.engine.initialize()
        
        self.design_brain = DesignBrain()
        success = await self.design_brain.initialize()
        if not success:
            raise RuntimeError("Failed to initialize DesignBrain")
        return self

    def validate_design_structure(self, design_data):
        """Validate that the design has proper engineering structure."""
        required_sections = [
            'system_requirements',
            'component_specifications',
            'design_parameters',
            'validation_criteria',
            'safety_considerations'
        ]

        missing_sections = []
        for section in required_sections:
            if section not in design_data:
                missing_sections.append(section)

        return missing_sections

    def validate_engineering_standards(self, design_data):
        """Check if design meets basic engineering standards."""
        issues = []

        # Check for quantitative specifications
        if 'design_parameters' in design_data:
            params = design_data['design_parameters']
            if not any(isinstance(v, (int, float)) for v in params.values() if v is not None):
                issues.append("No quantitative design parameters found")

        # Check for safety considerations
        if 'safety_considerations' not in design_data or not design_data['safety_considerations']:
            issues.append("Missing safety considerations")

        # Check for validation criteria
        if 'validation_criteria' not in design_data or not design_data['validation_criteria']:
            issues.append("Missing validation criteria")

        return issues

    async def test_design_generation(self, design_request):
        """Test the complete design generation pipeline."""
        try:
            print(f"Testing design generation for: {design_request}")

            # Generate design using the RAG pipeline
            design_result = await self.engine.create_design_project(design_request)

            if not design_result:
                return {"success": False, "error": "No design generated"}

            # Validate structure
            missing_sections = self.validate_design_structure(design_result)
            if missing_sections:
                return {
                    "success": False,
                    "error": f"Missing required sections: {missing_sections}",
                    "design": design_result
                }

            # Validate engineering standards
            quality_issues = self.validate_engineering_standards(design_result)
            if quality_issues:
                return {
                    "success": False,
                    "error": f"Engineering standard issues: {quality_issues}",
                    "design": design_result
                }

            return {
                "success": True,
                "design": design_result,
                "validation": {
                    "structure_complete": True,
                    "standards_met": True,
                    "sections_present": list(design_result.keys())
                }
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

async def main():
    """Run design quality validation tests."""
    try:
        validator = await DesignQualityValidator().initialize()
    except RuntimeError as e:
        print(f"❌ Failed to initialize validator: {e}")
        return False

    # Test cases for different engineering domains
    test_cases = [
        "Design a high-efficiency solar panel mounting system for residential rooftops",
        "Create specifications for an industrial conveyor belt system with safety interlocks",
        "Design a water filtration system for municipal water treatment",
        "Specify requirements for an automated packaging line with quality control"
    ]

    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i}: {test_case}")
        print(f"{'='*60}")

        result = await validator.test_design_generation(test_case)
        results.append(result)

        if result["success"]:
            print("✅ PASSED: Design meets professional engineering standards")
            print(f"   Generated sections: {', '.join(result['validation']['sections_present'])}")
        else:
            print("❌ FAILED:")
            print(f"   Error: {result['error']}")

        # Show a summary of the design if successful
        if result["success"] and "design" in result:
            design = result["design"]
            print("\n   Design Summary:")
            for section, content in design.items():
                if isinstance(content, dict):
                    print(f"   - {section}: {len(content)} specifications")
                elif isinstance(content, list):
                    print(f"   - {section}: {len(content)} items")
                else:
                    print(f"   - {section}: {str(content)[:100]}...")

    # Overall results
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")

    passed = sum(1 for r in results if r["success"])
    total = len(results)

    print(f"Tests Passed: {passed}/{total} ({passed/total*100:.1f}%)")

    if passed == total:
        print("🎉 All tests passed! RAG pipeline generates professional-quality engineering designs.")
    else:
        print("⚠️  Some tests failed. Review the errors above for improvement areas.")

    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)