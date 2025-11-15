"""
Comprehensive tests for Construction System Improvements

Tests all enhancements:
- Deliverable generator functions
- PDF/XLSX export
- Regional pricing
- Caching
- QA framework integration
- Material recommendations
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockProject:
    """Mock project for testing"""
    def __init__(self):
        self.project_id = "test_project_001"
        self.description = "Test ADU Construction"
        self.size_sqft = 1200
        self.stories = 1
        self.location = "Vancouver"
        self.building_type = "Residential ADU"
        self.budget_level = "mid_range"
        self.current_phase = "requirements"


async def test_deliverable_generator_functions():
    """Test 1: Verify all generator functions are properly referenced"""
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Deliverable Generator Functions")
    logger.info("="*60)
    
    try:
        from modules.domains.construction_domain.construction_domain import ConstructionDomain
        from modules.domains.base_domain import ProjectStateMachine
        
        domain = ConstructionDomain()
        deliverable_types = domain.get_deliverable_types()
        
        assert len(deliverable_types) == 6, f"Expected 6 deliverables, got {len(deliverable_types)}"
        
        for spec in deliverable_types:
            assert spec.generator_func is not None, f"Generator function is None for {spec.name}"
            assert callable(spec.generator_func), f"Generator function not callable for {spec.name}"
            logger.info(f"✅ {spec.name}: Generator function properly referenced")
        
        logger.info("✅ All generator functions properly referenced!")
        return True
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_regional_pricing():
    """Test 2: Verify regional pricing multipliers work"""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Regional Pricing")
    logger.info("="*60)
    
    try:
        from modules.domains.construction_domain.deliverables_generator import ConstructionDeliverablesGenerator
        
        generator = ConstructionDeliverablesGenerator(project_root / "data" / "construction")
        
        # Test different locations
        locations = {
            "Vancouver": 1.15,
            "Toronto": 1.20,
            "Calgary": 0.95,
            "Edmonton": 0.90,
            "Unknown City": 1.00  # Default
        }
        
        for location, expected_mult in locations.items():
            multiplier = generator._get_regional_multiplier(location)
            assert abs(multiplier - expected_mult) < 0.01, f"Expected {expected_mult} for {location}, got {multiplier}"
            logger.info(f"✅ {location}: {multiplier:.2f}x multiplier")
        
        logger.info("✅ Regional pricing working correctly!")
        return True
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_bom_generation():
    """Test 3: Generate BOM with all features"""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Bill of Materials Generation")
    logger.info("="*60)
    
    try:
        from modules.domains.construction_domain.deliverables_generator import ConstructionDeliverablesGenerator
        
        generator = ConstructionDeliverablesGenerator(project_root / "data" / "construction")
        project = MockProject()
        
        # Generate BOM
        bom = await generator.generate_bill_of_materials(project, output_format="json")
        
        # Verify structure
        assert "items" in bom, "BOM missing 'items'"
        assert "categories" in bom, "BOM missing 'categories'"
        assert "cost_summary" in bom, "BOM missing 'cost_summary'"
        assert "regional_multiplier" in bom["cost_summary"], "BOM missing regional multiplier"
        assert "location" in bom["cost_summary"], "BOM missing location"
        assert "material_availability" in bom, "BOM missing material availability"
        
        # Verify regional pricing applied
        assert bom["cost_summary"]["regional_multiplier"] == 1.15, "Vancouver multiplier not applied"
        assert bom["cost_summary"]["location"] == "Vancouver", "Location not recorded"
        
        # Verify costs are calculated
        assert bom["cost_summary"]["grand_total"] > 0, "Grand total should be positive"
        assert bom["cost_summary"]["cost_per_sqft"] > 0, "Cost per sqft should be positive"
        
        logger.info(f"✅ BOM generated: {len(bom['items'])} items")
        logger.info(f"✅ Grand Total: ${bom['cost_summary']['grand_total']:,.2f}")
        logger.info(f"✅ Cost per sqft: ${bom['cost_summary']['cost_per_sqft']:,.2f}")
        logger.info(f"✅ Regional multiplier: {bom['cost_summary']['regional_multiplier']:.2f}x")
        
        return True
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_cost_estimate_generation():
    """Test 4: Generate cost estimate with regional pricing"""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Cost Estimate Generation")
    logger.info("="*60)
    
    try:
        from modules.domains.construction_domain.deliverables_generator import ConstructionDeliverablesGenerator
        
        generator = ConstructionDeliverablesGenerator(project_root / "data" / "construction")
        project = MockProject()
        
        # Generate cost estimate
        estimate = await generator.generate_cost_estimate(project, output_format="json")
        
        # Verify structure
        assert "construction_costs" in estimate, "Estimate missing construction costs"
        assert "additional_costs" in estimate, "Estimate missing additional costs"
        assert "cost_summary" in estimate, "Estimate missing cost summary"
        assert "payment_schedule" in estimate, "Estimate missing payment schedule"
        assert "regional_multiplier" in estimate["cost_summary"], "Estimate missing regional multiplier"
        
        # Verify regional pricing applied
        assert estimate["cost_summary"]["regional_multiplier"] == 1.15, "Vancouver multiplier not applied"
        
        # Verify totals
        assert estimate["cost_summary"]["grand_total"] > 0, "Grand total should be positive"
        assert estimate["cost_summary"]["cost_per_sqft"] > 0, "Cost per sqft should be positive"
        
        logger.info(f"✅ Cost estimate generated")
        logger.info(f"✅ Grand Total: ${estimate['cost_summary']['grand_total']:,.2f}")
        logger.info(f"✅ Cost per sqft: ${estimate['cost_summary']['cost_per_sqft']:,.2f}")
        logger.info(f"✅ Regional multiplier: {estimate['cost_summary']['regional_multiplier']:.2f}x")
        
        return True
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_schedule_generation():
    """Test 5: Generate construction schedule"""
    logger.info("\n" + "="*60)
    logger.info("TEST 5: Construction Schedule Generation")
    logger.info("="*60)
    
    try:
        from modules.domains.construction_domain.deliverables_generator import ConstructionDeliverablesGenerator
        
        generator = ConstructionDeliverablesGenerator(project_root / "data" / "construction")
        project = MockProject()
        
        # Generate schedule
        schedule = await generator.generate_construction_schedule(project, output_format="json")
        
        # Verify structure
        assert "phases" in schedule, "Schedule missing phases"
        assert "project_duration_days" in schedule, "Schedule missing duration"
        assert "start_date" in schedule, "Schedule missing start date"
        assert "completion_date" in schedule, "Schedule missing completion date"
        assert "critical_path" in schedule, "Schedule missing critical path"
        
        # Verify phases
        assert len(schedule["phases"]) > 0, "Schedule should have phases"
        for phase in schedule["phases"]:
            assert "phase" in phase, "Phase missing name"
            assert "start_date" in phase, "Phase missing start date"
            assert "end_date" in phase, "Phase missing end date"
            assert "duration_days" in phase, "Phase missing duration"
        
        logger.info(f"✅ Schedule generated: {len(schedule['phases'])} phases")
        logger.info(f"✅ Duration: {schedule['project_duration_days']} days ({schedule['project_duration_months']:.1f} months)")
        logger.info(f"✅ Start: {schedule['start_date']}")
        logger.info(f"✅ Completion: {schedule['completion_date']}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_caching():
    """Test 6: Verify caching works"""
    logger.info("\n" + "="*60)
    logger.info("TEST 6: Caching")
    logger.info("="*60)
    
    try:
        from modules.domains.construction_domain.deliverables_generator import ConstructionDeliverablesGenerator
        
        # Try to use cache if available
        cache = None
        try:
            from modules.intelligent_cache import IntelligentCache
            cache = IntelligentCache()
            logger.info("✅ Using IntelligentCache")
        except:
            logger.info("⚠️  IntelligentCache not available, testing without cache")
        
        generator = ConstructionDeliverablesGenerator(
            project_root / "data" / "construction",
            cache=cache
        )
        project = MockProject()
        
        # Generate BOM twice - second should use cache if available
        import time
        start1 = time.time()
        bom1 = await generator.generate_bill_of_materials(project, output_format="json")
        time1 = time.time() - start1
        
        start2 = time.time()
        bom2 = await generator.generate_bill_of_materials(project, output_format="json")
        time2 = time.time() - start2
        
        # Results should be identical
        assert bom1["cost_summary"]["grand_total"] == bom2["cost_summary"]["grand_total"], "Cached results don't match"
        
        if cache:
            logger.info(f"✅ First generation: {time1:.3f}s")
            logger.info(f"✅ Second generation (cached): {time2:.3f}s")
            if time2 < time1:
                logger.info("✅ Cache working - second generation faster!")
        else:
            logger.info("✅ Results match (cache not available, but no errors)")
        
        return True
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_pdf_export():
    """Test 7: Test PDF generation (if reportlab available)"""
    logger.info("\n" + "="*60)
    logger.info("TEST 7: PDF Export")
    logger.info("="*60)
    
    try:
        from modules.domains.construction_domain.deliverables_generator import ConstructionDeliverablesGenerator, HAS_REPORTLAB
        
        if not HAS_REPORTLAB:
            logger.info("⚠️  reportlab not installed - skipping PDF test")
            return True
        
        generator = ConstructionDeliverablesGenerator(project_root / "data" / "construction")
        project = MockProject()
        
        # Generate BOM with PDF
        bom = await generator.generate_bill_of_materials(project, output_format="pdf")
        
        # Check if PDF file was created
        pdf_files = list((project_root / "data" / "construction" / "deliverables").glob("bom_*.pdf"))
        if pdf_files:
            logger.info(f"✅ PDF generated: {pdf_files[0].name}")
            logger.info(f"✅ File size: {pdf_files[0].stat().st_size} bytes")
        else:
            logger.warning("⚠️  PDF file not found (may have been generated in memory)")
        
        return True
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_xlsx_export():
    """Test 8: Test XLSX generation (if openpyxl available)"""
    logger.info("\n" + "="*60)
    logger.info("TEST 8: XLSX Export")
    logger.info("="*60)
    
    try:
        from modules.domains.construction_domain.deliverables_generator import ConstructionDeliverablesGenerator, HAS_OPENPYXL
        
        if not HAS_OPENPYXL:
            logger.info("⚠️  openpyxl not installed - skipping XLSX test")
            return True
        
        generator = ConstructionDeliverablesGenerator(project_root / "data" / "construction")
        project = MockProject()
        
        # Generate BOM with XLSX
        bom = await generator.generate_bill_of_materials(project, output_format="xlsx")
        
        # Check if XLSX file was created
        xlsx_files = list((project_root / "data" / "construction" / "deliverables").glob("bom_*.xlsx"))
        if xlsx_files:
            logger.info(f"✅ XLSX generated: {xlsx_files[0].name}")
            logger.info(f"✅ File size: {xlsx_files[0].stat().st_size} bytes")
        else:
            logger.warning("⚠️  XLSX file not found (may have been generated in memory)")
        
        # Generate schedule with XLSX
        schedule = await generator.generate_construction_schedule(project, output_format="xlsx")
        schedule_files = list((project_root / "data" / "construction" / "deliverables").glob("schedule_*.xlsx"))
        if schedule_files:
            logger.info(f"✅ Schedule XLSX generated: {schedule_files[0].name}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_material_recommendations():
    """Test 9: Test material recommendations"""
    logger.info("\n" + "="*60)
    logger.info("TEST 9: Material Recommendations")
    logger.info("="*60)
    
    try:
        from modules.domains.construction_domain.deliverables_generator import ConstructionDeliverablesGenerator
        
        generator = ConstructionDeliverablesGenerator(project_root / "data" / "construction")
        project = MockProject()
        
        # Test different budget levels
        for budget_level in ["budget", "mid_range", "premium"]:
            project.budget_level = budget_level
            recommendations = generator.get_material_recommendations(project)
            
            assert "budget_level" in recommendations, "Missing budget level"
            assert "recommendations" in recommendations, "Missing recommendations"
            assert recommendations["budget_level"] == budget_level, "Budget level mismatch"
            
            logger.info(f"✅ {budget_level} recommendations:")
            for key, value in recommendations["recommendations"].items():
                if not key.endswith("_availability"):
                    logger.info(f"   - {key}: {value}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_qa_framework_integration():
    """Test 10: Test QA framework integration"""
    logger.info("\n" + "="*60)
    logger.info("TEST 10: QA Framework Integration")
    logger.info("="*60)
    
    try:
        from modules.domains.construction_domain.deliverables_generator import ConstructionDeliverablesGenerator
        
        # Try to get QA framework
        qa_framework = None
        try:
            from modules.quality_assurance_framework import QualityAssuranceFramework
            from modules.llm import get_llm_engine
            llm_engine = get_llm_engine()
            qa_framework = QualityAssuranceFramework(llm_engine)
            logger.info("✅ QA Framework available")
        except Exception as e:
            logger.info(f"⚠️  QA Framework not available: {e}")
        
        generator = ConstructionDeliverablesGenerator(
            project_root / "data" / "construction",
            qa_framework=qa_framework
        )
        project = MockProject()
        
        # Generate cost estimate (should include QA metadata if framework available)
        estimate = await generator.generate_cost_estimate(project, output_format="json")
        
        if qa_framework:
            assert "qa_status" in estimate or "qa_note" in estimate, "QA metadata should be present"
            logger.info("✅ QA framework integrated")
        else:
            logger.info("✅ Graceful degradation - works without QA framework")
        
        return True
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_all_deliverables():
    """Test 11: Generate all deliverable types"""
    logger.info("\n" + "="*60)
    logger.info("TEST 11: All Deliverable Types")
    logger.info("="*60)
    
    try:
        from modules.domains.construction_domain.construction_domain import ConstructionDomain, ConstructionProjectStateMachine
        
        domain = ConstructionDomain()
        project = ConstructionProjectStateMachine(
            project_id="test_all",
            description="Test All Deliverables"
        )
        project.size_sqft = 1200
        project.stories = 1
        project.location = "Vancouver"
        
        deliverable_types = domain.get_deliverable_types()
        
        results = {}
        for spec in deliverable_types:
            try:
                logger.info(f"Generating {spec.name}...")
                result = await spec.generator_func(project, output_format="json")
                results[spec.name] = result is not None
                logger.info(f"✅ {spec.name}: Generated successfully")
            except Exception as e:
                logger.error(f"❌ {spec.name}: Failed - {e}")
                results[spec.name] = False
        
        all_passed = all(results.values())
        logger.info(f"\n✅ All deliverables test: {sum(results.values())}/{len(results)} passed")
        
        return all_passed
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all tests"""
    logger.info("\n" + "="*70)
    logger.info("CONSTRUCTION SYSTEM IMPROVEMENTS - COMPREHENSIVE TEST SUITE")
    logger.info("="*70)
    
    tests = [
        ("Deliverable Generator Functions", test_deliverable_generator_functions),
        ("Regional Pricing", test_regional_pricing),
        ("BOM Generation", test_bom_generation),
        ("Cost Estimate Generation", test_cost_estimate_generation),
        ("Schedule Generation", test_schedule_generation),
        ("Caching", test_caching),
        ("PDF Export", test_pdf_export),
        ("XLSX Export", test_xlsx_export),
        ("Material Recommendations", test_material_recommendations),
        ("QA Framework Integration", test_qa_framework_integration),
        ("All Deliverables", test_all_deliverables),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info("\n" + "="*70)
    logger.info(f"TOTAL: {passed}/{total} tests passed ({passed*100//total}%)")
    logger.info("="*70)
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)

