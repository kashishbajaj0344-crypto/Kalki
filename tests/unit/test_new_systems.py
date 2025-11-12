"""
Comprehensive tests for new professional systems:
- Professional Team Orchestrator
- Professional Deliverable Generator
- Cross-Domain Learning
- Professional Workflow System
- Quality Assurance Framework
- Construction Copilot Integration
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test results
test_results = {
    "passed": [],
    "failed": [],
    "warnings": []
}


def log_test(name: str, passed: bool, message: str = ""):
    """Log test result"""
    if passed:
        test_results["passed"].append(name)
        logger.info(f"✅ {name}: PASSED {message}")
    else:
        test_results["failed"].append(name)
        logger.error(f"❌ {name}: FAILED {message}")


def log_warning(name: str, message: str):
    """Log test warning"""
    test_results["warnings"].append(f"{name}: {message}")
    logger.warning(f"⚠️ {name}: {message}")


async def test_professional_team_orchestrator():
    """Test Professional Team Orchestrator"""
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Professional Team Orchestrator")
    logger.info("="*60)
    
    try:
        from modules.agents.agent_manager import AgentManager
        from modules.agents.event_bus import EventBus
        from modules.llm import LLMEngine
        from modules.professional_team_orchestrator import (
            ProfessionalTeamOrchestrator,
            ProfessionalRole
        )
        from modules.agents.base_agent import AgentCapability
        
        # Initialize
        event_bus = EventBus()
        agent_manager = AgentManager(event_bus)
        llm_engine = LLMEngine()
        orchestrator = ProfessionalTeamOrchestrator(agent_manager, llm_engine)
        
        # Test role assignment
        result = await orchestrator.assign_role(
            role=ProfessionalRole.ARCHITECT,
            agent_capability=AgentCapability.DESIGN
        )
        log_test("Role Assignment", result, "Architect role assigned")
        
        # Test team coordination
        team_status = orchestrator.get_team_status()
        log_test("Team Status", len(team_status) > 0, f"Team has {len(team_status)} roles")
        
        # Test task delegation
        task_result = await orchestrator.coordinate_team_task(
            task="Design a 1200 sqft ADU floor plan",
            required_roles=[ProfessionalRole.ARCHITECT],
            context={"project_type": "adu", "square_feet": 1200},
            domain="construction"
        )
        log_test("Task Delegation", task_result is not None, "Task delegated successfully")
        
        return True
        
    except Exception as e:
        log_test("Professional Team Orchestrator", False, str(e))
        return False


async def test_professional_deliverable_generator():
    """Test Professional Deliverable Generator"""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Professional Deliverable Generator")
    logger.info("="*60)
    
    try:
        from modules.llm import LLMEngine
        from modules.visual_knowledge_graph import VisualKnowledgeGraph
        from modules.professional_deliverable_generator import (
            ProfessionalDeliverableGenerator,
            DeliverableType
        )
        from modules.domains.base_domain import ProjectStateMachine
        
        # Initialize
        llm_engine = LLMEngine()
        knowledge_graph = VisualKnowledgeGraph()
        generator = ProfessionalDeliverableGenerator(llm_engine, knowledge_graph)
        
        # Create mock project
        class MockProject:
            project_id = "test_project_001"
            description = "Test ADU Project"
            domain = "construction"
            current_phase = "design"
        
        project = MockProject()
        
        # Test document generation
        output_dir = Path("output/test_deliverables")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        doc_path = await generator.generate_deliverable(
            project=project,
            deliverable_type=DeliverableType.TECHNICAL_DOCUMENT,
            specifications={"title": "Test Document", "content": "Test content"},
            output_dir=output_dir
        )
        log_test("Document Generation", doc_path is not None and doc_path.exists(), 
                f"Document generated at {doc_path}")
        
        # Test BOM generation
        bom_path = await generator.generate_deliverable(
            project=project,
            deliverable_type=DeliverableType.BILL_OF_MATERIALS,
            specifications={"items": ["Lumber", "Concrete", "Steel"]},
            output_dir=output_dir
        )
        log_test("BOM Generation", bom_path is not None, "BOM generated")
        
        return True
        
    except Exception as e:
        log_test("Professional Deliverable Generator", False, str(e))
        import traceback
        logger.error(traceback.format_exc())
        return False


async def test_cross_domain_learning():
    """Test Cross-Domain Learning"""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Cross-Domain Learning")
    logger.info("="*60)
    
    try:
        from modules.llm import LLMEngine
        from modules.meta_learning_system import MetaLearningSystem
        from modules.domains.domain_registry import DomainRegistry
        from modules.cross_domain_learning import CrossDomainLearning
        from modules.domains.base_domain import ProjectStateMachine
        
        # Initialize
        llm_engine = LLMEngine()
        meta_learning = MetaLearningSystem()
        domain_registry = DomainRegistry()
        cross_learning = CrossDomainLearning(domain_registry, meta_learning, llm_engine)
        
        # Test skill retrieval
        skills = cross_learning.get_transferable_skills()
        log_test("Skill Retrieval", len(skills) > 0, 
                f"Retrieved {len(skills)} transferable skills")
        
        # Test skill transfer
        if "project_management" in skills:
            transfer_result = await cross_learning.transfer_skill(
                source_domain="construction",
                target_domain="game_dev",
                skill="project_management"
            )
            log_test("Skill Transfer", "error" not in transfer_result, 
                    "Skill transferred successfully" if "error" not in transfer_result else transfer_result.get("error", "Unknown error"))
        else:
            log_warning("Skill Transfer", "project_management skill not found")
        
        return True
        
    except Exception as e:
        log_test("Cross-Domain Learning", False, str(e))
        import traceback
        logger.error(traceback.format_exc())
        return False


async def test_professional_workflow():
    """Test Professional Workflow System"""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Professional Workflow System")
    logger.info("="*60)
    
    try:
        from modules.agents.agent_manager import AgentManager
        from modules.agents.event_bus import EventBus
        from modules.llm import LLMEngine
        from modules.professional_team_orchestrator import ProfessionalTeamOrchestrator
        from modules.professional_workflow import (
            ProfessionalWorkflow,
            ProfessionalWorkflowExecutor,
            WorkflowStep,
            WorkflowStepStatus,
            ProfessionalRole
        )
        
        # Initialize
        event_bus = EventBus()
        agent_manager = AgentManager(event_bus)
        llm_engine = LLMEngine()
        team_orchestrator = ProfessionalTeamOrchestrator(agent_manager, llm_engine)
        workflow_executor = ProfessionalWorkflowExecutor(team_orchestrator, llm_engine)
        
        # Create test workflow
        workflow = ProfessionalWorkflow(
            name="Construction Design Workflow",
            description="Design → Validate → Schedule → Estimate",
            domain="construction",
            steps=[
                WorkflowStep(
                    name="design",
                    role=ProfessionalRole.ARCHITECT,
                    action="Design building layout",
                    inputs=[],
                    outputs=["design_spec"]
                ),
                WorkflowStep(
                    name="validate",
                    role=ProfessionalRole.STRUCTURAL_ENGINEER,
                    action="Validate structural integrity",
                    inputs=["design"],
                    outputs=["validation_report"]
                ),
                WorkflowStep(
                    name="schedule",
                    role=ProfessionalRole.PROJECT_MANAGER,
                    action="Create project schedule",
                    inputs=["design", "validate"],
                    outputs=["schedule"]
                ),
                WorkflowStep(
                    name="estimate",
                    role=ProfessionalRole.COST_ESTIMATOR,
                    action="Provide cost estimate",
                    inputs=["design", "schedule"],
                    outputs=["cost_estimate"]
                )
            ]
        )
        
        # Test workflow generation
        generated_workflow = await workflow_executor.generate_workflow_from_requirements(
            requirements="Design a 1200 sqft ADU with validation, scheduling, and cost estimation",
            domain="construction",
            context={"square_feet": 1200, "project_type": "adu"}
        )
        log_test("Workflow Generation", generated_workflow is not None, 
                f"Generated workflow: {generated_workflow.name}")
        
        # Test workflow execution (simplified - may take time)
        log_warning("Workflow Execution", "Skipping full execution test (requires agents)")
        
        return True
        
    except Exception as e:
        log_test("Professional Workflow System", False, str(e))
        import traceback
        logger.error(traceback.format_exc())
        return False


async def test_quality_assurance():
    """Test Quality Assurance Framework"""
    logger.info("\n" + "="*60)
    logger.info("TEST 5: Quality Assurance Framework")
    logger.info("="*60)
    
    try:
        from modules.llm import LLMEngine
        from modules.quality_assurance_framework import (
            QualityAssuranceFramework,
            QualityStandard
        )
        from modules.professional_deliverable_generator import DeliverableType
        
        # Initialize
        llm_engine = LLMEngine()
        qa_framework = QualityAssuranceFramework(llm_engine)
        
        # Create test deliverable
        test_dir = Path("output/test_qa")
        test_dir.mkdir(parents=True, exist_ok=True)
        test_file = test_dir / "test_document.txt"
        test_file.write_text("""
        Construction Project Specification
        ==================================
        
        Project: Test ADU
        Square Footage: 1200 sqft
        Building Type: Residential ADU
        Structural System: Wood Frame
        Foundation: Concrete Slab
        
        This is a test document for quality assurance validation.
        """)
        
        # Test validation
        validation_result = await qa_framework.validate_deliverable(
            deliverable=test_file,
            deliverable_type=DeliverableType.TECHNICAL_DOCUMENT,
            quality_standard=QualityStandard.BUILDING_CODE,
            domain="construction"
        )
        
        log_test("Quality Validation", validation_result is not None,
                f"Validation score: {validation_result.overall_score:.2f}")
        log_test("Validation Checks", len(validation_result.checks) > 0,
                f"Performed {len(validation_result.checks)} checks")
        log_test("Recommendations", len(validation_result.recommendations) > 0,
                f"Generated {len(validation_result.recommendations)} recommendations")
        
        return True
        
    except Exception as e:
        log_test("Quality Assurance Framework", False, str(e))
        import traceback
        logger.error(traceback.format_exc())
        return False


async def test_construction_copilot_integration():
    """Test Construction Copilot Integration"""
    logger.info("\n" + "="*60)
    logger.info("TEST 6: Construction Copilot Integration")
    logger.info("="*60)
    
    try:
        from modules.construction_copilot_enhanced import EnhancedConstructionCopilot
        
        # Initialize copilot
        copilot = EnhancedConstructionCopilot()
        
        # Test that new systems are initialized
        has_team_orchestrator = hasattr(copilot, 'team_orchestrator')
        log_test("Team Orchestrator Integration", has_team_orchestrator,
                "Team orchestrator initialized")
        
        has_deliverable_generator = hasattr(copilot, 'deliverable_generator')
        log_test("Deliverable Generator Integration", has_deliverable_generator,
                "Deliverable generator initialized")
        
        has_cross_learning = hasattr(copilot, 'cross_learning')
        log_test("Cross-Domain Learning Integration", has_cross_learning,
                "Cross-domain learning initialized")
        
        has_workflow_executor = hasattr(copilot, 'workflow_executor')
        log_test("Workflow Executor Integration", has_workflow_executor,
                "Workflow executor initialized")
        
        has_quality_framework = hasattr(copilot, 'quality_framework')
        log_test("Quality Framework Integration", has_quality_framework,
                "Quality framework initialized")
        
        # Test role initialization
        await copilot._ensure_roles_initialized()
        team_status = copilot.team_orchestrator.get_team_status()
        log_test("Role Initialization", len(team_status) > 0,
                f"Initialized {len(team_status)} professional roles")
        
        return True
        
    except Exception as e:
        log_test("Construction Copilot Integration", False, str(e))
        import traceback
        logger.error(traceback.format_exc())
        return False


async def run_all_tests():
    """Run all tests"""
    logger.info("\n" + "="*80)
    logger.info("COMPREHENSIVE TEST SUITE FOR NEW PROFESSIONAL SYSTEMS")
    logger.info("="*80)
    
    tests = [
        ("Professional Team Orchestrator", test_professional_team_orchestrator),
        ("Professional Deliverable Generator", test_professional_deliverable_generator),
        ("Cross-Domain Learning", test_cross_domain_learning),
        ("Professional Workflow System", test_professional_workflow),
        ("Quality Assurance Framework", test_quality_assurance),
        ("Construction Copilot Integration", test_construction_copilot_integration),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = await test_func()
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    logger.info(f"✅ Passed: {len(test_results['passed'])}")
    logger.info(f"❌ Failed: {len(test_results['failed'])}")
    logger.info(f"⚠️  Warnings: {len(test_results['warnings'])}")
    
    logger.info("\nPassed Tests:")
    for test in test_results['passed']:
        logger.info(f"  ✅ {test}")
    
    if test_results['failed']:
        logger.info("\nFailed Tests:")
        for test in test_results['failed']:
            logger.info(f"  ❌ {test}")
    
    if test_results['warnings']:
        logger.info("\nWarnings:")
        for warning in test_results['warnings']:
            logger.info(f"  ⚠️  {warning}")
    
    logger.info("\n" + "="*80)
    success_rate = len(test_results['passed']) / (len(test_results['passed']) + len(test_results['failed'])) * 100 if (len(test_results['passed']) + len(test_results['failed'])) > 0 else 0
    logger.info(f"Success Rate: {success_rate:.1f}%")
    logger.info("="*80)
    
    return results


if __name__ == "__main__":
    asyncio.run(run_all_tests())

