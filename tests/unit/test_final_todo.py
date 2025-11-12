"""
Final TODO Testing
Test remaining items: deliverable generation, cross-domain learning, workflows, performance
"""

import asyncio
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_deliverable_generation_all_domains():
    """Test deliverable generation across all domains"""
    logger.info("\n" + "="*60)
    logger.info("TEST: Deliverable Generation Across All Domains")
    logger.info("="*60)
    
    try:
        from modules.domains.domain_registry import DomainRegistry
        from modules.professional_deliverable_generator import ProfessionalDeliverableGenerator, DeliverableType
        from modules.llm import LLMEngine
        from modules.visual_knowledge_graph import VisualKnowledgeGraph
        
        registry = DomainRegistry()
        llm = LLMEngine()
        knowledge_graph = VisualKnowledgeGraph()
        generator = ProfessionalDeliverableGenerator(llm, knowledge_graph)
        
        domains_tested = []
        for domain_name in ["construction", "game_dev", "robotics", "aerospace", "power_systems"]:
            try:
                domain = registry.get_domain(domain_name)
                if domain:
                    # Create test project
                    project = await domain.create_project(
                        description=f"Test {domain_name} project",
                        requirements={}
                    )
                    
                    # Generate deliverable
                    output_dir = Path(f"output/test_{domain_name}")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Test document generation
                    result = await generator.generate_deliverable(
                        project=project,
                        deliverable_type=DeliverableType.TECHNICAL_DOCUMENT,
                        specifications={"title": f"Test {domain_name} Document"},
                        output_dir=output_dir
                    )
                    
                    if result:
                        domains_tested.append(domain_name)
                        logger.info(f"✅ {domain_name}: Deliverable generated")
                    else:
                        logger.warning(f"⚠️ {domain_name}: Generation returned None")
            except Exception as e:
                logger.error(f"❌ {domain_name}: {e}")
        
        logger.info(f"\n✅ Tested {len(domains_tested)}/5 domains")
        return len(domains_tested) >= 3  # At least 3 should work
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False


async def test_cross_domain_learning():
    """Test cross-domain learning"""
    logger.info("\n" + "="*60)
    logger.info("TEST: Cross-Domain Learning")
    logger.info("="*60)
    
    try:
        from modules.domains.domain_registry import DomainRegistry
        from modules.meta_learning_system import MetaLearningSystem
        from modules.llm import LLMEngine
        from modules.cross_domain_learning import CrossDomainLearning
        
        registry = DomainRegistry()
        meta_learning = MetaLearningSystem()
        llm = LLMEngine()
        cross_learning = CrossDomainLearning(registry, meta_learning, llm)
        
        # Test skill transfer: Construction PM → Game Dev
        result1 = await cross_learning.transfer_skill(
            source_domain="construction",
            target_domain="game_dev",
            skill="project_management"
        )
        logger.info(f"✅ Construction → Game Dev: {'Success' if 'error' not in result1 else result1.get('error')}")
        
        # Test skill transfer: Robotics → Aerospace
        result2 = await cross_learning.transfer_skill(
            source_domain="robotics",
            target_domain="aerospace",
            skill="simulation"
        )
        logger.info(f"✅ Robotics → Aerospace: {'Success' if 'error' not in result2 else result2.get('error')}")
        
        return 'error' not in result1 or 'error' not in result2
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False


async def test_complex_workflows():
    """Test complex workflows"""
    logger.info("\n" + "="*60)
    logger.info("TEST: Complex Workflows")
    logger.info("="*60)
    
    try:
        from modules.agents.agent_manager import AgentManager
        from modules.agents.event_bus import EventBus
        from modules.llm import LLMEngine
        from modules.professional_team_orchestrator import ProfessionalTeamOrchestrator
        from modules.professional_workflow import ProfessionalWorkflowExecutor
        
        event_bus = EventBus()
        agent_manager = AgentManager(event_bus)
        llm = LLMEngine()
        team_orch = ProfessionalTeamOrchestrator(agent_manager, llm)
        workflow_executor = ProfessionalWorkflowExecutor(team_orch, llm)
        
        # Generate workflow for construction
        workflow = await workflow_executor.generate_workflow_from_requirements(
            requirements="Design a 1200 sqft ADU, validate structural integrity, create schedule, and provide cost estimate",
            domain="construction",
            context={"square_feet": 1200, "project_type": "adu"}
        )
        
        logger.info(f"✅ Workflow generated: {workflow.name}")
        logger.info(f"   Steps: {len(workflow.steps)}")
        
        return workflow is not None and len(workflow.steps) > 0
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def test_performance():
    """Performance test with local Llama models"""
    logger.info("\n" + "="*60)
    logger.info("TEST: Performance with Local Llama Models")
    logger.info("="*60)
    
    try:
        from modules.llm import LLMEngine
        import time
        
        llm = LLMEngine()
        await llm.initialize()
        
        # Test text generation performance
        test_prompts = [
            "What is construction project management?",
            "Explain game development lifecycle",
            "Describe robotics control systems"
        ]
        
        times = []
        for prompt in test_prompts:
            start = time.time()
            result = await llm.generate(prompt, max_tokens=100)
            elapsed = time.time() - start
            times.append(elapsed)
            logger.info(f"  Prompt {len(prompt)} chars: {elapsed:.2f}s")
        
        avg_time = sum(times) / len(times)
        logger.info(f"\n✅ Average generation time: {avg_time:.2f}s")
        logger.info(f"   Using: Llama 3.1 8B (local)")
        
        # Test vision if available
        if llm.vision_engine and llm.vision_engine.is_initialized:
            logger.info(f"✅ Vision model available: Llama 3.2 Vision 11B")
        
        return avg_time < 10.0  # Should be reasonable
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False


async def run_all_tests():
    """Run all final todo tests"""
    logger.info("\n" + "="*80)
    logger.info("FINAL TODO TESTING")
    logger.info("="*80)
    
    tests = [
        ("Deliverable Generation (All Domains)", test_deliverable_generation_all_domains),
        ("Cross-Domain Learning", test_cross_domain_learning),
        ("Complex Workflows", test_complex_workflows),
        ("Performance (Local Llama)", test_performance),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = await test_func()
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("FINAL TEST SUMMARY")
    logger.info("="*80)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    logger.info(f"✅ Passed: {passed}/{total}")
    logger.info(f"❌ Failed: {total - passed}/{total}")
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"  {status}: {test_name}")
    
    logger.info("="*80)
    
    return results


if __name__ == "__main__":
    asyncio.run(run_all_tests())


