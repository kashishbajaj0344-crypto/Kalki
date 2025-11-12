#!/usr/bin/env python3
"""
Test script for evolutionary agents integration (Phase 24)
Verifies that all 13 evolutionary agents are properly registered
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from kalki_complete import KalkiOrchestrator
from modules.utils.logging_config import setup_logging, get_logger

async def test_evolutionary_agents():
    """Test that evolutionary agents are properly integrated"""
    setup_logging(log_level="INFO")
    logger = get_logger("Test.EvolutionaryAgents")
    
    logger.info("🧪 Testing Evolutionary Agents Integration (Phase 24)")
    
    # Initialize orchestrator
    orchestrator = KalkiOrchestrator()
    success = await orchestrator.initialize_system()
    
    if not success:
        logger.error("❌ System initialization failed")
        return False
    
    # Get system status
    status = await orchestrator.get_system_status()
    
    # Verify evolutionary agents
    logger.info(f"📊 System Status:")
    logger.info(f"  - Total Phases Active: {status['phases_active']}")
    logger.info(f"  - Total Agents: {status['total_agents']}")
    logger.info(f"  - Evolutionary Agents: {status['evolutionary_agents_count']}")
    logger.info(f"  - Design Engine Active: {status['design_engine_active']}")
    logger.info(f"  - Supreme Synthesis Active: {status['supreme_synthesis_active']}")
    
    # Check phase agents
    evolutionary_agents = orchestrator.phase_agents.get('evolutionary', [])
    
    logger.info(f"\n🧬 Evolutionary Agents Registered ({len(evolutionary_agents)} total):")
    
    expected_agents = [
        "AutoFineTuneAgent",
        "AutonomousCurriculumDesigner", 
        "RecursiveKnowledgeGenerator",
        "KnowledgeLifecycleAgent",
        "RollbackManager",
        "DreamModeAgent",
        "IdeaFusionAgent",
        "PatternRecognitionAgent",
        "ConsensusAgent",
        "ComputeClusterAgent",
        "ObservabilityAgent",
        "SensorFusionAgent",
        "ARInsightAgent"
    ]
    
    found_agents = []
    for agent in evolutionary_agents:
        agent_name = agent.__class__.__name__
        found_agents.append(agent_name)
        logger.info(f"  ✅ {agent_name}")
    
    # Verify all expected agents are present
    missing_agents = set(expected_agents) - set(found_agents)
    extra_agents = set(found_agents) - set(expected_agents)
    
    if missing_agents:
        logger.error(f"❌ Missing agents: {missing_agents}")
        return False
    
    if extra_agents:
        logger.warning(f"⚠️ Unexpected agents: {extra_agents}")
    
    if len(found_agents) == len(expected_agents):
        logger.info(f"\n✅ All {len(expected_agents)} evolutionary agents successfully registered!")
        logger.info(f"✅ Phase 24 integration complete!")
        return True
    else:
        logger.error(f"❌ Expected {len(expected_agents)} agents, found {len(found_agents)}")
        return False

async def main():
    """Main test runner"""
    try:
        success = await test_evolutionary_agents()
        
        if success:
            print("\n" + "="*80)
            print("🎉 EVOLUTIONARY AGENTS INTEGRATION TEST PASSED")
            print("="*80)
            sys.exit(0)
        else:
            print("\n" + "="*80)
            print("❌ EVOLUTIONARY AGENTS INTEGRATION TEST FAILED")
            print("="*80)
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
