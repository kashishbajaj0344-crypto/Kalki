#!/usr/bin/env python3
"""
Test suite for Kalki Agents (Phases 4-12)
"""
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.agents.base_agent import BaseAgent
from modules.agents.phase4_memory import SessionAgent, MemoryAgent
from modules.agents.phase5_reasoning import PlannerAgent, OrchestratorAgent, ComputeOptimizerAgent, CopilotAgent
from modules.agents.phase6_adaptive import MetaHypothesisAgent, FeedbackAgent, PerformanceMonitorAgent, ConflictDetectionAgent
from modules.agents.phase7_knowledge import KnowledgeLifecycleAgent, RollbackManager
from modules.agents.phase8_distributed import ComputeClusterAgent, LoadBalancerAgent, SelfHealingAgent
from modules.agents.phase9_simulation import SimulationAgent, SandboxExperimentAgent, HypotheticalTestingLoop
from modules.agents.phase10_creative import CreativeAgent, PatternRecognitionAgent, IdeaFusionAgent
from modules.agents.phase11_evolutionary import AutoFineTuneAgent, RecursiveKnowledgeGenerator, AutonomousCurriculumDesigner
from modules.agents.phase12_safety import EthicsAgent, RiskAssessmentAgent, OmniEthicsEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kalki.test")


def test_phase4_memory():
    """Test Phase 4: Memory and Session agents"""
    logger.info("=== Testing Phase 4: Memory & Session ===")
    
    # Test SessionAgent
    session_agent = SessionAgent()
    session_agent.initialize()
    
    result = session_agent.execute({
        "action": "create",
        "user_id": "test_user",
        "metadata": {"source": "test"}
    })
    assert result["status"] == "success"
    logger.info(f"✓ SessionAgent created session: {result.get('session_id')}")
    
    # Test MemoryAgent
    memory_agent = MemoryAgent()
    memory_agent.initialize()
    
    result = memory_agent.execute({
        "action": "store",
        "type": "episodic",
        "event": {"summary": "Test event", "details": "Testing memory storage"}
    })
    assert result["status"] == "success"
    logger.info(f"✓ MemoryAgent stored memory: {result.get('memory_id')}")


def test_phase5_reasoning():
    """Test Phase 5: Reasoning and Planning agents"""
    logger.info("=== Testing Phase 5: Reasoning & Planning ===")
    
    # Test PlannerAgent
    planner = PlannerAgent()
    result = planner.execute({
        "action": "create_plan",
        "task": "Ingest PDF and query knowledge",
        "context": {"domain": "science"}
    })
    assert result["status"] == "success"
    logger.info(f"✓ PlannerAgent created plan with {len(result['plan']['subtasks'])} subtasks")
    
    # Test ComputeOptimizerAgent
    optimizer = ComputeOptimizerAgent()
    result = optimizer.execute({
        "action": "get_resources"
    })
    assert result["status"] == "success"
    logger.info(f"✓ ComputeOptimizerAgent reported CPU: {result['resources'].get('cpu_percent')}%")


def test_phase6_adaptive():
    """Test Phase 6: Adaptive Cognition agents"""
    logger.info("=== Testing Phase 6: Adaptive Cognition ===")
    
    # Test MetaHypothesisAgent
    hypothesis_agent = MetaHypothesisAgent()
    result = hypothesis_agent.execute({
        "action": "generate",
        "problem": "Improve retrieval accuracy",
        "observations": [{"metric": "accuracy", "value": 0.7}]
    })
    assert result["status"] == "success"
    logger.info(f"✓ MetaHypothesisAgent generated hypothesis: {result['hypothesis']['hypothesis_id']}")
    
    # Test PerformanceMonitorAgent
    monitor = PerformanceMonitorAgent()
    result = monitor.execute({
        "action": "record",
        "metric_name": "query_latency",
        "value": 0.5
    })
    assert result["status"] == "success"
    logger.info("✓ PerformanceMonitorAgent recorded metric")


def test_phase7_knowledge():
    """Test Phase 7: Knowledge Lifecycle agents"""
    logger.info("=== Testing Phase 7: Knowledge Lifecycle ===")
    
    # Test KnowledgeLifecycleAgent
    lifecycle = KnowledgeLifecycleAgent()
    lifecycle.initialize()
    
    result = lifecycle.execute({
        "action": "create_version",
        "knowledge_id": "test_knowledge",
        "content": {"data": "Test knowledge content"}
    })
    assert result["status"] == "success"
    logger.info(f"✓ KnowledgeLifecycleAgent created version: {result.get('version_id')}")
    
    # Test RollbackManager
    rollback = RollbackManager()
    rollback.initialize()
    
    result = rollback.execute({
        "action": "create",
        "name": "test_checkpoint",
        "state": {"data": "checkpoint state"}
    })
    assert result["status"] == "success"
    logger.info(f"✓ RollbackManager created checkpoint: {result.get('checkpoint_id')}")


def test_phase8_distributed():
    """Test Phase 8: Distributed Compute agents"""
    logger.info("=== Testing Phase 8: Distributed Compute ===")
    
    # Test ComputeClusterAgent
    cluster = ComputeClusterAgent()
    result = cluster.execute({
        "action": "register_node",
        "node_id": "node_1",
        "capabilities": {"cpu": 8, "gpu": 1}
    })
    assert result["status"] == "success"
    logger.info("✓ ComputeClusterAgent registered node")
    
    # Test SelfHealingAgent
    healing = SelfHealingAgent()
    result = healing.execute({
        "action": "health_check",
        "component": "vector_db"
    })
    assert result["status"] == "success"
    logger.info(f"✓ SelfHealingAgent health check: {result['health']['status']}")


def test_phase9_simulation():
    """Test Phase 9: Simulation agents"""
    logger.info("=== Testing Phase 9: Simulation & Experimentation ===")
    
    # Test SimulationAgent
    sim = SimulationAgent()
    result = sim.execute({
        "action": "create",
        "sim_type": "physics",
        "parameters": {"mass": 10, "velocity": 5}
    })
    assert result["status"] == "success"
    sim_id = result.get("sim_id")
    
    result = sim.execute({
        "action": "run",
        "sim_id": sim_id
    })
    assert result["status"] == "success"
    logger.info(f"✓ SimulationAgent ran physics simulation: {result['simulation']['results']['outcome']}")
    
    # Test SandboxExperimentAgent
    sandbox = SandboxExperimentAgent()
    result = sandbox.execute({
        "action": "create_sandbox",
        "sandbox_id": "test_sandbox",
        "isolation_level": "high"
    })
    assert result["status"] == "success"
    logger.info("✓ SandboxExperimentAgent created sandbox")


def test_phase10_creative():
    """Test Phase 10: Creative Cognition agents"""
    logger.info("=== Testing Phase 10: Creative Cognition ===")
    
    # Test CreativeAgent
    creative = CreativeAgent()
    result = creative.execute({
        "action": "generate",
        "domain": "technology"
    })
    assert result["status"] == "success"
    logger.info(f"✓ CreativeAgent generated idea: novelty={result['idea']['novelty_score']:.2f}")
    
    # Test PatternRecognitionAgent
    pattern = PatternRecognitionAgent()
    result = pattern.execute({
        "action": "analyze",
        "data": [{"value": 1}, {"value": 2}, {"value": 3}],
        "analysis_type": "trend"
    })
    assert result["status"] == "success"
    logger.info(f"✓ PatternRecognitionAgent detected {len(result['patterns'])} patterns")


def test_phase11_evolutionary():
    """Test Phase 11: Evolutionary Intelligence agents"""
    logger.info("=== Testing Phase 11: Evolutionary Intelligence ===")
    
    # Test AutoFineTuneAgent
    tuner = AutoFineTuneAgent()
    result = tuner.execute({
        "action": "tune",
        "model_id": "test_model",
        "performance_metrics": {"accuracy": 0.65, "speed": 0.8}
    })
    assert result["status"] == "success"
    logger.info(f"✓ AutoFineTuneAgent tuned model: strategy={result['tuning']['strategy']}")
    
    # Test RecursiveKnowledgeGenerator
    gen = RecursiveKnowledgeGenerator()
    result = gen.execute({
        "action": "generate",
        "topic": "machine_learning",
        "max_depth": 2
    })
    assert result["status"] == "success"
    logger.info(f"✓ RecursiveKnowledgeGenerator created knowledge tree")


def test_phase12_safety():
    """Test Phase 12: Safety & Ethics agents"""
    logger.info("=== Testing Phase 12: Safety & Ethics ===")
    
    # Test EthicsAgent
    ethics = EthicsAgent()
    result = ethics.execute({
        "action": "evaluate",
        "action_to_evaluate": {"type": "help_user", "description": "Assist with query"},
        "context": {}
    })
    assert result["status"] == "success"
    logger.info(f"✓ EthicsAgent evaluation: score={result['evaluation']['ethical_score']:.2f}")
    
    # Test RiskAssessmentAgent
    risk = RiskAssessmentAgent()
    result = risk.execute({
        "action": "assess",
        "action_to_assess": {"type": "data_access", "accesses_data": True}
    })
    assert result["status"] == "success"
    logger.info(f"✓ RiskAssessmentAgent: risk={result['assessment']['severity']}")
    
    # Test OmniEthicsEngine
    omni = OmniEthicsEngine()
    result = omni.execute({
        "action": "simulate",
        "action_to_simulate": {"type": "deploy_feature"}
    })
    assert result["status"] == "success"
    logger.info(f"✓ OmniEthicsEngine: recommendation={result['simulation']['recommendation']}")


def main():
    """Run all tests"""
    logger.info("Starting Kalki Agents Test Suite")
    logger.info("=" * 50)
    
    try:
        test_phase4_memory()
        test_phase5_reasoning()
        test_phase6_adaptive()
        test_phase7_knowledge()
        test_phase8_distributed()
        test_phase9_simulation()
        test_phase10_creative()
        test_phase11_evolutionary()
        test_phase12_safety()
        
        logger.info("=" * 50)
        logger.info("✓ All tests passed successfully!")
        return 0
    except Exception as e:
        logger.exception(f"✗ Test failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
