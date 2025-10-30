#!/usr/bin/env python3
"""
Kalki v2.3 - Example Usage Demonstrations
Shows how to use agents from Phases 4-12
"""
import logging
from modules.agents.phase4_memory import SessionAgent, MemoryAgent
from modules.agents.phase5_reasoning import PlannerAgent, OrchestratorAgent, ComputeOptimizerAgent
from modules.agents.phase6_adaptive import MetaHypothesisAgent, FeedbackAgent
from modules.agents.phase7_knowledge import KnowledgeLifecycleAgent, RollbackManager
from modules.agents.phase9_simulation import SimulationAgent, SandboxExperimentAgent
from modules.agents.phase10_creative import CreativeAgent, IdeaFusionAgent
from modules.agents.phase11_evolutionary import AutoFineTuneAgent, RecursiveKnowledgeGenerator
from modules.agents.phase12_safety import EthicsAgent, RiskAssessmentAgent, OmniEthicsEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger("kalki.examples")


def example_session_and_memory():
    """Example: Managing user sessions and storing memories"""
    logger.info("=== Example: Session and Memory Management ===")
    
    # Create and manage a session
    session_agent = SessionAgent()
    session_agent.initialize()
    
    result = session_agent.execute({
        "action": "create",
        "user_id": "researcher_001",
        "metadata": {"project": "AI Safety Research", "clearance": "high"}
    })
    session_id = result["session_id"]
    logger.info(f"Created session: {session_id}")
    
    # Update session with context
    session_agent.execute({
        "action": "update",
        "session_id": session_id,
        "context_update": {
            "query": "What are the latest developments in AI alignment?",
            "timestamp": "2025-10-30T17:00:00"
        }
    })
    
    # Store memories
    memory_agent = MemoryAgent()
    memory_agent.initialize()
    
    # Store episodic memory
    memory_agent.execute({
        "action": "store",
        "type": "episodic",
        "event": {
            "summary": "User asked about AI alignment",
            "details": "Queried recent developments in AI safety and alignment",
            "outcome": "Provided comprehensive answer with citations"
        }
    })
    
    # Store semantic memory
    memory_agent.execute({
        "action": "store",
        "type": "semantic",
        "concept": "AI_alignment",
        "knowledge": {
            "definition": "Process of ensuring AI systems act in accordance with human values",
            "importance": "Critical for safe AI deployment",
            "key_methods": ["RLHF", "Constitutional AI", "Value learning"]
        }
    })
    
    # Recall recent memories
    result = memory_agent.execute({
        "action": "recall",
        "type": "episodic",
        "limit": 5
    })
    logger.info(f"Recalled {len(result['memories'])} episodic memories")


def example_planning_and_orchestration():
    """Example: Planning complex tasks and orchestrating agents"""
    logger.info("=== Example: Planning and Orchestration ===")
    
    # Create a plan for a complex task
    planner = PlannerAgent()
    plan_result = planner.execute({
        "action": "create_plan",
        "task": "Ingest research papers, analyze patterns, generate summary report",
        "context": {"domain": "machine_learning", "priority": "high"}
    })
    
    plan = plan_result["plan"]
    logger.info(f"Created plan with {len(plan['subtasks'])} subtasks")
    for subtask in plan['subtasks']:
        logger.info(f"  - {subtask['description']}")
    
    # Orchestrate execution
    orchestrator = OrchestratorAgent()
    workflow_id = f"workflow_{plan['plan_id']}"
    
    orchestrator.execute({
        "action": "create_workflow",
        "workflow_id": workflow_id,
        "plan": plan
    })
    
    result = orchestrator.execute({
        "action": "execute_workflow",
        "workflow_id": workflow_id
    })
    logger.info(f"Workflow status: {result['workflow']['status']}")


def example_creative_innovation():
    """Example: Creative idea generation and fusion"""
    logger.info("=== Example: Creative Innovation ===")
    
    creative = CreativeAgent()
    
    # Generate ideas in different domains
    idea1 = creative.execute({
        "action": "generate",
        "domain": "technology",
        "constraints": {"focus": "sustainability"}
    })
    logger.info(f"Tech idea: {idea1['idea']['description']}")
    logger.info(f"  Novelty: {idea1['idea']['novelty_score']:.2f}")
    
    idea2 = creative.execute({
        "action": "generate",
        "domain": "biology"
    })
    logger.info(f"Bio idea: {idea2['idea']['description']}")
    
    # Dream mode - unconstrained exploration
    creative.enable_dream_mode()
    dreams = creative.execute({
        "action": "dream",
        "theme": "future of computing"
    })
    logger.info(f"Dream mode generated {len(dreams['dreams'])} ideas")
    
    # Fuse ideas for cross-domain innovation
    fusion = IdeaFusionAgent()
    result = fusion.execute({
        "action": "fuse",
        "ideas": [idea1["idea"], idea2["idea"]]
    })
    logger.info(f"Fused idea: {result['fusion']['description']}")
    logger.info(f"  Cross-domain value: {result['fusion']['cross_domain_value']:.2f}")


def example_simulation_and_experimentation():
    """Example: Running simulations in sandbox"""
    logger.info("=== Example: Simulation and Experimentation ===")
    
    sim_agent = SimulationAgent()
    
    # Create a physics simulation
    sim_result = sim_agent.execute({
        "action": "create",
        "sim_type": "physics",
        "parameters": {
            "scenario": "particle_collision",
            "mass": 10,
            "velocity": 50,
            "friction": 0.05
        }
    })
    sim_id = sim_result["sim_id"]
    
    # Run the simulation
    result = sim_agent.execute({
        "action": "run",
        "sim_id": sim_id
    })
    logger.info(f"Simulation result: {result['simulation']['results']}")
    
    # Run experiment in sandbox
    sandbox = SandboxExperimentAgent()
    sandbox.execute({
        "action": "create_sandbox",
        "sandbox_id": "exp_sandbox_001",
        "isolation_level": "high"
    })
    
    exp_result = sandbox.execute({
        "action": "run_experiment",
        "sandbox_id": "exp_sandbox_001",
        "experiment": {
            "type": "parameter_tuning",
            "parameters": {"learning_rate": 0.001, "batch_size": 32}
        }
    })
    logger.info(f"Experiment status: {exp_result['experiment']['status']}")


def example_ethics_and_safety():
    """Example: Ethical validation and risk assessment"""
    logger.info("=== Example: Ethics and Safety ===")
    
    # Evaluate ethical implications
    ethics = EthicsAgent(config={"framework": "utilitarian"})
    
    action_to_evaluate = {
        "type": "deploy_model",
        "description": "Deploy new recommendation system",
        "affects_users": True,
        "data_access": "user_preferences"
    }
    
    result = ethics.execute({
        "action": "evaluate",
        "action_to_evaluate": action_to_evaluate,
        "context": {"sensitivity": "medium", "scale": "large"}
    })
    
    evaluation = result["evaluation"]
    logger.info(f"Ethical score: {evaluation['ethical_score']:.2f}")
    logger.info(f"Recommendation: {evaluation['recommendation']}")
    if evaluation['issues']:
        logger.info("Ethical issues identified:")
        for issue in evaluation['issues']:
            logger.info(f"  - {issue}")
    
    # Assess risks
    risk_agent = RiskAssessmentAgent(config={"threshold": 0.7})
    
    risk_result = risk_agent.execute({
        "action": "assess",
        "action_to_assess": action_to_evaluate,
        "domain": "machine_learning"
    })
    
    assessment = risk_result["assessment"]
    logger.info(f"Overall risk: {assessment['overall_risk']:.2f} ({assessment['severity']})")
    logger.info("Suggested mitigations:")
    for mitigation in assessment['mitigations']:
        logger.info(f"  - {mitigation}")
    
    # Simulate consequences
    omni = OmniEthicsEngine()
    
    consequence_result = omni.execute({
        "action": "simulate",
        "action_to_simulate": action_to_evaluate,
        "time_horizons": ["immediate", "short_term", "long_term"]
    })
    
    sim = consequence_result["simulation"]
    logger.info(f"Overall impact: {sim['overall_impact']:.2f}")
    logger.info(f"Recommendation: {sim['recommendation']}")


def example_knowledge_lifecycle():
    """Example: Knowledge versioning and rollback"""
    logger.info("=== Example: Knowledge Lifecycle Management ===")
    
    lifecycle = KnowledgeLifecycleAgent()
    lifecycle.initialize()
    
    # Create knowledge versions
    v1 = lifecycle.execute({
        "action": "create_version",
        "knowledge_id": "ai_safety_best_practices",
        "content": {
            "practices": ["Testing", "Monitoring", "Sandboxing"],
            "version_notes": "Initial version"
        },
        "metadata": {"author": "Safety Team", "reviewed": True}
    })
    logger.info(f"Created version: {v1['version_id']}")
    
    v2 = lifecycle.execute({
        "action": "create_version",
        "knowledge_id": "ai_safety_best_practices",
        "content": {
            "practices": ["Testing", "Monitoring", "Sandboxing", "Formal Verification"],
            "version_notes": "Added formal verification"
        },
        "metadata": {"author": "Safety Team", "reviewed": True}
    })
    logger.info(f"Updated version: {v2['version_id']}")
    
    # Create checkpoints for rollback
    rollback = RollbackManager()
    rollback.initialize()
    
    checkpoint = rollback.execute({
        "action": "create",
        "name": "before_major_update",
        "state": {
            "knowledge_version": v2['version_id'],
            "system_config": {"mode": "production"}
        }
    })
    logger.info(f"Created checkpoint: {checkpoint['checkpoint_id']}")


def example_self_improvement():
    """Example: Auto-tuning and recursive learning"""
    logger.info("=== Example: Self-Improvement ===")
    
    # Auto-tune a model
    tuner = AutoFineTuneAgent()
    
    result = tuner.execute({
        "action": "tune",
        "model_id": "query_model_v1",
        "performance_metrics": {
            "accuracy": 0.68,
            "speed": 0.75,
            "user_satisfaction": 0.70
        }
    })
    
    tuning = result["tuning"]
    logger.info(f"Tuning strategy: {tuning['strategy']}")
    logger.info(f"Expected improvement: {tuning['improvement']:.2%}")
    
    # Generate knowledge recursively
    gen = RecursiveKnowledgeGenerator()
    
    result = gen.execute({
        "action": "generate",
        "topic": "reinforcement_learning",
        "max_depth": 3
    })
    
    knowledge = result["knowledge"]
    logger.info(f"Generated knowledge tree for '{knowledge['topic']}'")
    logger.info(f"Total concepts: {len(knowledge['knowledge'])}")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("Kalki v2.3 - Agent Usage Examples")
    print("="*70 + "\n")
    
    example_session_and_memory()
    print()
    
    example_planning_and_orchestration()
    print()
    
    example_creative_innovation()
    print()
    
    example_simulation_and_experimentation()
    print()
    
    example_ethics_and_safety()
    print()
    
    example_knowledge_lifecycle()
    print()
    
    example_self_improvement()
    print()
    
    print("="*70)
    print("All examples completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
