#!/usr/bin/env python3
"""
Standalone Agent System Demo
Demonstrates agent orchestration without RAG dependencies
"""
import logging
from modules.agents.phase4_memory import SessionAgent, MemoryAgent
from modules.agents.phase5_reasoning import PlannerAgent, OrchestratorAgent, ComputeOptimizerAgent
from modules.agents.phase12_safety import EthicsAgent, RiskAssessmentAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(message)s'
)
logger = logging.getLogger("kalki.standalone")


def demo_full_workflow():
    """Demonstrate a complete workflow using multiple agents"""
    logger.info("="*70)
    logger.info("Kalki Agent System - Complete Workflow Demo")
    logger.info("="*70 + "\n")
    
    # Initialize agents
    session_agent = SessionAgent()
    session_agent.initialize()
    memory_agent = MemoryAgent()
    memory_agent.initialize()
    planner = PlannerAgent()
    orchestrator = OrchestratorAgent()
    compute_optimizer = ComputeOptimizerAgent()
    ethics_agent = EthicsAgent()
    risk_agent = RiskAssessmentAgent()
    
    logger.info("Step 1: Creating user session...")
    session_result = session_agent.execute({
        "action": "create",
        "user_id": "research_user",
        "metadata": {"project": "AI Safety Research", "clearance": "high"}
    })
    session_id = session_result["session_id"]
    logger.info(f"✓ Created session: {session_id}\n")
    
    # Simulate a research task
    task = "Analyze recent AI safety papers and generate summary report"
    logger.info(f"Step 2: Planning task: '{task}'...")
    
    plan_result = planner.execute({
        "action": "create_plan",
        "task": task,
        "context": {"domain": "AI_safety", "priority": "high"}
    })
    plan = plan_result["plan"]
    logger.info(f"✓ Created plan with {len(plan['subtasks'])} subtasks:")
    for i, subtask in enumerate(plan['subtasks'], 1):
        logger.info(f"  {i}. {subtask['description']}")
    logger.info("")
    
    logger.info("Step 3: Ethics validation...")
    ethics_result = ethics_agent.execute({
        "action": "evaluate",
        "action_to_evaluate": {
            "type": "research_analysis",
            "task": task,
            "accesses_data": True,
            "affects_safety": False
        },
        "context": {"domain": "research", "sensitivity": "medium"}
    })
    evaluation = ethics_result["evaluation"]
    logger.info(f"✓ Ethics score: {evaluation['ethical_score']:.2f}")
    logger.info(f"  Recommendation: {evaluation['recommendation']}")
    if evaluation['issues']:
        logger.info(f"  Issues: {', '.join(evaluation['issues'])}")
    logger.info("")
    
    logger.info("Step 4: Risk assessment...")
    risk_result = risk_agent.execute({
        "action": "assess",
        "action_to_assess": {
            "type": "data_analysis",
            "task": task,
            "accesses_data": True
        },
        "domain": "research"
    })
    assessment = risk_result["assessment"]
    logger.info(f"✓ Risk level: {assessment['overall_risk']:.2f} ({assessment['severity']})")
    logger.info(f"  Mitigations:")
    for mitigation in assessment['mitigations']:
        logger.info(f"    - {mitigation}")
    logger.info("")
    
    logger.info("Step 5: Resource allocation...")
    compute_result = compute_optimizer.execute({
        "action": "get_resources"
    })
    resources = compute_result["resources"]
    logger.info(f"✓ System resources:")
    logger.info(f"  CPU: {resources.get('cpu_percent', 0):.1f}% used")
    logger.info(f"  Memory: {resources.get('memory_percent', 0):.1f}% used")
    logger.info("")
    
    alloc_result = compute_optimizer.execute({
        "action": "allocate",
        "task_id": f"task_{plan['plan_id']}",
        "requirements": {
            "cpu_cores": 2,
            "memory_gb": 4,
            "priority": "high"
        }
    })
    logger.info(f"✓ Allocated resources for task")
    logger.info("")
    
    logger.info("Step 6: Workflow orchestration...")
    workflow_id = f"workflow_{plan['plan_id']}"
    orchestrator.execute({
        "action": "create_workflow",
        "workflow_id": workflow_id,
        "plan": plan
    })
    
    workflow_result = orchestrator.execute({
        "action": "execute_workflow",
        "workflow_id": workflow_id
    })
    workflow = workflow_result["workflow"]
    logger.info(f"✓ Workflow status: {workflow['status']}")
    logger.info(f"  Completed {len(workflow['results'])} subtasks")
    logger.info("")
    
    logger.info("Step 7: Storing in memory...")
    memory_agent.execute({
        "action": "store",
        "type": "episodic",
        "event": {
            "summary": f"Completed task: {task}",
            "details": f"Plan: {len(plan['subtasks'])} subtasks, Status: {workflow['status']}",
            "session_id": session_id,
            "ethics_score": evaluation['ethical_score'],
            "risk_level": assessment['severity']
        }
    })
    
    memory_agent.execute({
        "action": "store",
        "type": "semantic",
        "concept": "AI_safety_research",
        "knowledge": {
            "task": task,
            "methodology": "Multi-agent orchestration",
            "safety_validated": True
        }
    })
    logger.info("✓ Stored episodic and semantic memories\n")
    
    logger.info("Step 8: Updating session...")
    session_agent.execute({
        "action": "update",
        "session_id": session_id,
        "context_update": {
            "completed_task": task,
            "status": workflow['status'],
            "timestamp": "2025-10-30T17:30:00"
        }
    })
    logger.info("✓ Session updated\n")
    
    logger.info("Step 9: Recalling memories...")
    recall_result = memory_agent.execute({
        "action": "recall",
        "type": "episodic",
        "limit": 5
    })
    logger.info(f"✓ Recalled {len(recall_result['memories'])} episodic memories")
    for i, mem in enumerate(recall_result['memories'][:3], 1):
        logger.info(f"  {i}. {mem['event']['summary']}")
    logger.info("")
    
    logger.info("Step 10: Cleanup...")
    compute_optimizer.execute({
        "action": "release",
        "task_id": f"task_{plan['plan_id']}"
    })
    logger.info("✓ Resources released\n")
    
    logger.info("="*70)
    logger.info("WORKFLOW COMPLETE")
    logger.info("="*70)
    logger.info("\nSummary:")
    logger.info(f"  Session: {session_id}")
    logger.info(f"  Task: {task}")
    logger.info(f"  Plan: {len(plan['subtasks'])} subtasks")
    logger.info(f"  Ethics Score: {evaluation['ethical_score']:.2f}")
    logger.info(f"  Risk Level: {assessment['severity']}")
    logger.info(f"  Workflow Status: {workflow['status']}")
    logger.info(f"  Memories Stored: 2")
    logger.info("="*70)


if __name__ == "__main__":
    demo_full_workflow()
