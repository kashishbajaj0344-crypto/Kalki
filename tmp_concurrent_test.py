import asyncio
import logging
from random import choice

from agents.agent_manager import AgentManager
from agents.safety_ethics import EthicsAgent, RiskAssessmentAgent, SimulationVerifierAgent
from agents.base_agent import AgentCapability

# Configure logging
logging.basicConfig(level=logging.INFO)

async def test_concurrent_tasks():
    manager = AgentManager()

    # Create and register agents
    ethics_agent = EthicsAgent()
    risk_agent = RiskAssessmentAgent()
    sim_agent = SimulationVerifierAgent()

    await manager.register_agent(ethics_agent)
    await manager.register_agent(risk_agent)
    await manager.register_agent(sim_agent)

    # Define multiple tasks
    tasks = [
        {"agent": "EthicsAgent", "task": {"action": "review", "params": {"action_description": desc}}}
        for desc in [
            "Collect user private data",
            "Launch marketing campaign",
            "Automated moderation on forum"
        ]
    ] + [
        {"agent": "RiskAssessmentAgent", "task": {"action": "assess", "params": {"scenario": scenario, "factors": ["network", "load"]}}}
        for scenario in ["Server outage", "Data breach", "Service spike"]
    ] + [
        {"agent": "SimulationVerifierAgent", "task": {"action": "verify_simulation", "params": {"simulation_data": {"scenario": scenario}, "expected_outcomes": []}}}
        for scenario in ["Optimization test", "AI experiment", "Algorithm stress test"]
    ]

    # Run all tasks concurrently
    async def run_task(task_info):
        agent_name = task_info["agent"]
        task = task_info["task"]
        result = await manager.execute_task(agent_name, task)
        print(f"[{agent_name}] result:", result)

    await asyncio.gather(*(run_task(t) for t in tasks))

    # Stats after concurrent execution
    stats = manager.get_system_stats()
    print("System stats after concurrent tasks:", stats)

    # Shutdown
    await manager.shutdown_all()
    print("All agents shutdown successfully after concurrent test")

# Run the test
asyncio.run(test_concurrent_tasks())
