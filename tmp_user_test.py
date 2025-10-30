import asyncio
import logging

from agents.agent_manager import AgentManager
from agents.safety_ethics import EthicsAgent, RiskAssessmentAgent, SimulationVerifierAgent
from agents.base_agent import AgentCapability

# Optional: configure logging for readability
logging.basicConfig(level=logging.INFO)

async def test_agent_manager():
    # 1. Create AgentManager
    manager = AgentManager()

    # 2. Create agents
    ethics_agent = EthicsAgent()
    risk_agent = RiskAssessmentAgent()
    sim_agent = SimulationVerifierAgent()

    # 3. Register agents
    await manager.register_agent(ethics_agent)
    await manager.register_agent(risk_agent)
    await manager.register_agent(sim_agent)

    # 4. Execute tasks by agent name
    ethics_result = await manager.execute_task("EthicsAgent", {
        "action": "review",
        "params": {"action_description": "Collecting user personal data for analytics"}
    })
    print("EthicsAgent result:", ethics_result)

    risk_result = await manager.execute_task("RiskAssessmentAgent", {
        "action": "assess",
        "params": {"scenario": "Server outage", "factors": ["load", "network"]}
    })
    print("RiskAssessmentAgent result:", risk_result)

    sim_result = await manager.execute_task("SimulationVerifierAgent", {
        "action": "verify_simulation",
        "params": {"simulation_data": {"scenario": "AI optimization test"}, "expected_outcomes": []}
    })
    print("SimulationVerifierAgent result:", sim_result)

    # 5. Execute task by capability (auto-selected)
    capability_result = await manager.execute_by_capability(
        capability=AgentCapability.ETHICS,
        task={"action": "review", "params": {"action_description": "Automated content moderation"}}
    )
    print("Capability-based execution (ETHICS):", capability_result)

    # 6. System stats
    stats = manager.get_system_stats()
    print("System stats:", stats)

    # 7. Shutdown all
    await manager.shutdown_all()
    print("All agents shutdown successfully")

# Run the test
asyncio.run(test_agent_manager())
