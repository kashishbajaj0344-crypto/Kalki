import asyncio
import logging
from modules.agents.agent_manager import AgentManager
from modules.eventbus import EventBus
from modules.agents.base_agent import AgentCapability

# Import example agents
from modules.agents.safety import (
    EthicsAgent,
    RiskAssessmentAgent,
    SimulationVerifierAgent
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kalki.test")

async def main():
    # 1️⃣ Create EventBus and AgentManager
    event_bus = EventBus()
    agent_manager = AgentManager(event_bus=event_bus)
    
    # Optional: subscribe to agent.registered events
    def on_agent_registered(event):
        logger.info(f"EVENT: Agent registered -> {event}")

    # subscribe is synchronous in this EventBus implementation
    event_bus.subscribe("agent.registered", on_agent_registered)

    # 2️⃣ Instantiate agents
    ethics_agent = EthicsAgent()
    risk_agent = RiskAssessmentAgent()
    sim_agent = SimulationVerifierAgent()

    # 3️⃣ Register agents
    for agent in [ethics_agent, risk_agent, sim_agent]:
        success = await agent_manager.register_agent(agent)
        logger.info(f"Registered {agent.name}: {success}")

    # 4️⃣ Execute a task by agent name
    task_ethics = {
        "action": "review",
        "params": {
            "action_description": "Access user personal data to optimize AI recommendations",
            "context": {}
        }
    }
    result = await agent_manager.execute_task("EthicsAgent", task_ethics)
    logger.info(f"EthicsAgent task result: {result}")

    # 5️⃣ Execute a task by capability
    task_risk = {
        "action": "assess",
        "params": {
            "scenario": "Deploy new AI feature to production",
            "factors": ["performance", "user data privacy", "scalability"]
        }
    }
    result = await agent_manager.execute_by_capability(AgentCapability.RISK_ASSESSMENT, task_risk)
    logger.info(f"RiskAssessmentAgent task result: {result}")

    # 6️⃣ Execute SimulationVerifierAgent task
    sim_task = {
        "action": "verify_experiment",
        "params": {
            "experiment_description": "Run sandboxed AI learning experiment"
        }
    }
    result = await agent_manager.execute_task("SimulationVerifierAgent", sim_task)
    logger.info(f"SimulationVerifierAgent task result: {result}")

    # 7️⃣ Health check
    health = await agent_manager.health_check_all()
    logger.info(f"Agents health check: {health}")

    # 8️⃣ System stats
    stats = agent_manager.get_system_stats()
    logger.info(f"System stats: {stats}")

    # 9️⃣ Shutdown all agents
    await agent_manager.shutdown_all()
    logger.info("All agents shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())
import asyncio
import logging
from modules.agents.agent_manager import AgentManager
from modules.eventbus import EventBus
from modules.agents.safety import (
    EthicsAgent,
    RiskAssessmentAgent,
    SimulationVerifierAgent
)
from modules.agents.base_agent import AgentCapability

# Configure logging for visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kalki.test")

async def main():
    # 1️⃣ Create EventBus and AgentManager
    event_bus = EventBus()
    agent_manager = AgentManager(event_bus=event_bus)
    
    # Optional: subscribe to all agent.registered events
    def on_agent_registered(event):
        logger.info(f"EVENT: Agent registered -> {event}")
    
    # EventBus.subscribe is synchronous in this repo
    event_bus.subscribe("agent.registered", on_agent_registered)
    
    # 2️⃣ Instantiate agents
    ethics_agent = EthicsAgent()
    risk_agent = RiskAssessmentAgent()
    sim_agent = SimulationVerifierAgent()
    
    # 3️⃣ Register agents
    for agent in [ethics_agent, risk_agent, sim_agent]:
        await agent_manager.register_agent(agent)
    
    # 4️⃣ Execute a task by agent name
    task_ethics = {
        "action": "review",
        "params": {
            "action_description": "Access user personal data to optimize AI recommendations",
            "context": {}
        }
    }
    
    result = await agent_manager.execute_task("EthicsAgent", task_ethics)
    logger.info(f"EthicsAgent task result: {result}")
    
    # 5️⃣ Execute a task by capability
    task_risk = {
        "action": "assess",
        "params": {
            "scenario": "Deploy new AI feature to production",
            "factors": ["performance", "user data privacy", "scalability"]
        }
    }
    
    result = await agent_manager.execute_by_capability(AgentCapability.RISK_ASSESSMENT, task_risk)
    logger.info(f"RiskAssessmentAgent task result: {result}")
    
    # 6️⃣ Execute SimulationVerifierAgent task
    sim_task = {
        "action": "verify_experiment",
        "params": {
            "experiment_description": "Run sandboxed AI learning experiment"
        }
    }
    
    result = await agent_manager.execute_task("SimulationVerifierAgent", sim_task)
    logger.info(f"SimulationVerifierAgent task result: {result}")
    
    # 7️⃣ Check health of all agents
    health = await agent_manager.health_check_all()
    logger.info(f"Agents health check: {health}")
    
    # 8️⃣ Print system stats
    stats = agent_manager.get_system_stats()
    logger.info(f"System stats: {stats}")
    
    # 9️⃣ Shutdown all agents
    await agent_manager.shutdown_all()

if __name__ == "__main__":
    asyncio.run(main())
