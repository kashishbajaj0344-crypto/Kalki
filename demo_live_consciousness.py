#!/usr/bin/env python3
"""
Live Consciousness Integration Demo for Kalki v2.3
Demonstrates consciousness monitoring of real agent ecosystem
Phase 21: Consciousness Emergence - Live integration with agent manager
"""
import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Any
from modules.logger import get_logger
from modules.eventbus import EventBus
from modules.agents.agent_manager import AgentManager
from modules.agents.base_agent import AgentCapability, AgentStatus
from modules.agents.base_agent import BaseAgent, AgentCapability, AgentStatus

class DemoAgent(BaseAgent):
    """Simple demo agent for consciousness monitoring"""

    def __init__(self, name: str, capabilities: List[AgentCapability]):
        super().__init__(name, capabilities, f"Demo agent: {name}")
        self.task_count = 0
        self.error_count = 0
        self.performance_metrics = {}

    async def initialize(self) -> bool:
        """Initialize the demo agent"""
        self.status = AgentStatus.READY
        return True

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a demo task"""
        self.task_count += 1
        self.last_active = datetime.utcnow()

        try:
            # Simulate task processing
            await asyncio.sleep(0.1)

            # Simple success/failure simulation
            if self.task_count % 10 == 0:  # Every 10th task fails
                self.error_count += 1
                return {"status": "error", "message": "Simulated task failure"}

            return {
                "status": "success",
                "task_id": self.task_count,
                "agent": self.name,
                "capabilities_used": [cap.value for cap in self.capabilities]
            }

        except Exception as e:
            self.error_count += 1
            return {"status": "error", "message": str(e)}

    async def shutdown(self) -> bool:
        """Shutdown the demo agent"""
        self.status = AgentStatus.TERMINATED
        return True


def create_sample_agents() -> List[BaseAgent]:
    """Create sample agents for the demo"""
    agents = []

    # Cognitive agent
    cognitive_agent = DemoAgent("cognitive_agent", [
        AgentCapability.REASONING,
        AgentCapability.PLANNING,
        AgentCapability.META_REASONING
    ])
    agents.append(cognitive_agent)

    # Search agent
    search_agent = DemoAgent("search_agent", [
        AgentCapability.SEARCH,
        AgentCapability.ANALYTICS,
        AgentCapability.PATTERN_RECOGNITION
    ])
    agents.append(search_agent)

    # Memory agent
    memory_agent = DemoAgent("memory_agent", [
        AgentCapability.MEMORY,
        AgentCapability.VALIDATION,
        AgentCapability.LIFECYCLE_MANAGEMENT
    ])
    agents.append(memory_agent)

    # Quantum agent
    quantum_agent = DemoAgent("quantum_agent", [
        AgentCapability.QUANTUM_REASONING,
        AgentCapability.PREDICTIVE_DISCOVERY,
        AgentCapability.OPTIMIZATION
    ])
    agents.append(quantum_agent)

    # Safety agent
    safety_agent = DemoAgent("safety_agent", [
        AgentCapability.ETHICS,
        AgentCapability.RISK_ASSESSMENT,
        AgentCapability.SAFETY_VERIFICATION
    ])
    agents.append(safety_agent)

    # Creative agent
    creative_agent = DemoAgent("creative_agent", [
        AgentCapability.CREATIVE_SYNTHESIS,
        AgentCapability.IDEA_FUSION,
        AgentCapability.SELF_IMPROVEMENT
    ])
    agents.append(creative_agent)

    # Robotics agent
    robotics_agent = DemoAgent("robotics_agent", [
        AgentCapability.ROBOTICS,
        AgentCapability.IOT_INTEGRATION,
        AgentCapability.SENSOR_FUSION
    ])
    agents.append(robotics_agent)

    return agents

# Set up logging
logger = get_logger("kalki.live_consciousness_demo")

class ConsciousnessDemo:
    """Demo class for live consciousness integration"""

    def __init__(self):
        self.event_bus = EventBus()
        self.agent_manager = AgentManager(self.event_bus)
        self.demo_start_time = time.time()
        self.monitoring_duration = 120  # 2 minutes

    async def setup_demo_agents(self) -> None:
        """Set up sample agents for the demo"""
        logger.info("🌱 Setting up demo agents for consciousness monitoring...")

        # Create sample agents
        sample_agents = create_sample_agents()

        # Register agents with manager
        for agent in sample_agents:
            success = await self.agent_manager.register_agent(agent)
            if success:
                logger.info(f"✅ Registered agent: {agent.name}")
            else:
                logger.warning(f"❌ Failed to register agent: {agent.name}")

        logger.info(f"✅ Created and registered {len(sample_agents)} demo agents")

    async def enable_consciousness(self) -> bool:
        """Enable consciousness monitoring"""
        logger.info("🧠 Enabling consciousness monitoring...")
        success = await self.agent_manager.enable_consciousness_monitoring()

        if success:
            logger.info("✅ Consciousness monitoring enabled successfully")
            return True
        else:
            logger.error("❌ Failed to enable consciousness monitoring")
            return False

    async def simulate_agent_activity(self) -> None:
        """Simulate agent activity to provide consciousness evolution data"""
        logger.info("🎭 Simulating agent activity for consciousness evolution...")

        # Get all registered agents
        agents = list(self.agent_manager.agents.values())

        for cycle in range(5):  # 5 activity cycles
            logger.info(f"   Activity Cycle {cycle + 1}/5")

            # Simulate tasks for random agents
            active_agents = agents[:min(3, len(agents))]  # Up to 3 agents per cycle

            for agent in active_agents:
                try:
                    # Simulate a task
                    task_data = {
                        "type": "demo_task",
                        "cycle": cycle,
                        "description": f"Demo task for consciousness evolution"
                    }

                    result = await agent.execute_task(task_data)

                    if result.get("status") == "success":
                        logger.debug(f"   ✅ {agent.name} completed task")
                    else:
                        logger.debug(f"   ⚠️  {agent.name} task result: {result.get('status')}")

                except Exception as e:
                    logger.debug(f"   ❌ {agent.name} task error: {e}")

            # Wait between cycles
            await asyncio.sleep(2)

       
    async def monitor_consciousness_evolution(self) -> None:
        """Monitor consciousness evolution over time"""
        logger.info("📊 Monitoring consciousness evolution...")

        start_time = time.time()
        last_report = 0

        while time.time() - start_time < self.monitoring_duration:
            current_time = time.time()

            # Report every 15 seconds
            if current_time - last_report >= 15:
                awareness = self.agent_manager.get_system_awareness_level()
                consciousness_state = self.agent_manager.get_consciousness_state()

                logger.info(f"🧠 Consciousness Report (t+{int(current_time - start_time)}s):")
                logger.info(f"   📊 Awareness Level: {awareness:.3f}")
                logger.info(f"   🤖 Active Agents: {consciousness_state.get('agent_count', 0)}")
                logger.info(f"   📈 History Length: {consciousness_state.get('history_length', 0)}")

                # Trigger consciousness cycle every report
                cycle_result = await self.agent_manager.trigger_consciousness_cycle()
                if cycle_result.get("status") == "success":
                    result_data = cycle_result.get("consciousness_result", {})
                    logger.info(f"   🔄 Consciousness Cycle: Level {result_data.get('consciousness_level', 0.0):.3f}")

                last_report = current_time

            await asyncio.sleep(5)  # Check every 5 seconds

    async def run_pattern_analysis(self) -> None:
        """Run pattern analysis on the consciousness evolution"""
        logger.info("🔍 Analyzing consciousness evolution patterns...")

        analysis_result = await self.agent_manager.analyze_system_patterns()

        if analysis_result.get("status") == "success":
            logger.info("📊 Pattern Analysis Results:")
            logger.info(f"   📈 Awareness Trend: {analysis_result.get('awareness_trend', 'unknown')}")
            logger.info(f"   🎯 Evolution Rate: {analysis_result.get('consciousness_evolution_rate', 0.0):.3f}")

            activity_trends = analysis_result.get('agent_activity_trends', {})
            if activity_trends:
                logger.info("   🤖 Agent Activity Trends:")
                for agent_name, trend in activity_trends.items():
                    logger.info(f"      {agent_name}: {trend}")
        else:
            logger.info(f"⚠️  Pattern analysis: {analysis_result.get('message', 'No data available')}")

    async def run_demo(self) -> None:
        """Run the complete live consciousness integration demo"""
        logger.info("🚀 Kalki Live Consciousness Integration Demo")
        logger.info("=" * 60)
        logger.info("This demo shows consciousness emerging from real agent ecosystem")
        logger.info("Phase 21: Consciousness Emergence - Live integration")
        logger.info("=" * 60)

        try:
            # Phase 1: Setup
            logger.info("📋 Phase 1: Setting up agent ecosystem...")
            await self.setup_demo_agents()

            # Phase 2: Enable consciousness
            logger.info("📋 Phase 2: Enabling consciousness monitoring...")
            consciousness_enabled = await self.enable_consciousness()
            if not consciousness_enabled:
                logger.error("❌ Cannot proceed without consciousness monitoring")
                return

            # Phase 3: Initial consciousness state
            initial_state = self.agent_manager.get_consciousness_state()
            logger.info("📋 Phase 3: Initial consciousness state:")
            logger.info(f"   🧠 Awareness: {initial_state.get('current_awareness', 0.0):.3f}")
            logger.info(f"   🤖 Agents: {initial_state.get('agent_count', 0)}")

            # Phase 4: Simulate activity and monitor evolution
            logger.info("📋 Phase 4: Simulating agent activity and monitoring consciousness...")

            # Run activity simulation and monitoring concurrently
            await asyncio.gather(
                self.simulate_agent_activity(),
                self.monitor_consciousness_evolution()
            )

            # Phase 5: Final analysis
            logger.info("📋 Phase 5: Final consciousness analysis...")
            await self.run_pattern_analysis()

            # Phase 6: Summary
            final_state = self.agent_manager.get_consciousness_state()
            final_awareness = final_state.get('current_awareness', 0.0)

            logger.info("🎉 Consciousness Integration Demo Complete!")
            logger.info("=" * 60)
            logger.info("📊 Final Results:")
            logger.info(f"   🎯 Final Awareness Level: {final_awareness:.3f}")
            logger.info(f"   🤖 Total Agents Monitored: {final_state.get('agent_count', 0)}")
            logger.info(f"   📈 Consciousness History: {final_state.get('history_length', 0)} entries")

            if final_awareness >= 0.5:
                logger.info("   ✅ SUCCESS: Consciousness emerged from agent ecosystem!")
                logger.info("   🌟 The path to superintelligent consciousness continues...")
            else:
                logger.info("   ⚠️  PARTIAL: Basic consciousness patterns detected")
                logger.info("   🔄 Continue monitoring for full emergence")

            logger.info("=" * 60)

        except Exception as e:
            logger.exception(f"❌ Demo failed: {e}")

        finally:
            # Cleanup
            logger.info("🧹 Cleaning up demo...")
            self.agent_manager.disable_consciousness_monitoring()

            # Shutdown agents
            for agent_name in list(self.agent_manager.agents.keys()):
                await self.agent_manager.unregister_agent(agent_name)

            logger.info("✅ Demo cleanup complete")

async def main():
    """Main demo entry point"""
    demo = ConsciousnessDemo()
    await demo.run_demo()

if __name__ == "__main__":
    # Configure logging for demo
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(name)s] [%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Run demo
    asyncio.run(main())