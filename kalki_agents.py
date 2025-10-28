"""
Kalki v2.3 Agent System CLI
Command-line interface for the agent framework
"""
import asyncio
import logging
import sys
from typing import Dict, Any
from agents import AgentManager, EventBus
from agents.base_agent import AgentCapability
from agents.core import (
    DocumentIngestAgent,
    SearchAgent,
    PlannerAgent,
    ReasoningAgent,
    MemoryAgent
)
from agents.cognitive import (
    MetaHypothesisAgent,
    CreativeAgent,
    FeedbackAgent,
    OptimizationAgent
)
from agents.safety import (
    EthicsAgent,
    RiskAssessmentAgent,
    SimulationVerifierAgent
)
from agents.multimodal import (
    VisionAgent,
    AudioAgent,
    SensorFusionAgent,
    ARInsightAgent
)

logger = logging.getLogger("kalki.agent_cli")


class KalkiAgentSystem:
    """Main Kalki Agent System"""
    
    def __init__(self):
        self.event_bus = EventBus()
        self.agent_manager = AgentManager(self.event_bus)
        self.running = False
    
    async def initialize(self):
        """Initialize all agents"""
        logger.info("Initializing Kalki Agent System v2.3...")
        
        # Phase 1-5: Core agents
        agents_to_register = [
            DocumentIngestAgent(),
            SearchAgent(),
            PlannerAgent(),
            ReasoningAgent(),
            MemoryAgent(),
            
            # Phase 6, 10, 11: Cognitive agents
            MetaHypothesisAgent(),
            CreativeAgent(),
            FeedbackAgent(),
            OptimizationAgent(),
            
            # Phase 12: Safety agents
            EthicsAgent(),
            RiskAssessmentAgent(),
            SimulationVerifierAgent(),
            
            # Phase 13, 17: Multi-modal agents
            VisionAgent(),
            AudioAgent(),
            SensorFusionAgent(),
            ARInsightAgent()
        ]
        
        # Register all agents
        for agent in agents_to_register:
            success = await self.agent_manager.register_agent(agent)
            if success:
                logger.info(f"✓ Registered: {agent.name}")
            else:
                logger.error(f"✗ Failed to register: {agent.name}")
        
        self.running = True
        logger.info(f"\n🚀 Kalki Agent System initialized with {len(self.agent_manager.agents)} agents")
        
    async def shutdown(self):
        """Shutdown all agents"""
        logger.info("Shutting down Kalki Agent System...")
        await self.agent_manager.shutdown_all()
        self.running = False
        logger.info("Shutdown complete")
    
    async def show_status(self):
        """Show system status"""
        print("\n" + "="*60)
        print("KALKI v2.3 AGENT SYSTEM STATUS")
        print("="*60)
        
        stats = self.agent_manager.get_system_stats()
        print(f"\nTotal Agents: {stats['total_agents']}")
        print(f"Total Tasks Executed: {stats['total_tasks_executed']}")
        print(f"Total Errors: {stats['total_errors']}")
        
        print("\nCapabilities:")
        for cap, count in stats['capabilities'].items():
            print(f"  - {cap}: {count} agent(s)")
        
        print("\nAgent Health:")
        health = await self.agent_manager.health_check_all()
        for agent_name, status in health.items():
            emoji = "✓" if status['status'] == 'ready' else "✗"
            print(f"  {emoji} {agent_name}: {status['status']} (tasks: {status['task_count']}, errors: {status['error_count']})")
        
        print("\nEvent Bus:")
        eb_stats = stats['event_bus_stats']
        print(f"  - Subscribers: {eb_stats['total_subscribers']}")
        print(f"  - Event Types: {len(eb_stats['event_types'])}")
        print(f"  - History Size: {eb_stats['history_size']}")
        print("="*60 + "\n")
    
    async def execute_interactive(self):
        """Run interactive mode"""
        print("\n" + "="*60)
        print("KALKI v2.3 - INTERACTIVE AGENT MODE")
        print("="*60)
        print("\nCommands:")
        print("  status - Show system status")
        print("  list - List all agents")
        print("  search <query> - Search knowledge base")
        print("  reason <query> - Perform reasoning")
        print("  plan <goal> - Create a plan")
        print("  ideate <topic> - Generate creative ideas")
        print("  ethics <action> - Review ethics")
        print("  help - Show this help")
        print("  quit - Exit system")
        print("="*60 + "\n")
        
        while self.running:
            try:
                user_input = input("kalki> ").strip()
                
                if not user_input:
                    continue
                
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                if command == "quit" or command == "exit":
                    break
                elif command == "status":
                    await self.show_status()
                elif command == "list":
                    await self._list_agents()
                elif command == "search":
                    await self._search(args)
                elif command == "reason":
                    await self._reason(args)
                elif command == "plan":
                    await self._plan(args)
                elif command == "ideate":
                    await self._ideate(args)
                elif command == "ethics":
                    await self._ethics_review(args)
                elif command == "help":
                    await self.show_status()
                else:
                    print(f"Unknown command: {command}. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\nInterrupted by user")
                break
            except Exception as e:
                logger.exception(f"Error in interactive mode: {e}")
                print(f"Error: {e}")
    
    async def _list_agents(self):
        """List all agents"""
        print("\nRegistered Agents:")
        for name, agent in self.agent_manager.agents.items():
            caps = ", ".join([c.value for c in agent.capabilities])
            print(f"  - {name}: {agent.description}")
            print(f"    Capabilities: {caps}")
            print(f"    Status: {agent.status.value}, Tasks: {agent.task_count}")
    
    async def _search(self, query: str):
        """Execute search"""
        if not query:
            print("Please provide a search query")
            return
        
        print(f"\nSearching for: {query}")
        result = await self.agent_manager.execute_by_capability(
            AgentCapability.SEARCH,
            {
                "action": "search",
                "params": {"query": query, "top_k": 3}
            }
        )
        
        if result.get("status") == "success":
            print(f"\nFound {result.get('count', 0)} results:")
            for i, chunk in enumerate(result.get('results', []), 1):
                print(f"\n  Result {i}:")
                print(f"    {chunk.get('chunk', '')[:200]}...")
        else:
            print(f"Error: {result.get('error')}")
    
    async def _reason(self, query: str):
        """Execute reasoning"""
        if not query:
            print("Please provide a query to reason about")
            return
        
        print(f"\nReasoning about: {query}")
        result = await self.agent_manager.execute_by_capability(
            AgentCapability.REASONING,
            {
                "action": "reason",
                "params": {"query": query, "steps": 2}
            }
        )
        
        if result.get("status") == "success":
            print(f"\nAnswer: {result.get('answer', '')}")
        else:
            print(f"Error: {result.get('error')}")
    
    async def _plan(self, goal: str):
        """Create a plan"""
        if not goal:
            print("Please provide a goal")
            return
        
        print(f"\nCreating plan for: {goal}")
        result = await self.agent_manager.execute_by_capability(
            AgentCapability.PLANNING,
            {
                "action": "plan",
                "params": {"goal": goal}
            }
        )
        
        if result.get("status") == "success":
            print("\nPlan steps:")
            for step in result.get('plan', []):
                print(f"  {step['step']}. {step['description']}")
        else:
            print(f"Error: {result.get('error')}")
    
    async def _ideate(self, topic: str):
        """Generate ideas"""
        if not topic:
            print("Please provide a topic")
            return
        
        print(f"\nGenerating ideas for: {topic}")
        result = await self.agent_manager.execute_by_capability(
            AgentCapability.CREATIVE_SYNTHESIS,
            {
                "action": "ideate",
                "params": {"topic": topic, "count": 5}
            }
        )
        
        if result.get("status") == "success":
            print(f"\nNovelty Score: {result.get('novelty_score', 0):.2f}")
            print("\nIdeas:")
            for idea in result.get('ideas', []):
                print(f"  - {idea}")
        else:
            print(f"Error: {result.get('error')}")
    
    async def _ethics_review(self, action_desc: str):
        """Review ethics"""
        if not action_desc:
            print("Please provide an action to review")
            return
        
        print(f"\nReviewing ethics for: {action_desc}")
        result = await self.agent_manager.execute_by_capability(
            AgentCapability.ETHICS,
            {
                "action": "review",
                "params": {"action_description": action_desc}
            }
        )
        
        if result.get("status") == "success":
            print(f"\nEthical: {result.get('is_ethical', False)}")
            if result.get('violations'):
                print("Violations:")
                for v in result['violations']:
                    print(f"  - {v}")
            if result.get('concerns'):
                print("Concerns:")
                for c in result['concerns']:
                    print(f"  - {c}")
        else:
            print(f"Error: {result.get('error')}")


async def main():
    """Main entry point"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler('kalki_agents.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    system = KalkiAgentSystem()
    
    try:
        await system.initialize()
        await system.show_status()
        await system.execute_interactive()
    except Exception as e:
        logger.exception(f"System error: {e}")
    finally:
        await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
