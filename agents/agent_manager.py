"""
Agent Manager for Kalki v2.3
Manages agent lifecycle, discovery, and orchestration
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any
from .base_agent import BaseAgent, AgentCapability, AgentStatus
from .event_bus import EventBus


class AgentManager:
    """
    Central manager for all agents in the Kalki system
    Handles registration, discovery, lifecycle, and coordination
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        self.agents: Dict[str, BaseAgent] = {}
        self.capability_index: Dict[AgentCapability, List[str]] = {}
        self.event_bus = event_bus or EventBus()
        self.logger = logging.getLogger("kalki.agentmanager")
        self._lock = asyncio.Lock()
    
    async def register_agent(self, agent: BaseAgent) -> bool:
        """
        Register an agent with the manager
        
        Args:
            agent: Agent instance to register
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with self._lock:
                if agent.name in self.agents:
                    self.logger.warning(f"Agent {agent.name} already registered")
                    return False
                
                # Set event bus
                agent.set_event_bus(self.event_bus)
                
                # Initialize agent
                self.logger.info(f"Initializing agent {agent.name}")
                success = await agent.initialize()
                
                if not success:
                    self.logger.error(f"Failed to initialize agent {agent.name}")
                    return False
                
                # Register agent
                self.agents[agent.name] = agent
                
                # Index by capabilities
                for capability in agent.capabilities:
                    if capability not in self.capability_index:
                        self.capability_index[capability] = []
                    self.capability_index[capability].append(agent.name)
                
                agent.update_status(AgentStatus.READY)
                
                # Emit registration event
                await self.event_bus.publish("agent.registered", {
                    "agent_name": agent.name,
                    "capabilities": [c.value for c in agent.capabilities]
                })
                
                self.logger.info(f"Agent {agent.name} registered successfully")
                return True
                
        except Exception as e:
            self.logger.exception(f"Error registering agent {agent.name}: {e}")
            return False
    
    async def unregister_agent(self, agent_name: str) -> bool:
        """
        Unregister and shutdown an agent
        
        Args:
            agent_name: Name of agent to unregister
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with self._lock:
                if agent_name not in self.agents:
                    self.logger.warning(f"Agent {agent_name} not found")
                    return False
                
                agent = self.agents[agent_name]
                
                # Shutdown agent
                await agent.shutdown()
                
                # Remove from capability index
                for capability in agent.capabilities:
                    if capability in self.capability_index:
                        if agent_name in self.capability_index[capability]:
                            self.capability_index[capability].remove(agent_name)
                
                # Remove from registry
                del self.agents[agent_name]
                
                # Emit unregistration event
                await self.event_bus.publish("agent.unregistered", {
                    "agent_name": agent_name
                })
                
                self.logger.info(f"Agent {agent_name} unregistered")
                return True
                
        except Exception as e:
            self.logger.exception(f"Error unregistering agent {agent_name}: {e}")
            return False
    
    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """Get agent by name"""
        return self.agents.get(agent_name)
    
    def find_agents_by_capability(self, capability: AgentCapability) -> List[BaseAgent]:
        """
        Find all agents with a specific capability
        
        Args:
            capability: Capability to search for
            
        Returns:
            List of agents with the capability
        """
        agent_names = self.capability_index.get(capability, [])
        return [self.agents[name] for name in agent_names if name in self.agents]
    
    async def execute_task(self, agent_name: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task on a specific agent
        
        Args:
            agent_name: Name of agent to execute task
            task: Task dictionary
            
        Returns:
            Result dictionary
        """
        agent = self.get_agent(agent_name)
        if not agent:
            return {
                "status": "error",
                "error": f"Agent {agent_name} not found"
            }
        
        if agent.status != AgentStatus.READY:
            return {
                "status": "error",
                "error": f"Agent {agent_name} not ready (status: {agent.status.value})"
            }
        
        try:
            agent.update_status(AgentStatus.WORKING)
            agent.increment_task_count()
            
            result = await agent.execute(task)
            
            agent.update_status(AgentStatus.READY)
            return result
            
        except Exception as e:
            self.logger.exception(f"Error executing task on {agent_name}: {e}")
            agent.increment_error_count()
            agent.update_status(AgentStatus.ERROR)
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def execute_by_capability(self, 
                                    capability: AgentCapability, 
                                    task: Dict[str, Any],
                                    strategy: str = "first") -> Dict[str, Any]:
        """
        Execute a task on an agent with a specific capability
        
        Args:
            capability: Required capability
            task: Task dictionary
            strategy: Selection strategy ("first", "least_loaded", "round_robin")
            
        Returns:
            Result dictionary
        """
        agents = self.find_agents_by_capability(capability)
        
        if not agents:
            return {
                "status": "error",
                "error": f"No agents found with capability {capability.value}"
            }
        
        # Select agent based on strategy
        if strategy == "first":
            agent = agents[0]
        elif strategy == "least_loaded":
            agent = min(agents, key=lambda a: a.task_count)
        else:  # default to first
            agent = agents[0]
        
        return await self.execute_task(agent.name, task)
    
    async def health_check_all(self) -> Dict[str, Any]:
        """
        Run health check on all agents
        
        Returns:
            Dictionary of health status for all agents
        """
        results = {}
        for name, agent in self.agents.items():
            results[name] = await agent.health_check()
        return results
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        total_tasks = sum(a.task_count for a in self.agents.values())
        total_errors = sum(a.error_count for a in self.agents.values())
        
        return {
            "total_agents": len(self.agents),
            "capabilities": {cap.value: len(agents) for cap, agents in self.capability_index.items()},
            "total_tasks_executed": total_tasks,
            "total_errors": total_errors,
            "event_bus_stats": self.event_bus.get_stats()
        }
    
    async def shutdown_all(self):
        """Shutdown all agents"""
        self.logger.info("Shutting down all agents")
        agent_names = list(self.agents.keys())
        for name in agent_names:
            await self.unregister_agent(name)
        await self.event_bus.clear_history()
        self.logger.info("All agents shutdown complete")
