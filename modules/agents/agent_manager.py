"""
Agent Manager for Kalki v2.3
Manages agent lifecycle, discovery, orchestration, and dynamic resource allocation
Enhanced for Phase 3: Dynamic resource management, compute optimization, multi-agent coordination
"""
import asyncio
import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from .base_agent import BaseAgent, AgentCapability, AgentStatus
from .consciousness_agent import ConsciousnessAgent
from modules.eventbus import EventBus

# Optional psutil import for resource monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False


@dataclass
class ResourceAllocation:
    """Resource allocation for an agent"""
    cpu_cores: int = 1
    memory_mb: int = 256
    gpu_memory_mb: int = 0
    priority: int = 1  # 1-10, higher = more priority


@dataclass
class SystemResources:
    """Current system resource usage"""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: int = 0
    memory_total_mb: int = 0
    gpu_memory_used_mb: int = 0
    gpu_memory_total_mb: int = 0


class AgentManager:
    """
    Enhanced Central manager for all agents in the Kalki system
    Handles registration, discovery, lifecycle, coordination, and dynamic resource allocation
    Phase 3 enhancements: Resource optimization, multi-agent scheduling, performance monitoring
    """

    def __init__(self, event_bus: Optional[EventBus] = None, max_concurrent_agents: int = 8):
        self.agents: Dict[str, BaseAgent] = {}
        self.capability_index: Dict[AgentCapability, List[str]] = {}
        self.event_bus = event_bus or EventBus()
        self.logger = logging.getLogger("kalki.agentmanager")

        # Resource management
        self.resource_allocations: Dict[str, ResourceAllocation] = {}
        self.max_concurrent_agents = max_concurrent_agents
        self.active_agents: set = set()
        self.resource_monitor_thread: Optional[threading.Thread] = None
        self.monitoring_active = False
        self.system_resources = SystemResources()

        # Consciousness integration
        self.consciousness_agent: Optional[ConsciousnessAgent] = None
        self.consciousness_monitoring_active = False
        self.system_awareness_level = 0.0

        self._lock = asyncio.Lock()

    # ------------------------------
    # Resource Management Methods
    # ------------------------------
    def start_resource_monitoring(self) -> None:
        """Start background resource monitoring"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.resource_monitor_thread = threading.Thread(
            target=self._monitor_resources,
            daemon=True,
            name="ResourceMonitor"
        )
        self.resource_monitor_thread.start()
        self.logger.info("Resource monitoring started")

    def stop_resource_monitoring(self) -> None:
        """Stop background resource monitoring"""
        self.monitoring_active = False
        if self.resource_monitor_thread:
            self.resource_monitor_thread.join(timeout=5.0)
        self.logger.info("Resource monitoring stopped")

    def _monitor_resources(self) -> None:
        """Background thread for monitoring system resources"""
        while self.monitoring_active:
            try:
                self.system_resources = self._get_current_resources()
                # Log warnings if resources are critically low
                if self.system_resources.memory_percent > 90:
                    self.logger.warning(f"High memory usage: {self.system_resources.memory_percent:.1f}%")
                if self.system_resources.cpu_percent > 95:
                    self.logger.warning(f"High CPU usage: {self.system_resources.cpu_percent:.1f}%")

                threading.Event().wait(5.0)  # Monitor every 5 seconds

            except Exception as e:
                self.logger.exception(f"Resource monitoring error: {e}")
                threading.Event().wait(10.0)  # Wait longer on error

    def _get_current_resources(self) -> SystemResources:
        """Get current system resource usage"""
        resources = SystemResources()

        if HAS_PSUTIL:
            try:
                # CPU usage
                resources.cpu_percent = psutil.cpu_percent(interval=1)

                # Memory usage
                memory = psutil.virtual_memory()
                resources.memory_percent = memory.percent
                resources.memory_used_mb = memory.used // (1024 * 1024)
                resources.memory_total_mb = memory.total // (1024 * 1024)

                # GPU memory detection
                try:
                    # Try to detect GPU memory using torch if available
                    import torch
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.get_device_properties(0).total_memory
                        resources.gpu_memory_total_mb = gpu_memory // (1024 * 1024)
                    elif torch.backends.mps.is_available():
                        # For Apple Silicon MPS
                        resources.gpu_memory_total_mb = 0  # MPS doesn't expose total memory easily
                    else:
                        resources.gpu_memory_total_mb = 0
                except ImportError:
                    # torch not available
                    resources.gpu_memory_total_mb = 0
                except Exception as e:
                    self.logger.debug(f"Failed to detect GPU memory: {e}")
                    resources.gpu_memory_total_mb = 0

            except Exception as e:
                self.logger.debug(f"Failed to get system resources: {e}")

        return resources

    def allocate_resources(self, agent_name: str, allocation: ResourceAllocation) -> bool:
        """
        Allocate resources for an agent

        Returns:
            True if allocation successful, False otherwise
        """
        # Check if allocation would exceed system limits
        total_allocated_cpu = sum(a.cpu_cores for a in self.resource_allocations.values())
        total_allocated_memory = sum(a.memory_mb for a in self.resource_allocations.values())

        if total_allocated_cpu + allocation.cpu_cores > psutil.cpu_count() if HAS_PSUTIL else 8:
            self.logger.warning(f"CPU allocation would exceed available cores for {agent_name}")
            return False

        # Estimate memory limit (leave 1GB headroom)
        available_memory_mb = self.system_resources.memory_total_mb - 1024
        if total_allocated_memory + allocation.memory_mb > available_memory_mb:
            self.logger.warning(f"Memory allocation would exceed available RAM for {agent_name}")
            return False

        self.resource_allocations[agent_name] = allocation
        self.logger.info(f"Allocated resources for {agent_name}: CPU={allocation.cpu_cores}, "
                        f"Memory={allocation.memory_mb}MB, Priority={allocation.priority}")
        return True

    def deallocate_resources(self, agent_name: str) -> None:
        """Deallocate resources for an agent"""
        if agent_name in self.resource_allocations:
            del self.resource_allocations[agent_name]
            self.logger.info(f"Deallocated resources for {agent_name}")

    # ------------------------------
    # Consciousness Integration Methods
    # ------------------------------
    async def enable_consciousness_monitoring(self) -> bool:
        """
        Enable consciousness monitoring for the agent ecosystem

        Returns:
            True if successfully enabled, False otherwise
        """
        try:
            if self.consciousness_agent is None:
                self.consciousness_agent = ConsciousnessAgent("system_consciousness", self.event_bus)
                self.consciousness_agent.set_agent_manager(self)

                # Register the consciousness agent
                success = await self.register_agent(self.consciousness_agent)
                if not success:
                    self.logger.error("Failed to register consciousness agent")
                    return False

            if not self.consciousness_monitoring_active:
                self.consciousness_monitoring_active = True
                # Start monitoring in background
                asyncio.create_task(self.consciousness_agent.start_monitoring())
                self.logger.info("Consciousness monitoring enabled")

            return True

        except Exception as e:
            self.logger.exception(f"Failed to enable consciousness monitoring: {e}")
            return False

    def disable_consciousness_monitoring(self) -> None:
        """Disable consciousness monitoring"""
        if self.consciousness_agent:
            self.consciousness_agent.stop_monitoring()
        self.consciousness_monitoring_active = False
        self.logger.info("Consciousness monitoring disabled")

    def get_system_awareness_level(self) -> float:
        """
        Get the current system awareness level from consciousness monitoring

        Returns:
            Current awareness level (0.0 to 1.0)
        """
        if self.consciousness_agent:
            return self.consciousness_agent.current_awareness_level
        return 0.0

    async def trigger_consciousness_cycle(self) -> Dict[str, Any]:
        """
        Manually trigger a consciousness evolution cycle

        Returns:
            Consciousness cycle results
        """
        if not self.consciousness_agent:
            return {"status": "error", "message": "Consciousness monitoring not enabled"}

        return await self.consciousness_agent.execute({"type": "achieve_consciousness"})

    def get_consciousness_state(self) -> Dict[str, Any]:
        """
        Get current consciousness state of the system

        Returns:
            Current consciousness state information
        """
        if not self.consciousness_agent:
            return {"status": "not_enabled", "awareness_level": 0.0}

        return self.consciousness_agent._get_consciousness_state()

    async def analyze_system_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in agent behavior and consciousness evolution

        Returns:
            Pattern analysis results
        """
        if not self.consciousness_agent:
            return {"status": "error", "message": "Consciousness monitoring not enabled"}

        return await self.consciousness_agent.execute({"type": "analyze_patterns"})

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
        Execute a task on a specific agent with resource management

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

        # Check concurrent agent limit
        if len(self.active_agents) >= self.max_concurrent_agents:
            return {
                "status": "error",
                "error": "Maximum concurrent agents reached"
            }

        # Allocate resources if not already allocated
        if agent_name not in self.resource_allocations:
            # Default allocation based on task complexity
            task_complexity = task.get("complexity", "medium")
            default_allocation = self._get_default_allocation(task_complexity)
            if not self.allocate_resources(agent_name, default_allocation):
                return {
                    "status": "error",
                    "error": "Failed to allocate resources for task"
                }

        try:
            # Mark agent as active
            self.active_agents.add(agent_name)
            agent.update_status(AgentStatus.WORKING)
            agent.increment_task_count()

            # Memory Hook: Store task initiation as episodic memory
            await self._store_task_event("task_started", agent_name, task)

            self.logger.info(f"Starting task execution on {agent_name} "
                           f"(active agents: {len(self.active_agents)})")

            result = await agent.execute(task)

            # Memory Hook: Store task completion/result as episodic memory
            await self._store_task_event("task_completed", agent_name, task, result)

            # Memory Hook: Store structured knowledge as semantic memory
            await self._store_task_knowledge(agent_name, task, result)

            agent.update_status(AgentStatus.READY)
            self.active_agents.discard(agent_name)

            return result

        except Exception as e:
            self.logger.exception(f"Error executing task on {agent_name}: {e}")
            
            # Memory Hook: Store task failure as episodic memory
            await self._store_task_event("task_failed", agent_name, task, {"error": str(e)})
            
            agent.increment_error_count()
            agent.update_status(AgentStatus.ERROR)
            self.active_agents.discard(agent_name)
            return {
                "status": "error",
                "error": str(e)
            }

    def _get_default_allocation(self, complexity: str) -> ResourceAllocation:
        """Get default resource allocation based on task complexity"""
        if complexity == "high":
            return ResourceAllocation(cpu_cores=2, memory_mb=1024, priority=8)
        elif complexity == "low":
            return ResourceAllocation(cpu_cores=1, memory_mb=128, priority=3)
        else:  # medium
            return ResourceAllocation(cpu_cores=1, memory_mb=512, priority=5)
    
    async def execute_by_capability(self, 
                                    capability: AgentCapability, 
                                    task: Dict[str, Any],
                                    strategy: str = "optimal") -> Dict[str, Any]:
        """
        Execute a task on an agent with a specific capability using intelligent selection

        Args:
            capability: Required capability
            task: Task dictionary
            strategy: Selection strategy ("optimal", "first", "least_loaded", "round_robin")

        Returns:
            Result dictionary
        """
        # Check if this is a complex task requiring multi-agent coordination
        task_complexity = task.get("complexity", "medium")
        task_description = task.get("description", "")
        
        # Use LLM planning for high complexity tasks or tasks with detailed descriptions
        if (task_complexity == "high" or 
            (task_description and len(task_description.split()) > 20) or
            task.get("requires_coordination", False)):
            
            self.logger.info(f"Using LLM-powered planning for complex task (complexity: {task_complexity})")
            return await self.plan_and_execute_complex_task(task)
        
        if strategy == "optimal":
            # Use intelligent agent selection based on load, resources, and reliability
            agent = self.get_optimal_agent(capability, task_complexity)
            if not agent:
                return {
                    "status": "error",
                    "error": f"No suitable agent found for capability {capability.value}"
                }
            return await self.execute_task(agent.name, task)

        else:
            # Fallback to original strategies
            agents = self.find_agents_by_capability(capability)

            if not agents:
                return {
                    "status": "error",
                    "error": f"No agents found with capability {capability.value}"
                }

            # Filter to ready agents only
            ready_agents = [a for a in agents if a.status == AgentStatus.READY]

            if not ready_agents:
                return {
                    "status": "error",
                    "error": f"No ready agents found with capability {capability.value}"
                }

            # Select agent based on strategy
            if strategy == "first":
                agent = ready_agents[0]
            elif strategy == "least_loaded":
                agent = min(ready_agents, key=lambda a: a.task_count)
            else:  # default to first
                agent = ready_agents[0]

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
        """Get comprehensive system statistics including resource usage"""
        total_tasks = sum(a.task_count for a in self.agents.values())
        total_errors = sum(a.error_count for a in self.agents.values())

        # Calculate resource utilization
        total_allocated_cpu = sum(a.cpu_cores for a in self.resource_allocations.values())
        total_allocated_memory = sum(a.memory_mb for a in self.resource_allocations.values())

        return {
            "total_agents": len(self.agents),
            "active_agents": len(self.active_agents),
            "capabilities": {cap.value: len(agents) for cap, agents in self.capability_index.items()},
            "total_tasks_executed": total_tasks,
            "total_errors": total_errors,
            "error_rate": (total_errors / max(total_tasks, 1)) * 100,

            # Resource information
            "resource_monitoring_active": self.monitoring_active,
            "system_resources": {
                "cpu_percent": self.system_resources.cpu_percent,
                "memory_percent": self.system_resources.memory_percent,
                "memory_used_mb": self.system_resources.memory_used_mb,
                "memory_total_mb": self.system_resources.memory_total_mb,
            },
            "resource_allocations": {
                agent_name: {
                    "cpu_cores": alloc.cpu_cores,
                    "memory_mb": alloc.memory_mb,
                    "gpu_memory_mb": alloc.gpu_memory_mb,
                    "priority": alloc.priority
                }
                for agent_name, alloc in self.resource_allocations.items()
            },
            "total_allocated_cpu": total_allocated_cpu,
            "total_allocated_memory_mb": total_allocated_memory,
            "max_concurrent_agents": self.max_concurrent_agents,

            # Event bus stats
            "event_bus_stats": self.event_bus.get_stats()
        }
    
    async def _store_task_event(self, event_type: str, agent_name: str, task: Dict[str, Any], result: Dict[str, Any] = None):
        """Store task execution event in episodic memory"""
        try:
            # Only store if we have memory agents available
            memory_agents = self.find_agents_by_capability(AgentCapability.MEMORY)
            if not memory_agents:
                return

            event = {
                "type": "task_execution",
                "event_type": event_type,
                "agent_name": agent_name,
                "capability": task.get("action", "unknown"),
                "task_summary": str(task)[:200] + "..." if len(str(task)) > 200 else str(task),
                "timestamp": datetime.utcnow().isoformat(),
                "complexity": task.get("complexity", "medium")
            }

            if result:
                event["result_status"] = result.get("status")
                event["result_summary"] = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)

            memory_task = {
                "action": "store",
                "type": "episodic",
                "event": event
            }

            # Execute on first available memory agent
            await self.execute_task(memory_agents[0].name, memory_task)

        except Exception as e:
            self.logger.debug(f"Failed to store task event in memory: {e}")

    async def _store_task_knowledge(self, agent_name: str, task: Dict[str, Any], result: Dict[str, Any]):
        """Store structured knowledge from task execution in semantic memory"""
        try:
            # Only store successful tasks with meaningful results
            if result.get("status") != "success":
                return

            memory_agents = self.find_agents_by_capability(AgentCapability.MEMORY)
            if not memory_agents:
                return

            # Extract concepts from task and result
            concepts = self._extract_concepts_from_task(task, result)
            
            for concept in concepts:
                knowledge = {
                    "agent": agent_name,
                    "capability": task.get("action", "unknown"),
                    "task_pattern": self._extract_task_pattern(task),
                    "success_indicators": self._extract_success_indicators(result),
                    "last_successful": datetime.utcnow().isoformat(),
                    "usage_count": 1
                }

                memory_task = {
                    "action": "store",
                    "type": "semantic",
                    "concept": concept,
                    "knowledge": knowledge
                }

                await self.execute_task(memory_agents[0].name, memory_task)

        except Exception as e:
            self.logger.debug(f"Failed to store task knowledge in memory: {e}")

    def _extract_concepts_from_task(self, task: Dict[str, Any], result: Dict[str, Any]) -> List[str]:
        """Extract key concepts from task and result for semantic memory"""
        concepts = []
        
        # Extract from task action
        action = task.get("action", "")
        if action:
            concepts.append(f"action_{action}")
        
        # Extract from task parameters
        params = task.get("params", {})
        for key, value in params.items():
            if isinstance(value, str) and len(value) > 3:
                # Simple concept extraction - could be enhanced with NLP
                words = value.lower().split()[:3]  # First 3 words
                concepts.extend(words)
        
        # Extract from result if available
        if result.get("status") == "success":
            concepts.append("successful_execution")
        
        return list(set(concepts))[:5]  # Unique concepts, max 5

    def _extract_task_pattern(self, task: Dict[str, Any]) -> str:
        """Extract task execution pattern"""
        action = task.get("action", "unknown")
        complexity = task.get("complexity", "medium")
        return f"{action}_{complexity}_complexity"

    def _extract_success_indicators(self, result: Dict[str, Any]) -> List[str]:
        """Extract indicators of successful execution"""
        indicators = []
        
        if result.get("status") == "success":
            indicators.append("status_success")
        
        # Add more sophisticated indicators based on result structure
        if "result" in result:
            indicators.append("has_result_data")
        
        if "completed_steps" in result:
            steps = result.get("completed_steps", 0)
            if steps > 0:
                indicators.append(f"completed_{steps}_steps")
        
        return indicators
    
    def get_optimal_agent(self, capability: AgentCapability, task_complexity: str) -> Optional[BaseAgent]:
        """
        Use LLM-powered reasoning to select the optimal agent for a task
        
        Args:
            capability: Required capability
            task_complexity: Task complexity level ("low", "medium", "high")
            
        Returns:
            Optimal agent for the task, or None if no suitable agent found
        """
        try:
            # Get all agents with the required capability
            candidates = self.find_agents_by_capability(capability)
            if not candidates:
                return None
            
            # Filter to ready agents only
            ready_candidates = [a for a in candidates if a.status == AgentStatus.READY]
            if not ready_candidates:
                return None
            
            if len(ready_candidates) == 1:
                return ready_candidates[0]
            
            # Use LLM for intelligent agent selection
            optimal_agent = self._llm_select_optimal_agent(ready_candidates, capability, task_complexity)
            return optimal_agent
            
        except Exception as e:
            self.logger.exception(f"Error in get_optimal_agent: {e}")
            # Fallback to simple selection
            candidates = self.find_agents_by_capability(capability)
            ready_candidates = [a for a in candidates if a.status == AgentStatus.READY]
            return ready_candidates[0] if ready_candidates else None
    
    def _llm_select_optimal_agent(self, candidates: List[BaseAgent], capability: AgentCapability, task_complexity: str) -> Optional[BaseAgent]:
        """
        Use LLM to analyze agent performance and select the optimal one
        
        Args:
            candidates: List of candidate agents
            capability: Required capability
            task_complexity: Task complexity level
            
        Returns:
            Selected optimal agent
        """
        try:
            # Import LLM engine
            from modules.llm import LLMEngine
            
            # Get LLM instance
            llm_engine = LLMEngine()
            
            # Build agent analysis prompt
            agent_summaries = []
            for agent in candidates:
                # Calculate agent metrics
                success_rate = 0.0
                if agent.task_count > 0:
                    success_rate = ((agent.task_count - agent.error_count) / agent.task_count) * 100
                
                avg_response_time = getattr(agent, 'avg_response_time', 0.0)
                
                summary = f"""
Agent: {agent.name}
- Status: {agent.status.value}
- Task Count: {agent.task_count}
- Error Count: {agent.error_count}
- Success Rate: {success_rate:.1f}%
- Average Response Time: {avg_response_time:.2f}s
- Capabilities: {[cap.value for cap in agent.capabilities]}
- Current Load: {'High' if agent.task_count > 10 else 'Medium' if agent.task_count > 5 else 'Low'}
"""
                agent_summaries.append(summary)
            
            # Get system resource context
            system_load = "High" if self.system_resources.cpu_percent > 80 else "Medium" if self.system_resources.cpu_percent > 50 else "Low"
            memory_pressure = "High" if self.system_resources.memory_percent > 80 else "Medium" if self.system_resources.memory_percent > 60 else "Low"
            
            prompt = f"""
You are an intelligent agent coordinator for a complex AI system. You need to select the optimal agent for a task requiring the {capability.value} capability with {task_complexity} complexity.

Available Agents:
{chr(10).join(agent_summaries)}

System Context:
- Current CPU Load: {system_load} ({self.system_resources.cpu_percent:.1f}%)
- Memory Pressure: {memory_pressure} ({self.system_resources.memory_percent:.1f}%)
- Active Agents: {len(self.active_agents)}/{self.max_concurrent_agents}
- Task Complexity: {task_complexity}

Selection Criteria (in priority order):
1. Agent reliability (success rate and error history)
2. Current workload and system resource availability
3. Agent specialization match for the capability
4. Response time performance
5. Task complexity alignment

Analyze the agents and select the single best one for this task. Provide your reasoning and the selected agent name.

Response format:
REASONING: [Your detailed analysis]
SELECTED_AGENT: [agent_name]
"""
            
            # Get LLM response
            response = llm_engine.generate(prompt, max_tokens=500, temperature=0.3)
            
            if not response:
                # Fallback to rule-based selection
                return self._fallback_agent_selection(candidates, task_complexity)
            
            # Parse response to extract selected agent
            response_lines = response.strip().split('\n')
            selected_agent_name = None
            
            for line in response_lines:
                if line.startswith('SELECTED_AGENT:'):
                    selected_agent_name = line.replace('SELECTED_AGENT:', '').strip()
                    break
            
            if selected_agent_name:
                # Find the agent by name
                for agent in candidates:
                    if agent.name == selected_agent_name:
                        self.logger.info(f"LLM selected agent {selected_agent_name} for {capability.value} task (complexity: {task_complexity})")
                        return agent
            
            # If parsing failed, fallback
            self.logger.warning(f"Failed to parse LLM response for agent selection: {response}")
            return self._fallback_agent_selection(candidates, task_complexity)
            
        except Exception as e:
            self.logger.exception(f"Error in LLM agent selection: {e}")
            return self._fallback_agent_selection(candidates, task_complexity)
    
    def _fallback_agent_selection(self, candidates: List[BaseAgent], task_complexity: str) -> Optional[BaseAgent]:
        """
        Fallback rule-based agent selection when LLM fails
        
        Args:
            candidates: List of candidate agents
            task_complexity: Task complexity level
            
        Returns:
            Selected agent using simple heuristics
        """
        if not candidates:
            return None
        
        # Simple selection based on task count (least loaded)
        return min(candidates, key=lambda a: a.task_count)
    
    async def shutdown_all(self):
        """Shutdown all agents and cleanup resources"""
        self.logger.info("Shutting down all agents and cleaning up resources")

        # Stop resource monitoring
        self.stop_resource_monitoring()

        # Clear resource allocations
        self.resource_allocations.clear()
        self.active_agents.clear()

        agent_names = list(self.agents.keys())
        for name in agent_names:
            await self.unregister_agent(name)
        await self.event_bus.clear_history()
        self.logger.info("All agents shutdown complete")

    async def plan_and_execute_complex_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM-powered planning to break down complex tasks and coordinate multiple agents
        
        Args:
            task: Complex task requiring multi-agent coordination
            
        Returns:
            Execution results from coordinated agents
        """
        try:
            # Import LLM engine
            from modules.llm import LLMEngine
            
            llm_engine = LLMEngine()
            
            # Get available agents and their capabilities
            available_agents = {}
            for agent_name, agent in self.agents.items():
                available_agents[agent_name] = {
                    "capabilities": [cap.value for cap in agent.capabilities],
                    "status": agent.status.value,
                    "task_count": agent.task_count
                }
            
            # Build planning prompt
            task_description = task.get("description", str(task))
            task_complexity = task.get("complexity", "high")
            
            planning_prompt = f"""
You are an expert task planner for a multi-agent AI system. You need to break down a complex task into coordinated subtasks that can be executed by specialized agents.

Complex Task: {task_description}
Task Complexity: {task_complexity}

Available Agents and Capabilities:
{chr(10).join([f"- {name}: {info['capabilities']} (Status: {info['status']}, Tasks: {info['task_count']})" for name, info in available_agents.items()])}

Your task is to:
1. Analyze the complex task and identify required subtasks
2. Match subtasks to appropriate agent capabilities
3. Determine the execution order and dependencies
4. Consider parallel execution opportunities
5. Plan for error handling and fallback strategies

Provide a detailed execution plan in the following format:

SUBTASKS:
1. [Subtask description] -> [Agent capability] -> [Agent name if specific]
2. [Subtask description] -> [Agent capability] -> [Agent name if specific]
...

EXECUTION_ORDER:
1. Execute subtask 1
2. Execute subtasks 2,3 in parallel
3. Execute subtask 4 (depends on 2,3)
...

DEPENDENCIES:
- Subtask 4 requires results from subtasks 2 and 3
- Subtask 3 requires subtask 1 completion
...

FALLBACK_STRATEGIES:
- If [agent capability] unavailable, use [alternative approach]
...
"""
            
            # Get LLM planning response
            plan_response = llm_engine.generate(planning_prompt, max_tokens=1000, temperature=0.2)
            
            if not plan_response:
                # Fallback to simple execution
                return await self._execute_simple_task(task)
            
            # Parse the plan and execute
            execution_plan = self._parse_execution_plan(plan_response)
            
            # Execute the plan
            return await self._execute_coordinated_plan(execution_plan, task)
            
        except Exception as e:
            self.logger.exception(f"Error in complex task planning: {e}")
            # Fallback to simple execution
            return await self._execute_simple_task(task)
    
    def _parse_execution_plan(self, plan_response: str) -> Dict[str, Any]:
        """
        Parse LLM-generated execution plan into structured format
        
        Args:
            plan_response: Raw LLM response with execution plan
            
        Returns:
            Structured execution plan
        """
        plan = {
            "subtasks": [],
            "execution_order": [],
            "dependencies": [],
            "fallback_strategies": []
        }
        
        current_section = None
        lines = plan_response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('SUBTASKS:'):
                current_section = 'subtasks'
                continue
            elif line.startswith('EXECUTION_ORDER:'):
                current_section = 'execution_order'
                continue
            elif line.startswith('DEPENDENCIES:'):
                current_section = 'dependencies'
                continue
            elif line.startswith('FALLBACK_STRATEGIES:'):
                current_section = 'fallback_strategies'
                continue
            
            if current_section and line.startswith('- ') or line[0].isdigit():
                content = line.lstrip('- ').lstrip('0123456789. ')
                if current_section == 'subtasks':
                    plan['subtasks'].append(content)
                elif current_section == 'execution_order':
                    plan['execution_order'].append(content)
                elif current_section == 'dependencies':
                    plan['dependencies'].append(content)
                elif current_section == 'fallback_strategies':
                    plan['fallback_strategies'].append(content)
        
        return plan
    
    async def _execute_coordinated_plan(self, plan: Dict[str, Any], original_task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a coordinated multi-agent plan
        
        Args:
            plan: Structured execution plan
            original_task: Original complex task
            
        Returns:
            Combined execution results
        """
        results = {
            "status": "success",
            "subtask_results": [],
            "coordination_notes": []
        }
        
        # Simple execution for now - execute subtasks in order
        # This could be enhanced with parallel execution and dependency management
        
        for subtask_desc in plan.get('subtasks', []):
            try:
                # Parse subtask to extract capability and agent
                if '->' in subtask_desc:
                    parts = subtask_desc.split('->')
                    if len(parts) >= 2:
                        task_description = parts[0].strip()
                        capability_name = parts[1].strip()
                        
                        # Find capability enum
                        capability = None
                        for cap in AgentCapability:
                            if cap.value == capability_name:
                                capability = cap
                                break
                        
                        if capability:
                            # Execute subtask
                            subtask = {
                                "action": task_description,
                                "complexity": original_task.get("complexity", "medium"),
                                "description": f"Subtask of complex task: {task_description}"
                            }
                            
                            subtask_result = await self.execute_by_capability(capability, subtask)
                            results["subtask_results"].append({
                                "subtask": task_description,
                                "result": subtask_result
                            })
                            
                            if subtask_result.get("status") != "success":
                                results["status"] = "partial_success"
                                results["coordination_notes"].append(f"Subtask failed: {task_description}")
                        else:
                            results["coordination_notes"].append(f"Unknown capability: {capability_name}")
                    else:
                        results["coordination_notes"].append(f"Malformed subtask: {subtask_desc}")
                else:
                    results["coordination_notes"].append(f"Could not parse subtask: {subtask_desc}")
                    
            except Exception as e:
                self.logger.exception(f"Error executing subtask {subtask_desc}: {e}")
                results["subtask_results"].append({
                    "subtask": subtask_desc,
                    "result": {"status": "error", "error": str(e)}
                })
                results["status"] = "partial_success"
        
        return results
    
    async def _execute_simple_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback simple task execution when planning fails
        
        Args:
            task: Task to execute
            
        Returns:
            Execution result
        """
        # Try to find an appropriate capability and execute
        action = task.get("action", "unknown")
        
        # Map common actions to capabilities
        capability_map = {
            "search": AgentCapability.SEARCH,
            "ingest": AgentCapability.DOCUMENT_INGESTION,
            "plan": AgentCapability.PLANNING,
            "reason": AgentCapability.REASONING,
            "analyze": AgentCapability.ANALYTICS,
            "store": AgentCapability.MEMORY
        }
        
        capability = capability_map.get(action, AgentCapability.REASONING)
        return await self.execute_by_capability(capability, task)

    async def analyze_agent_performance_and_adapt(self) -> Dict[str, Any]:
        """
        Use LLM to analyze agent performance patterns and suggest adaptations
        
        Returns:
            Performance analysis and adaptation recommendations
        """
        try:
            from modules.llm import LLMEngine
            
            llm_engine = LLMEngine()
            
            # Collect agent performance data
            agent_performance = {}
            for agent_name, agent in self.agents.items():
                success_rate = 0.0
                if agent.task_count > 0:
                    success_rate = ((agent.task_count - agent.error_count) / agent.task_count) * 100
                
                agent_performance[agent_name] = {
                    "task_count": agent.task_count,
                    "error_count": agent.error_count,
                    "success_rate": success_rate,
                    "capabilities": [cap.value for cap in agent.capabilities],
                    "status": agent.status.value
                }
            
            # Get recent task history from memory if available
            recent_tasks = []
            try:
                memory_agents = self.find_agents_by_capability(AgentCapability.MEMORY)
                if memory_agents:
                    # Query recent task executions
                    memory_query = {
                        "action": "query",
                        "type": "episodic",
                        "query": "recent task executions",
                        "limit": 20
                    }
                    memory_result = await self.execute_task(memory_agents[0].name, memory_query)
                    if memory_result.get("status") == "success":
                        recent_tasks = memory_result.get("results", [])
            except Exception as e:
                self.logger.debug(f"Could not retrieve task history: {e}")
            
            analysis_prompt = f"""
You are a performance analyst for a multi-agent AI system. Analyze the current agent performance data and recent task history to identify patterns, bottlenecks, and opportunities for improvement.

Agent Performance Data:
{chr(10).join([f"- {name}: {data['task_count']} tasks, {data['error_count']} errors, {data['success_rate']:.1f}% success rate, capabilities: {data['capabilities']}" for name, data in agent_performance.items()])}

Recent Task History:
{chr(10).join([f"- {task.get('capability', 'unknown')} task by {task.get('agent_name', 'unknown')}: {task.get('result_status', 'unknown')}" for task in recent_tasks[:10]])}

Analyze the following:
1. Which agents are performing well and why?
2. Which agents need improvement and what specific issues they have?
3. Are there capability gaps or imbalances in the agent ecosystem?
4. What patterns do you see in task success/failure?
5. Recommendations for agent training, resource allocation, or system improvements

Provide specific, actionable recommendations for improving the overall system performance.

Response format:
PERFORMANCE_ANALYSIS: [Your detailed analysis]
RECOMMENDATIONS: [Specific actionable recommendations]
ADAPTATION_PLAN: [Step-by-step plan for implementing improvements]
"""
            
            analysis_response = llm_engine.generate(analysis_prompt, max_tokens=800, temperature=0.3)
            
            if analysis_response:
                # Store analysis results for future reference
                await self._store_performance_analysis(analysis_response)
                
                return {
                    "status": "success",
                    "analysis": analysis_response,
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to generate performance analysis"
                }
                
        except Exception as e:
            self.logger.exception(f"Error in performance analysis: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def _store_performance_analysis(self, analysis: str) -> None:
        """
        Store performance analysis results in semantic memory
        
        Args:
            analysis: LLM-generated performance analysis
        """
        try:
            memory_agents = self.find_agents_by_capability(AgentCapability.MEMORY)
            if not memory_agents:
                return
            
            # Store as semantic knowledge
            knowledge = {
                "analysis_type": "agent_performance",
                "timestamp": datetime.utcnow().isoformat(),
                "insights": analysis[:500],  # Truncate for storage
                "recommendations_extracted": self._extract_recommendations(analysis)
            }
            
            memory_task = {
                "action": "store",
                "type": "semantic",
                "concept": "agent_performance_analysis",
                "knowledge": knowledge
            }
            
            await self.execute_task(memory_agents[0].name, memory_task)
            
        except Exception as e:
            self.logger.debug(f"Failed to store performance analysis: {e}")
    
    def _extract_recommendations(self, analysis: str) -> List[str]:
        """
        Extract actionable recommendations from analysis text
        
        Args:
            analysis: Full analysis text
            
        Returns:
            List of extracted recommendations
        """
        recommendations = []
        lines = analysis.split('\n')
        
        in_recommendations = False
        for line in lines:
            if 'RECOMMENDATIONS:' in line.upper():
                in_recommendations = True
                continue
            elif in_recommendations and line.strip().startswith('-'):
                recommendations.append(line.strip()[1:].strip())
            elif in_recommendations and line.strip() and not line.startswith(' '):
                # End of recommendations section
                break
        
        return recommendations[:5]  # Limit to top 5