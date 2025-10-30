"""
Phase 17 - Multi-Agent Coordination Layer
Agent execution, registry, and messaging bus for inter-agent communication.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Callable
from datetime import datetime
from enum import Enum
import threading
from queue import Queue, Empty


class AgentStatus(Enum):
    """Status of an agent."""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class AgentContext:
    """Context passed to agent during execution."""
    task_id: str
    task_description: str
    task_data: Dict[str, Any] = field(default_factory=dict)
    shared_state: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Result from agent execution."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0


class Agent(ABC):
    """Abstract base class for agents."""
    
    def __init__(self, agent_id: str, capabilities: Set[str]):
        """
        Initialize agent.
        
        Args:
            agent_id: Unique agent identifier
            capabilities: Set of capability tags
        """
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.status = AgentStatus.IDLE
        self.resources: Dict[str, Any] = {}
    
    @abstractmethod
    def execute(self, task, context: AgentContext) -> AgentResult:
        """
        Execute a task.
        
        Args:
            task: Task to execute
            context: Execution context
            
        Returns:
            AgentResult with execution outcome
        """
        pass
    
    def can_handle(self, required_capabilities: Set[str]) -> bool:
        """Check if agent can handle tasks requiring given capabilities."""
        return required_capabilities.issubset(self.capabilities)


@dataclass
class Message:
    """Message for inter-agent communication."""
    message_id: str
    sender: str
    recipient: Optional[str] = None  # None means broadcast
    content: Any = None
    message_type: str = "info"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MessageBus:
    """Thread-safe messaging bus for agent communication."""
    
    def __init__(self, episodic_memory=None):
        """
        Initialize message bus.
        
        Args:
            episodic_memory: Optional episodic memory for storing messages
        """
        self.episodic_memory = episodic_memory
        self._subscribers: Dict[str, List[Callable]] = {}
        self._message_queue: Queue = Queue()
        self._message_counter = 0
        self._lock = threading.Lock()
    
    def publish(self, sender: str, content: Any, recipient: Optional[str] = None,
                message_type: str = "info", metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Publish a message to the bus.
        
        Args:
            sender: Sender agent ID
            content: Message content
            recipient: Optional recipient agent ID (None for broadcast)
            message_type: Type of message
            metadata: Optional metadata
            
        Returns:
            Message ID
        """
        with self._lock:
            self._message_counter += 1
            message_id = f"msg_{self._message_counter}_{datetime.now().timestamp()}"
        
        message = Message(
            message_id=message_id,
            sender=sender,
            recipient=recipient,
            content=content,
            message_type=message_type,
            metadata=metadata or {}
        )
        
        # Store in episodic memory if available
        if self.episodic_memory:
            self.episodic_memory.add_event(
                "agent_message",
                {
                    "message_id": message_id,
                    "sender": sender,
                    "recipient": recipient,
                    "message_type": message_type,
                    "content": content
                },
                metadata={'type': 'agent_communication'}
            )
        
        # Put in queue for delivery
        self._message_queue.put(message)
        
        # Notify subscribers
        self._deliver_message(message)
        
        return message_id
    
    def subscribe(self, agent_id: str, callback: Callable[[Message], None]) -> None:
        """
        Subscribe an agent to messages.
        
        Args:
            agent_id: Agent ID
            callback: Function to call when message is received
        """
        with self._lock:
            if agent_id not in self._subscribers:
                self._subscribers[agent_id] = []
            self._subscribers[agent_id].append(callback)
    
    def _deliver_message(self, message: Message) -> None:
        """Deliver message to subscribers."""
        # Deliver to specific recipient or broadcast
        if message.recipient:
            # Targeted message
            if message.recipient in self._subscribers:
                for callback in self._subscribers[message.recipient]:
                    try:
                        callback(message)
                    except Exception:
                        pass  # Ignore callback errors
        else:
            # Broadcast to all subscribers except sender
            for agent_id, callbacks in self._subscribers.items():
                if agent_id != message.sender:
                    for callback in callbacks:
                        try:
                            callback(message)
                        except Exception:
                            pass


class AgentRegistry:
    """Registry for managing agents."""
    
    def __init__(self):
        """Initialize agent registry."""
        self.agents: Dict[str, Agent] = {}
        self._lock = threading.Lock()
    
    def register(self, agent: Agent) -> None:
        """
        Register an agent.
        
        Args:
            agent: Agent to register
        """
        with self._lock:
            self.agents[agent.agent_id] = agent
    
    def unregister(self, agent_id: str) -> bool:
        """
        Unregister an agent.
        
        Args:
            agent_id: Agent ID to unregister
            
        Returns:
            True if unregistered, False if not found
        """
        with self._lock:
            if agent_id in self.agents:
                del self.agents[agent_id]
                return True
            return False
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID."""
        return self.agents.get(agent_id)
    
    def find_capable_agents(self, required_capabilities: Set[str]) -> List[Agent]:
        """
        Find agents with required capabilities.
        
        Args:
            required_capabilities: Set of required capabilities
            
        Returns:
            List of capable agents
        """
        return [
            agent for agent in self.agents.values()
            if agent.can_handle(required_capabilities)
        ]
    
    def get_available_agents(self) -> List[Agent]:
        """Get all idle agents."""
        return [
            agent for agent in self.agents.values()
            if agent.status == AgentStatus.IDLE
        ]


class AgentRunner:
    """Executes tasks concurrently using agents."""
    
    def __init__(self, registry: AgentRegistry, message_bus: MessageBus):
        """
        Initialize agent runner.
        
        Args:
            registry: Agent registry
            message_bus: Message bus for coordination
        """
        self.registry = registry
        self.message_bus = message_bus
        self._execution_threads: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
    
    def execute_task(self, task, context: Optional[AgentContext] = None) -> Optional[AgentResult]:
        """
        Execute a task by assigning it to a capable agent.
        
        Args:
            task: Task to execute (should have task_id, description, required_capabilities)
            context: Optional execution context
            
        Returns:
            AgentResult if executed, None if no capable agent found
        """
        # Find a capable, available agent
        capable_agents = self.registry.find_capable_agents(task.required_capabilities)
        available_agents = [a for a in capable_agents if a.status == AgentStatus.IDLE]
        
        if not available_agents:
            return None
        
        # Select first available agent
        agent = available_agents[0]
        
        # Create context if not provided
        if context is None:
            context = AgentContext(
                task_id=task.task_id,
                task_description=task.description
            )
        
        # Execute
        agent.status = AgentStatus.BUSY
        
        try:
            start_time = datetime.now()
            result = agent.execute(task, context)
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time
            
            # Publish completion message
            self.message_bus.publish(
                agent.agent_id,
                {"task_id": task.task_id, "success": result.success},
                message_type="task_complete"
            )
            
            agent.status = AgentStatus.IDLE
            return result
            
        except Exception as e:
            agent.status = AgentStatus.ERROR
            return AgentResult(success=False, error=str(e))
    
    def execute_task_async(self, task, context: Optional[AgentContext] = None,
                          callback: Optional[Callable[[AgentResult], None]] = None) -> bool:
        """
        Execute a task asynchronously.
        
        Args:
            task: Task to execute
            context: Optional execution context
            callback: Optional callback for result
            
        Returns:
            True if task was started, False if no capable agent
        """
        def run():
            result = self.execute_task(task, context)
            if callback and result:
                callback(result)
        
        # Check if we have capable agents
        capable_agents = self.registry.find_capable_agents(task.required_capabilities)
        if not capable_agents:
            return False
        
        # Start execution thread
        thread = threading.Thread(target=run, daemon=True)
        
        with self._lock:
            self._execution_threads[task.task_id] = thread
        
        thread.start()
        return True
