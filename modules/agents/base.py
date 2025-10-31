"""
Kalki Agents Module - Base Classes and Infrastructure
Enhanced with memory integration, cooperative chaining, and self-monitoring.
"""

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from queue import PriorityQueue, Queue
from threading import Lock, Event
from typing import Any, Dict, List, Optional, Callable, Union, Set
from concurrent.futures import ThreadPoolExecutor

# Memory integration imports (with fallbacks)
try:
    from modules.memory.episodic_memory import EpisodicMemory
    from modules.memory.semantic_memory import SemanticMemory
except ImportError:
    # Fallback for when memory modules aren't available
    class EpisodicMemory:
        def add_event(self, event): pass

    class SemanticMemory:
        def store(self, key, value): pass
        def retrieve(self, key): return None

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent execution states."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class MessageType(Enum):
    """Message types for agent communication."""
    TASK = "task"
    RESULT = "result"
    STATUS = "status"
    ERROR = "error"
    CONTROL = "control"
    COORDINATION = "coordination"


@dataclass
class AgentTask:
    """Structured task definition for agents."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    priority: int = 1
    capabilities_required: Set[str] = field(default_factory=set)
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    assigned_to: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentTask':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class Message:
    """Message for agent communication."""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType = MessageType.TASK
    sender: str = ""
    recipient: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    priority: int = 1
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['message_type'] = self.message_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create from dictionary."""
        data_copy = data.copy()
        data_copy['message_type'] = MessageType(data['message_type'])
        return cls(**data_copy)


@dataclass
class AgentContext:
    """Context for agent execution."""
    agent_id: str
    capabilities: Set[str] = field(default_factory=set)
    memory_context: Dict[str, Any] = field(default_factory=dict)
    session_data: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentContext':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class AgentResult:
    """Result from agent execution."""
    task_id: str
    agent_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentResult':
        """Create from dictionary."""
        return cls(**data)


class Agent(ABC):
    """Base agent class with memory integration and self-monitoring."""

    def __init__(self, agent_id: str, capabilities: Set[str]):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.status = AgentStatus.IDLE
        self.context = AgentContext(agent_id, capabilities)
        self._lock = Lock()
        self._stop_event = Event()

        # Memory integration
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()

        # Performance tracking
        self.execution_count = 0
        self.success_count = 0
        self.average_execution_time = 0.0
        self.last_execution_time = 0.0

        logger.info(f"Agent {agent_id} initialized with capabilities: {capabilities}")

    @abstractmethod
    async def execute_task(self, task: AgentTask) -> AgentResult:
        """Execute a task asynchronously."""
        pass

    def can_handle_task(self, task: AgentTask) -> bool:
        """Check if agent can handle the given task."""
        return task.capabilities_required.issubset(self.capabilities)

    def update_performance_metrics(self, execution_time: float, success: bool):
        """Update performance metrics."""
        with self._lock:
            self.execution_count += 1
            if success:
                self.success_count += 1

            # Update average execution time
            if self.execution_count == 1:
                self.average_execution_time = execution_time
            else:
                self.average_execution_time = (
                    (self.average_execution_time * (self.execution_count - 1)) + execution_time
                ) / self.execution_count

            self.last_execution_time = execution_time

            # Log to episodic memory
            self.episodic_memory.add_event({
                'type': 'performance_update',
                'agent_id': self.agent_id,
                'execution_time': execution_time,
                'success': success,
                'timestamp': time.time()
            })

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        with self._lock:
            return {
                'execution_count': self.execution_count,
                'success_rate': self.success_count / max(self.execution_count, 1),
                'average_execution_time': self.average_execution_time,
                'last_execution_time': self.last_execution_time
            }

    def log_agent_activity(self, activity_type: str, details: Dict[str, Any]):
        """Log agent activity to episodic memory."""
        event = {
            'type': 'agent_activity',
            'agent_id': self.agent_id,
            'activity_type': activity_type,
            'details': details,
            'timestamp': time.time()
        }
        self.episodic_memory.add_event(event)

    def update_semantic_memory(self, key: str, value: Any):
        """Update semantic memory with learned knowledge."""
        self.semantic_memory.store(key, value)

    def retrieve_from_semantic_memory(self, key: str) -> Any:
        """Retrieve knowledge from semantic memory."""
        return self.semantic_memory.retrieve(key)

    def stop(self):
        """Stop the agent."""
        self._stop_event.set()
        self.status = AgentStatus.IDLE
        logger.info(f"Agent {self.agent_id} stopped")


class MessageBus:
    """Enhanced message bus with priority queues and persistence."""

    def __init__(self, persistence_path: Optional[str] = None):
        self._queues: Dict[str, PriorityQueue] = {}
        self._subscriptions: Dict[str, Set[str]] = {}
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        self.persistence_path = persistence_path
        self._message_history: List[Message] = []

        # Load persisted messages if path provided
        if persistence_path:
            self._load_persisted_messages()

    def _load_persisted_messages(self):
        """Load persisted messages from disk."""
        try:
            if self.persistence_path and os.path.exists(self.persistence_path):
                with open(self.persistence_path, 'r') as f:
                    data = json.load(f)
                    for msg_data in data.get('messages', []):
                        msg = Message.from_dict(msg_data)
                        self._route_message(msg, persist=False)
        except Exception as e:
            logger.error(f"Failed to load persisted messages: {e}")

    def _persist_message(self, message: Message):
        """Persist message to disk."""
        if not self.persistence_path:
            return

        try:
            self._message_history.append(message)
            # Keep only last 1000 messages
            if len(self._message_history) > 1000:
                self._message_history = self._message_history[-1000:]

            data = {
                'messages': [msg.to_dict() for msg in self._message_history]
            }

            with open(self.persistence_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist message: {e}")

    def subscribe(self, recipient: str, callback: Callable[[Message], None]):
        """Subscribe to messages for a recipient."""
        with self._lock:
            if recipient not in self._queues:
                self._queues[recipient] = PriorityQueue()
            if recipient not in self._subscriptions:
                self._subscriptions[recipient] = set()
            self._subscriptions[recipient].add(id(callback))

            # Start message processing thread
            self._executor.submit(self._process_messages, recipient, callback)

    def unsubscribe(self, recipient: str, callback: Callable[[Message], None]):
        """Unsubscribe from messages."""
        with self._lock:
            if recipient in self._subscriptions:
                self._subscriptions[recipient].discard(id(callback))

    def send_message(self, message: Message, persist: bool = True):
        """Send a message to the bus."""
        self._route_message(message, persist)

    def _route_message(self, message: Message, persist: bool = True):
        """Route message to appropriate recipients."""
        with self._lock:
            recipient = message.recipient
            if recipient not in self._queues:
                self._queues[recipient] = PriorityQueue()

            # Priority queue: higher priority (lower number) gets processed first
            self._queues[recipient].put((-message.priority, message))

            if persist:
                self._persist_message(message)

    def _process_messages(self, recipient: str, callback: Callable[[Message], None]):
        """Process messages for a recipient."""
        while True:
            try:
                if recipient not in self._queues:
                    break

                priority, message = self._queues[recipient].get(timeout=1.0)
                callback(message)
                self._queues[recipient].task_done()

            except Exception:
                # Queue empty or other error, continue
                continue

    def get_message_history(self, recipient: Optional[str] = None, limit: int = 100) -> List[Message]:
        """Get message history, optionally filtered by recipient."""
        with self._lock:
            if recipient:
                return [msg for msg in self._message_history
                       if msg.recipient == recipient][-limit:]
            return self._message_history[-limit:]

    def clear_history(self):
        """Clear message history."""
        with self._lock:
            self._message_history.clear()
            if self.persistence_path and os.path.exists(self.persistence_path):
                try:
                    os.remove(self.persistence_path)
                except Exception as e:
                    logger.error(f"Failed to clear persisted messages: {e}")


class AgentRegistry:
    """Enhanced agent registry with traits and performance metrics."""

    def __init__(self):
        self._agents: Dict[str, Agent] = {}
        self._agent_traits: Dict[str, Dict[str, Any]] = {}
        self._performance_history: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = Lock()

    def register_agent(self, agent: Agent, traits: Optional[Dict[str, Any]] = None):
        """Register an agent with optional traits."""
        with self._lock:
            self._agents[agent.agent_id] = agent
            self._agent_traits[agent.agent_id] = traits or {}
            self._performance_history[agent.agent_id] = []
            logger.info(f"Agent {agent.agent_id} registered with traits: {traits}")

    def unregister_agent(self, agent_id: str):
        """Unregister an agent."""
        with self._lock:
            if agent_id in self._agents:
                del self._agents[agent_id]
                del self._agent_traits[agent_id]
                del self._performance_history[agent_id]
                logger.info(f"Agent {agent_id} unregistered")

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID."""
        with self._lock:
            return self._agents.get(agent_id)

    def find_agents_for_task(self, task: AgentTask) -> List[Agent]:
        """Find agents that can handle a task using capability matching and performance."""
        with self._lock:
            candidates = []
            for agent in self._agents.values():
                if agent.can_handle_task(task) and agent.status == AgentStatus.IDLE:
                    candidates.append(agent)

            # Sort by performance (success rate, then execution time)
            candidates.sort(key=lambda a: (
                -a.get_performance_stats()['success_rate'],
                a.get_performance_stats()['average_execution_time']
            ))

            return candidates

    def update_agent_performance(self, agent_id: str, metrics: Dict[str, Any]):
        """Update performance metrics for an agent."""
        with self._lock:
            if agent_id in self._performance_history:
                self._performance_history[agent_id].append({
                    **metrics,
                    'timestamp': time.time()
                })
                # Keep only last 100 entries
                if len(self._performance_history[agent_id]) > 100:
                    self._performance_history[agent_id] = self._performance_history[agent_id][-100:]

    def get_agent_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get comprehensive stats for an agent."""
        with self._lock:
            agent = self._agents.get(agent_id)
            if not agent:
                return {}

            return {
                'traits': self._agent_traits.get(agent_id, {}),
                'performance': agent.get_performance_stats(),
                'performance_history': self._performance_history.get(agent_id, []),
                'status': agent.status.value,
                'capabilities': list(agent.capabilities)
            }

    def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents with their info."""
        with self._lock:
            return [{
                'agent_id': agent_id,
                'capabilities': list(agent.capabilities),
                'status': agent.status.value,
                'traits': self._agent_traits.get(agent_id, {})
            } for agent_id, agent in self._agents.items()]


class AgentMonitor:
    """Self-monitoring system for agent performance and health."""

    def __init__(self, registry: AgentRegistry, check_interval: float = 30.0):
        self.registry = registry
        self.check_interval = check_interval
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = Event()
        self._alerts: List[Dict[str, Any]] = []
        self._lock = Lock()

        # Monitoring thresholds
        self.failure_threshold = 0.3  # 30% failure rate
        self.slow_execution_threshold = 10.0  # 10 seconds
        self.inactive_threshold = 300.0  # 5 minutes

    def start_monitoring(self):
        """Start the monitoring thread."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return

        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Agent monitoring started")

    def stop_monitoring(self):
        """Stop the monitoring thread."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Agent monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self._stop_event.is_set():
            try:
                self._perform_health_checks()
                self._cleanup_inactive_agents()
                self._generate_alerts()
            except Exception as e:
                logger.error(f"Monitoring error: {e}")

            self._stop_event.wait(self.check_interval)

    def _perform_health_checks(self):
        """Perform health checks on all agents."""
        for agent_info in self.registry.list_agents():
            agent_id = agent_info['agent_id']
            stats = self.registry.get_agent_stats(agent_id)

            # Check failure rate
            perf = stats.get('performance', {})
            failure_rate = 1.0 - perf.get('success_rate', 1.0)
            if failure_rate > self.failure_threshold:
                self._add_alert(agent_id, 'high_failure_rate',
                              f"Failure rate: {failure_rate:.2%}")

            # Check execution time
            avg_time = perf.get('average_execution_time', 0.0)
            if avg_time > self.slow_execution_threshold:
                self._add_alert(agent_id, 'slow_execution',
                              f"Average execution time: {avg_time:.2f}s")

    def _cleanup_inactive_agents(self):
        """Remove agents that have been inactive too long."""
        current_time = time.time()
        inactive_agents = []

        for agent_info in self.registry.list_agents():
            agent_id = agent_info['agent_id']
            stats = self.registry.get_agent_stats(agent_id)
            perf_history = stats.get('performance_history', [])

            if not perf_history:
                continue

            last_activity = max(entry['timestamp'] for entry in perf_history)
            if current_time - last_activity > self.inactive_threshold:
                inactive_agents.append(agent_id)

        for agent_id in inactive_agents:
            logger.warning(f"Removing inactive agent: {agent_id}")
            self.registry.unregister_agent(agent_id)

    def _generate_alerts(self):
        """Generate and log alerts."""
        with self._lock:
            if self._alerts:
                logger.warning(f"Agent alerts: {len(self._alerts)}")
                for alert in self._alerts[-10:]:  # Log last 10 alerts
                    logger.warning(f"Alert: {alert}")
                self._alerts.clear()

    def _add_alert(self, agent_id: str, alert_type: str, message: str):
        """Add an alert to the queue."""
        with self._lock:
            self._alerts.append({
                'agent_id': agent_id,
                'type': alert_type,
                'message': message,
                'timestamp': time.time()
            })

    def get_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        with self._lock:
            return self._alerts[-limit:]


class AgentRunner:
    """Enhanced agent runner with cooperative chaining and async support."""

    def __init__(self, registry: AgentRegistry, message_bus: MessageBus):
        self.registry = registry
        self.message_bus = message_bus
        self._executor = ThreadPoolExecutor(max_workers=8)
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._task_dependencies: Dict[str, Set[str]] = {}
        self._completed_tasks: Set[str] = set()
        self._lock = Lock()

    async def execute_task_chain(self, tasks: List[AgentTask]) -> List[AgentResult]:
        """Execute a chain of dependent tasks cooperatively."""
        # Build dependency graph
        self._build_dependency_graph(tasks)

        results = []
        pending_tasks = {task.task_id: task for task in tasks}

        while pending_tasks:
            # Find tasks with satisfied dependencies
            ready_tasks = []
            for task_id, task in pending_tasks.items():
                deps = self._task_dependencies.get(task_id, set())
                if deps.issubset(self._completed_tasks):
                    ready_tasks.append(task)

            if not ready_tasks:
                # Circular dependency or unsatisfiable deps
                raise ValueError("Circular or unsatisfiable task dependencies")

            # Execute ready tasks concurrently
            batch_results = await self._execute_batch(ready_tasks)
            results.extend(batch_results)

            # Mark completed and remove from pending
            for result in batch_results:
                if result.success:
                    self._completed_tasks.add(result.task_id)
                    del pending_tasks[result.task_id]

        return results

    def _build_dependency_graph(self, tasks: List[AgentTask]):
        """Build dependency graph from tasks."""
        self._task_dependencies.clear()
        for task in tasks:
            self._task_dependencies[task.task_id] = set(task.dependencies)

    async def _execute_batch(self, tasks: List[AgentTask]) -> List[AgentResult]:
        """Execute a batch of independent tasks concurrently."""
        async def execute_single(task):
            return await self.execute_single_task(task)

        tasks_coroutines = [execute_single(task) for task in tasks]
        return await asyncio.gather(*tasks_coroutines, return_exceptions=True)

    async def execute_single_task(self, task: AgentTask) -> AgentResult:
        """Execute a single task using the best available agent."""
        start_time = time.time()

        try:
            # Find suitable agent
            candidates = self.registry.find_agents_for_task(task)
            if not candidates:
                return AgentResult(
                    task_id=task.task_id,
                    agent_id="",
                    success=False,
                    error="No suitable agent found",
                    execution_time=time.time() - start_time
                )

            # Use first (best) candidate
            agent = candidates[0]
            task.assigned_to = agent.agent_id

            # Execute task
            result = await agent.execute_task(task)

            # Update performance metrics
            execution_time = time.time() - start_time
            agent.update_performance_metrics(execution_time, result.success)
            self.registry.update_agent_performance(agent.agent_id, {
                'task_id': task.task_id,
                'execution_time': execution_time,
                'success': result.success
            })

            result.execution_time = execution_time
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Task execution failed: {e}")
            return AgentResult(
                task_id=task.task_id,
                agent_id="",
                success=False,
                error=str(e),
                execution_time=execution_time
            )

    def cancel_task(self, task_id: str):
        """Cancel a running task."""
        with self._lock:
            if task_id in self._active_tasks:
                self._active_tasks[task_id].cancel()
                del self._active_tasks[task_id]

    def get_active_tasks(self) -> List[str]:
        """Get IDs of currently active tasks."""
        with self._lock:
            return list(self._active_tasks.keys())

# [Kalki v2.3 — agents/base.py v1.0]