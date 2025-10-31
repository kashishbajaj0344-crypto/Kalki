"""
Phase 15 - Hierarchical Planner
Task decomposition and scheduling with capability-based agent assignment.
Enhanced with temporal/priority scheduling and memory persistence.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum
from datetime import datetime
import json


class TaskStatus(Enum):
    """Status of a task in the graph."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class Task:
    """Represents a single task in the task graph with temporal and priority features."""
    task_id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    required_capabilities: Set[str] = field(default_factory=set)
    dependencies: List[str] = field(default_factory=list)  # task_ids that must complete first
    subtasks: List[str] = field(default_factory=list)  # child task_ids
    parent_task: Optional[str] = None
    assigned_agent: Optional[str] = None
    result: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    # Enhanced temporal and priority features
    priority: int = 0  # Higher numbers = higher priority
    due_time: Optional[datetime] = None  # When task should be completed
    estimated_duration: Optional[int] = None  # Estimated seconds to complete
    actual_duration: Optional[int] = None  # Actual seconds taken
    success_rate: Optional[float] = None  # Historical success rate for similar tasks
    retry_count: int = 0  # Number of retry attempts
    max_retries: int = 3  # Maximum retry attempts

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization."""
        return {
            'task_id': self.task_id,
            'description': self.description,
            'status': self.status.value,
            'required_capabilities': list(self.required_capabilities),
            'dependencies': self.dependencies,
            'subtasks': self.subtasks,
            'parent_task': self.parent_task,
            'assigned_agent': self.assigned_agent,
            'result': self.result,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'priority': self.priority,
            'due_time': self.due_time.isoformat() if self.due_time else None,
            'estimated_duration': self.estimated_duration,
            'actual_duration': self.actual_duration,
            'success_rate': self.success_rate,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create task from dictionary."""
        return cls(
            task_id=data['task_id'],
            description=data['description'],
            status=TaskStatus(data['status']),
            required_capabilities=set(data.get('required_capabilities', [])),
            dependencies=data.get('dependencies', []),
            subtasks=data.get('subtasks', []),
            parent_task=data.get('parent_task'),
            assigned_agent=data.get('assigned_agent'),
            result=data.get('result'),
            metadata=data.get('metadata', {}),
            created_at=datetime.fromisoformat(data['created_at']),
            completed_at=datetime.fromisoformat(data['completed_at']) if data.get('completed_at') else None,
            priority=data.get('priority', 0),
            due_time=datetime.fromisoformat(data['due_time']) if data.get('due_time') else None,
            estimated_duration=data.get('estimated_duration'),
            actual_duration=data.get('actual_duration'),
            success_rate=data.get('success_rate'),
            retry_count=data.get('retry_count', 0),
            max_retries=data.get('max_retries', 3),
        )

    def is_overdue(self) -> bool:
        """Check if task is overdue."""
        if self.due_time is None:
            return False
        return datetime.now() > self.due_time

    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retry_count < self.max_retries

    def mark_started(self) -> None:
        """Mark task as started."""
        self.status = TaskStatus.IN_PROGRESS

    def mark_completed(self, result: Any = None) -> None:
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.completed_at = datetime.now()
        if self.created_at and self.completed_at:
            self.actual_duration = int((self.completed_at - self.created_at).total_seconds())

    def mark_failed(self, error: str = None) -> None:
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        if error:
            self.metadata['error'] = error
        self.retry_count += 1


class TaskGraph:
    """Represents a hierarchical task graph with dependencies and persistence."""

    def __init__(self, graph_id: Optional[str] = None):
        """Initialize empty task graph."""
        self.tasks: Dict[str, Task] = {}
        self._task_counter = 0
        self.graph_id = graph_id or f"graph_{datetime.now().timestamp()}"
        self.created_at = datetime.now()
        self.metadata: Dict[str, Any] = {}

    def add_task(self, description: str, required_capabilities: Optional[Set[str]] = None,
                 dependencies: Optional[List[str]] = None, parent_task: Optional[str] = None,
                 priority: int = 0, due_time: Optional[datetime] = None,
                 estimated_duration: Optional[int] = None) -> str:
        """
        Add a task to the graph with enhanced temporal/priority features.

        Args:
            description: Task description
            required_capabilities: Set of required agent capabilities
            dependencies: List of task IDs that must complete first
            parent_task: Parent task ID if this is a subtask
            priority: Task priority (higher = more important)
            due_time: When task should be completed
            estimated_duration: Estimated seconds to complete

        Returns:
            Task ID
        """
        self._task_counter += 1
        task_id = f"task_{self._task_counter}"

        task = Task(
            task_id=task_id,
            description=description,
            required_capabilities=required_capabilities or set(),
            dependencies=dependencies or [],
            parent_task=parent_task,
            priority=priority,
            due_time=due_time,
            estimated_duration=estimated_duration,
        )

        self.tasks[task_id] = task

        # Update parent's subtasks list
        if parent_task and parent_task in self.tasks:
            self.tasks[parent_task].subtasks.append(task_id)

        return task_id

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self.tasks.get(task_id)

    def update_status(self, task_id: str, status: TaskStatus, result: Any = None) -> bool:
        """
        Update task status.

        Args:
            task_id: Task ID
            status: New status
            result: Optional result data

        Returns:
            True if updated, False if task not found
        """
        if task_id not in self.tasks:
            return False

        task = self.tasks[task_id]
        task.status = status

        if result is not None:
            task.result = result

        if status == TaskStatus.COMPLETED:
            task.completed_at = datetime.now()
            if task.created_at and task.completed_at:
                task.actual_duration = int((task.completed_at - task.created_at).total_seconds())

        return True

    def get_ready_tasks(self, prioritize_temporal: bool = True) -> List[Task]:
        """
        Get tasks that are ready to execute (dependencies satisfied).
        Enhanced with temporal and priority ordering.

        Args:
            prioritize_temporal: If True, sort by due time then priority

        Returns:
            List of tasks with PENDING status and satisfied dependencies
        """
        ready = []

        for task in self.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue

            # Check if all dependencies are completed
            deps_satisfied = all(
                self.tasks.get(dep_id, Task("", "")).status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
            )

            if deps_satisfied:
                ready.append(task)

        if prioritize_temporal:
            # Sort by: overdue first, then by due time, then by priority (descending)
            def sort_key(task):
                overdue_penalty = 1000000 if task.is_overdue() else 0
                due_timestamp = task.due_time.timestamp() if task.due_time else 9999999999
                return (overdue_penalty + due_timestamp, -task.priority)

            ready.sort(key=sort_key)

        return ready

    def get_failed_tasks(self) -> List[Task]:
        """Get all failed tasks that can be retried."""
        return [task for task in self.tasks.values()
                if task.status == TaskStatus.FAILED and task.can_retry()]

    def retry_failed_tasks(self) -> List[str]:
        """
        Reset failed tasks that can be retried.

        Returns:
            List of task IDs that were reset for retry
        """
        reset_tasks = []
        for task in self.tasks.values():
            if task.status == TaskStatus.FAILED and task.can_retry():
                task.status = TaskStatus.PENDING
                task.metadata['retry_attempt'] = task.retry_count
                reset_tasks.append(task.task_id)
        return reset_tasks

    def get_subtasks(self, task_id: str) -> List[Task]:
        """Get all subtasks of a task."""
        task = self.tasks.get(task_id)
        if not task:
            return []

        return [self.tasks[tid] for tid in task.subtasks if tid in self.tasks]

    def is_complete(self) -> bool:
        """Check if all tasks are completed or failed."""
        return all(
            task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
            for task in self.tasks.values()
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive task statistics."""
        stats = {
            'graph_id': self.graph_id,
            'total': len(self.tasks),
            'pending': 0,
            'in_progress': 0,
            'completed': 0,
            'failed': 0,
            'blocked': 0,
            'overdue': 0,
            'avg_completion_time': None,
            'success_rate': None,
        }

        completed_times = []
        successful_tasks = 0

        for task in self.tasks.values():
            stats[task.status.value] += 1
            if task.is_overdue():
                stats['overdue'] += 1

            if task.status == TaskStatus.COMPLETED:
                successful_tasks += 1
                if task.actual_duration:
                    completed_times.append(task.actual_duration)

        if completed_times:
            stats['avg_completion_time'] = sum(completed_times) / len(completed_times)

        total_terminal = stats['completed'] + stats['failed']
        if total_terminal > 0:
            stats['success_rate'] = stats['completed'] / total_terminal

        return stats

    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary for serialization."""
        return {
            'graph_id': self.graph_id,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata,
            'tasks': {task_id: task.to_dict() for task_id, task in self.tasks.items()},
            'task_counter': self._task_counter,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskGraph':
        """Create graph from dictionary."""
        graph = cls(graph_id=data.get('graph_id'))
        graph.created_at = datetime.fromisoformat(data['created_at'])
        graph.metadata = data.get('metadata', {})
        graph._task_counter = data.get('task_counter', 0)

        for task_id, task_data in data.get('tasks', {}).items():
            graph.tasks[task_id] = Task.from_dict(task_data)

        return graph

    def save_to_memory(self, memory_store) -> bool:
        """
        Save graph to memory store.

        Args:
            memory_store: MemoryStore instance

        Returns:
            True if saved successfully
        """
        try:
            graph_data = self.to_dict()
            return memory_store.put(
                f"plan_{self.graph_id}",
                graph_data,
                metadata={"type": "task_graph", "goal": self.metadata.get("goal", "")}
            )
        except Exception:
            return False

    @classmethod
    def load_from_memory(cls, graph_id: str, memory_store) -> Optional['TaskGraph']:
        """
        Load graph from memory store.

        Args:
            graph_id: Graph ID to load
            memory_store: MemoryStore instance

        Returns:
            TaskGraph if found, None otherwise
        """
        try:
            entry = memory_store.get(f"plan_{graph_id}")
            if entry:
                return cls.from_dict(entry.value)
        except Exception:
            pass
        return None

# [Kalki v2.3 — planner/task_graph.py v1.0]