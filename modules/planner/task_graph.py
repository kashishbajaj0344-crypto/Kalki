"""
Phase 15 - Hierarchical Planner
Task decomposition and scheduling with capability-based agent assignment.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum
from datetime import datetime


class TaskStatus(Enum):
    """Status of a task in the graph."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class Task:
    """Represents a single task in the task graph."""
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


class TaskGraph:
    """Represents a hierarchical task graph with dependencies."""
    
    def __init__(self):
        """Initialize empty task graph."""
        self.tasks: Dict[str, Task] = {}
        self._task_counter = 0
    
    def add_task(self, description: str, required_capabilities: Optional[Set[str]] = None,
                 dependencies: Optional[List[str]] = None, parent_task: Optional[str] = None) -> str:
        """
        Add a task to the graph.
        
        Args:
            description: Task description
            required_capabilities: Set of required agent capabilities
            dependencies: List of task IDs that must complete first
            parent_task: Parent task ID if this is a subtask
            
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
            parent_task=parent_task
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
        
        return True
    
    def get_ready_tasks(self) -> List[Task]:
        """
        Get tasks that are ready to execute (dependencies satisfied).
        
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
        
        return ready
    
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
    
    def get_statistics(self) -> Dict[str, int]:
        """Get task statistics."""
        stats = {
            'total': len(self.tasks),
            'pending': 0,
            'in_progress': 0,
            'completed': 0,
            'failed': 0,
            'blocked': 0
        }
        
        for task in self.tasks.values():
            stats[task.status.value] += 1
        
        return stats
