"""
Hierarchical Planner - Goal decomposition and task scheduling.
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass

from .task_graph import Task, TaskGraph, TaskStatus
from ..memory import MemoryStore, MemoryQuery, EpisodicMemory


@dataclass
class PlanningContext:
    """Context information for planning."""
    goal: str
    constraints: Dict[str, Any]
    available_capabilities: Set[str]
    metadata: Dict[str, Any]


class Planner:
    """Hierarchical planner with goal decomposition and memory integration."""
    
    def __init__(self, memory_store: Optional[MemoryStore] = None):
        """
        Initialize planner.
        
        Args:
            memory_store: Optional memory store for consulting past plans
        """
        self.memory_store = memory_store
        self.episodic = EpisodicMemory(memory_store) if memory_store else None
        self._plan_counter = 0
    
    def plan(self, goal: str, context: Optional[PlanningContext] = None,
             constraints: Optional[Dict[str, Any]] = None) -> TaskGraph:
        """
        Create a plan (task graph) for achieving a goal.
        
        Args:
            goal: High-level goal description
            context: Optional planning context
            constraints: Optional constraints (e.g., time limit, resource limits)
            
        Returns:
            TaskGraph with decomposed tasks
        """
        self._plan_counter += 1
        
        # Create task graph
        graph = TaskGraph()
        
        # Create root task
        root_id = graph.add_task(
            description=goal,
            required_capabilities={"planning"}
        )
        
        # Decompose goal into subtasks
        subtasks = self._decompose_goal(goal, context, constraints)
        
        # Add subtasks to graph
        for i, subtask_desc in enumerate(subtasks):
            # Infer required capabilities from task description
            capabilities = self._infer_capabilities(subtask_desc)
            
            # Add subtask
            graph.add_task(
                description=subtask_desc,
                required_capabilities=capabilities,
                parent_task=root_id
            )
        
        # Log planning event to episodic memory
        if self.episodic:
            self.episodic.add_event(
                "planning",
                {
                    "plan_id": self._plan_counter,
                    "goal": goal,
                    "num_tasks": len(graph.tasks),
                    "constraints": constraints or {}
                },
                metadata={"type": "plan_created"}
            )
        
        return graph
    
    def refine(self, task: Task, context: Optional[PlanningContext] = None) -> List[Task]:
        """
        Refine a task into subtasks.
        
        Args:
            task: Task to refine
            context: Optional planning context
            
        Returns:
            List of subtasks
        """
        subtask_descriptions = self._decompose_goal(task.description, context)
        
        subtasks = []
        for desc in subtask_descriptions:
            capabilities = self._infer_capabilities(desc)
            subtasks.append(Task(
                task_id=f"{task.task_id}_sub_{len(subtasks)}",
                description=desc,
                required_capabilities=capabilities,
                parent_task=task.task_id
            ))
        
        return subtasks
    
    def _decompose_goal(self, goal: str, context: Optional[PlanningContext] = None,
                       constraints: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Decompose a goal into subtasks using heuristic rules.
        
        This is a simple heuristic decomposer. In a real system, this could use
        an LLM or more sophisticated planning algorithms.
        
        Args:
            goal: Goal to decompose
            context: Planning context
            constraints: Constraints
            
        Returns:
            List of subtask descriptions
        """
        # Check memory for similar past plans
        if self.episodic:
            recent_plans = self.episodic.get_recent_episodes(
                limit=5,
                event_type="planning"
            )
            # In a real system, we could learn from past successful plans
        
        # Simple heuristic decomposition based on keywords
        goal_lower = goal.lower()
        
        if "search" in goal_lower and "analyze" in goal_lower:
            return [
                "Search for relevant information",
                "Analyze search results",
                "Synthesize findings"
            ]
        elif "data" in goal_lower or "information" in goal_lower:
            return [
                "Gather required data",
                "Process and validate data",
                "Store or report results"
            ]
        elif "test" in goal_lower or "verify" in goal_lower:
            return [
                "Set up test environment",
                "Execute tests",
                "Analyze test results"
            ]
        else:
            # Default decomposition for generic goals
            return [
                f"Prepare resources for: {goal}",
                f"Execute main task: {goal}",
                f"Verify completion of: {goal}"
            ]
    
    def _infer_capabilities(self, task_description: str) -> Set[str]:
        """
        Infer required capabilities from task description.
        
        Args:
            task_description: Task description
            
        Returns:
            Set of required capability tags
        """
        desc_lower = task_description.lower()
        capabilities = set()
        
        # Map keywords to capabilities
        capability_keywords = {
            "search": {"search", "query"},
            "analyze": {"analysis", "reasoning"},
            "execute": {"execution"},
            "test": {"testing", "validation"},
            "verify": {"validation"},
            "gather": {"data_collection"},
            "process": {"data_processing"},
            "synthesize": {"synthesis", "reasoning"},
            "store": {"storage"},
            "report": {"reporting"},
        }
        
        for keyword, caps in capability_keywords.items():
            if keyword in desc_lower:
                capabilities.update(caps)
        
        # Always require at least basic execution capability
        if not capabilities:
            capabilities.add("execution")
        
        return capabilities


class Scheduler:
    """Capability-based task scheduler."""
    
    def __init__(self):
        """Initialize scheduler."""
        self.agent_capabilities: Dict[str, Set[str]] = {}
    
    def register_agent(self, agent_id: str, capabilities: Set[str]) -> None:
        """
        Register an agent with its capabilities.
        
        Args:
            agent_id: Agent identifier
            capabilities: Set of capability tags
        """
        self.agent_capabilities[agent_id] = capabilities
    
    def assign_task(self, task: Task) -> Optional[str]:
        """
        Assign a task to an agent based on capabilities.
        
        Args:
            task: Task to assign
            
        Returns:
            Agent ID if an agent can handle the task, None otherwise
        """
        # Find agents that have all required capabilities
        for agent_id, capabilities in self.agent_capabilities.items():
            if task.required_capabilities.issubset(capabilities):
                task.assigned_agent = agent_id
                return agent_id
        
        return None
    
    def assign_tasks(self, graph: TaskGraph) -> Dict[str, str]:
        """
        Assign all ready tasks to agents.
        
        Args:
            graph: Task graph
            
        Returns:
            Dictionary mapping task_id to agent_id
        """
        assignments = {}
        
        ready_tasks = graph.get_ready_tasks()
        
        for task in ready_tasks:
            agent_id = self.assign_task(task)
            if agent_id:
                assignments[task.task_id] = agent_id
        
        return assignments
