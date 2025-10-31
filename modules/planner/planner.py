"""
Hierarchical Planner - Goal decomposition and task scheduling.
Enhanced with memory integration, feedback loops, and LLM-based planning.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime

from .task_graph import Task, TaskGraph, TaskStatus
from ..agents.memory import EpisodicMemory, SemanticMemory, MemoryQuery
from ..llm import ask_kalki

logger = logging.getLogger('Kalki.Planner')


@dataclass
class PlanningContext:
    """Context information for planning with memory integration."""
    goal: str
    constraints: Dict[str, Any]
    available_capabilities: Set[str]
    metadata: Dict[str, Any]
    use_memory: bool = True
    use_llm: bool = False


class Planner:
    """Hierarchical planner with goal decomposition, memory integration, and LLM enhancement."""

    def __init__(self, episodic_memory: Optional[EpisodicMemory] = None,
                 semantic_memory: Optional[SemanticMemory] = None):
        """
        Initialize planner with memory integration.

        Args:
            episodic_memory: For logging and retrieving past planning events
            semantic_memory: For storing and retrieving plan templates
        """
        self.episodic = episodic_memory
        self.semantic = semantic_memory
        self._plan_counter = 0
        logger.info("[Kalki.Planner] Planner initialized with memory integration")

    async def plan(self, goal: str, context: Optional[PlanningContext] = None,
                   constraints: Optional[Dict[str, Any]] = None) -> TaskGraph:
        """
        Create a plan (task graph) for achieving a goal with memory-enhanced planning.

        Args:
            goal: High-level goal description
            context: Optional planning context
            constraints: Optional constraints (time, resources, etc.)

        Returns:
            TaskGraph with decomposed tasks
        """
        self._plan_counter += 1
        start_time = datetime.now()

        # Create planning context if not provided
        if context is None:
            context = PlanningContext(
                goal=goal,
                constraints=constraints or {},
                available_capabilities=set(),
                metadata={}
            )

        # Log planning started event
        if self.episodic:
            await self._log_planning_event_async("planning_started", {
                "plan_id": self._plan_counter,
                "goal": goal,
                "constraints": context.constraints,
                "use_memory": context.use_memory,
                "use_llm": context.use_llm
            })

        # Retrieve past planning knowledge
        past_insights = {}
        if context.use_memory and self.episodic:
            past_insights = await self._retrieve_past_plans(goal)

        # Create task graph
        graph = TaskGraph()
        graph.metadata.update({
            "goal": goal,
            "plan_id": self._plan_counter,
            "created_at": start_time.isoformat(),
            "use_memory": context.use_memory,
            "use_llm": context.use_llm
        })

        # Create root task
        root_id = graph.add_task(
            description=goal,
            required_capabilities={"planning"}
        )

        # Decompose goal into subtasks using enhanced methods
        subtasks = await self._decompose_goal_enhanced(goal, context, past_insights)

        # Add subtasks to graph with inferred properties
        for i, subtask_info in enumerate(subtasks):
            # Extract subtask details
            if isinstance(subtask_info, dict):
                desc = subtask_info.get('description', str(subtask_info))
                priority = subtask_info.get('priority', 0)
                due_time = subtask_info.get('due_time')
                capabilities = subtask_info.get('capabilities', self._infer_capabilities(desc))
                estimated_duration = subtask_info.get('estimated_duration')
            else:
                desc = str(subtask_info)
                priority = 0
                due_time = None
                capabilities = self._infer_capabilities(desc)
                estimated_duration = None

            # Add success rate from past performance
            success_rate = past_insights.get('success_rates', {}).get(desc, None)

            # Create subtask
            subtask = Task(
                task_id=f"{root_id}_sub_{i}",
                description=desc,
                required_capabilities=capabilities,
                parent_task=root_id,
                priority=priority,
                due_time=due_time,
                estimated_duration=estimated_duration,
                success_rate=success_rate
            )

            graph.tasks[subtask.task_id] = subtask
            graph.tasks[root_id].subtasks.append(subtask.task_id)

        # Calculate plan statistics
        plan_stats = graph.get_statistics()

        # Log planning completed event
        if self.episodic:
            await self._log_planning_event_async("plan_completed", {
                "plan_id": self._plan_counter,
                "goal": goal,
                "num_subtasks": len(graph.tasks) - 1,  # Exclude root task
                "success_rate": plan_stats.get('success_rate'),
                "avg_completion_time": plan_stats.get('avg_completion_time'),
                "planning_duration": (datetime.now() - start_time).total_seconds()
            })

        # Store plan template in semantic memory for future reuse
        if self.semantic:
            await self._store_plan_template(goal, subtasks)

        logger.info(f"[Kalki.Planner] Created plan {self._plan_counter} with {len(graph.tasks)-1} subtasks")
        return graph

    async def replan_failed_task(self, graph: TaskGraph, failed_task_id: str) -> bool:
        """
        Reactive replanning for failed tasks.

        Args:
            graph: Task graph containing the failed task
            failed_task_id: ID of the failed task

        Returns:
            True if replanning was successful
        """
        task = graph.get_task(failed_task_id)
        if not task or task.status != TaskStatus.FAILED:
            return False

        # Query episodic memory for similar failures
        if self.episodic:
            similar_failures = await self._query_similar_failures(task.description)

            # Log replanning attempt
            await self._log_planning_event_async("plan_revised", {
                "plan_id": graph.metadata.get("plan_id"),
                "failed_task": failed_task_id,
                "reason": "task_failure",
                "similar_failures_found": len(similar_failures)
            })

        # Attempt to refine the failed task
        refined_tasks = await self.refine_task(task)

        if refined_tasks:
            # Replace failed task with refined subtasks
            for refined_task in refined_tasks:
                graph.tasks[refined_task.task_id] = refined_task
                # Update dependencies if needed
                for other_task in graph.tasks.values():
                    if failed_task_id in other_task.dependencies:
                        other_task.dependencies.append(refined_task.task_id)

            # Remove the failed task
            if failed_task_id in graph.tasks:
                del graph.tasks[failed_task_id]

            logger.info(f"[Kalki.Planner] Replanned failed task {failed_task_id} into {len(refined_tasks)} subtasks")
            return True

        return False

    async def refine_task(self, task: Task, context: Optional[PlanningContext] = None) -> List[Task]:
        """
        Refine a task into subtasks using enhanced methods.

        Args:
            task: Task to refine
            context: Optional planning context

        Returns:
            List of subtasks
        """
        # Try LLM-based refinement first if enabled
        if context and context.use_llm:
            try:
                llm_subtasks = await self._llm_decompose_task(task.description)
                if llm_subtasks:
                    return self._create_subtasks_from_descriptions(
                        llm_subtasks, task.task_id, task.priority - 1
                    )
            except Exception as e:
                logger.warning(f"[Kalki.Planner] LLM refinement failed: {e}")

        # Fall back to heuristic decomposition
        subtask_descriptions = await self._decompose_goal_enhanced(task.description, context)
        return self._create_subtasks_from_descriptions(
            subtask_descriptions, task.task_id, task.priority - 1
        )

    async def _decompose_goal_enhanced(self, goal: str,
                                     context: Optional[PlanningContext] = None,
                                     past_insights: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        Enhanced goal decomposition using memory, heuristics, and LLM.

        Args:
            goal: Goal to decompose
            context: Planning context
            past_insights: Insights from past similar plans

        Returns:
            List of subtask descriptions or detailed subtask info
        """
        # Check semantic memory for similar plan templates
        if context and context.use_memory and self.semantic:
            template_results = self.semantic.search_similar(f"plan for {goal}", limit=3)
            if template_results and template_results[0]['score'] > 0.7:
                # Use stored template
                template = template_results[0]['metadata'].get('subtasks', [])
                if template:
                    logger.info(f"[Kalki.Planner] Using stored plan template for: {goal}")
                    return template

        # Try LLM-based decomposition if enabled
        if context and context.use_llm:
            try:
                llm_subtasks = await self._llm_decompose_goal(goal)
                if llm_subtasks:
                    return llm_subtasks
            except Exception as e:
                logger.warning(f"[Kalki.Planner] LLM decomposition failed: {e}")

        # Fall back to enhanced heuristic decomposition
        return self._heuristic_decompose_goal(goal, past_insights)

    async def _llm_decompose_goal(self, goal: str) -> List[Dict[str, Any]]:
        """
        Use LLM to decompose goal into detailed subtasks.

        Args:
            goal: Goal to decompose

        Returns:
            List of detailed subtask dictionaries
        """
        prompt = f"""
        Decompose the following goal into 3-5 specific, actionable subtasks.
        For each subtask, provide:
        - description: Clear, actionable description
        - priority: Number from 1-10 (higher = more important)
        - estimated_duration: Estimated minutes to complete
        - required_capabilities: List of skills needed

        Goal: {goal}

        Return as a JSON array of objects.
        """

        try:
            # Use synchronous ask_kalki in a thread pool to avoid blocking
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, ask_kalki, prompt)
            
            # Parse JSON response (assuming LLM returns valid JSON)
            import json
            subtasks = json.loads(response)
            return subtasks if isinstance(subtasks, list) else []
        except Exception as e:
            logger.warning(f"[Kalki.Planner] Failed to parse LLM response: {e}")
            return []

    async def _llm_decompose_task(self, task_description: str) -> List[str]:
        """
        Use LLM to break down a task into simpler steps.

        Args:
            task_description: Task to decompose

        Returns:
            List of simpler subtask descriptions
        """
        prompt = f"""
        Break down this task into 2-4 simpler, specific steps:

        Task: {task_description}

        Return only a numbered list of steps, one per line.
        """

        try:
            # Use synchronous ask_kalki in a thread pool
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, ask_kalki, prompt)
            
            # Parse numbered list
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            subtasks = []
            for line in lines:
                if line[0].isdigit() and '. ' in line:
                    subtasks.append(line.split('. ', 1)[1])
            return subtasks
        except Exception as e:
            logger.warning(f"[Kalki.Planner] Failed to parse LLM task breakdown: {e}")
            return []

    def _heuristic_decompose_goal(self, goal: str, past_insights: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Enhanced heuristic decomposition with past performance insights.

        Args:
            goal: Goal to decompose
            past_insights: Insights from past plans

        Returns:
            List of subtask descriptions
        """
        goal_lower = goal.lower()

        # Use past insights to inform decomposition
        success_rates = past_insights.get('success_rates', {}) if past_insights else {}

        # Domain-specific decomposition patterns
        if "search" in goal_lower and "analyze" in goal_lower:
            base_subtasks = [
                "Search for relevant information",
                "Analyze search results",
                "Synthesize findings"
            ]
        elif "data" in goal_lower or "information" in goal_lower:
            base_subtasks = [
                "Gather required data",
                "Process and validate data",
                "Store or report results"
            ]
        elif "test" in goal_lower or "verify" in goal_lower:
            base_subtasks = [
                "Set up test environment",
                "Execute tests",
                "Analyze test results"
            ]
        elif "plan" in goal_lower or "schedule" in goal_lower:
            base_subtasks = [
                "Define objectives and constraints",
                "Break down into manageable tasks",
                "Create timeline and assign resources",
                "Review and validate plan"
            ]
        else:
            # Generic decomposition with learning from past performance
            base_subtasks = [
                f"Prepare resources for: {goal}",
                f"Execute main task: {goal}",
                f"Verify completion of: {goal}"
            ]

        # Adjust based on historical success rates
        if success_rates:
            # Prioritize subtasks with higher historical success rates
            adjusted_subtasks = []
            for subtask in base_subtasks:
                if subtask in success_rates and success_rates[subtask] < 0.5:
                    # Add verification step for historically problematic subtasks
                    adjusted_subtasks.append(subtask)
                    adjusted_subtasks.append(f"Verify: {subtask}")
                else:
                    adjusted_subtasks.append(subtask)
            return adjusted_subtasks

        return base_subtasks

    async def _retrieve_past_plans(self, goal: str) -> Dict[str, Any]:
        """
        Retrieve insights from past similar planning attempts.

        Args:
            goal: Current goal

        Returns:
            Dictionary with past planning insights
        """
        insights = {
            'success_rates': {},
            'avg_durations': {},
            'common_patterns': []
        }

        try:
            # Query episodic memory for past planning events
            query = MemoryQuery(
                filter={"type": "plan_completed"},
                limit=10
            )

            past_plans = self.episodic.store.query(query)

            for plan_entry in past_plans:
                plan_data = plan_entry.value

                # Check if goal is similar
                past_goal = plan_data.get("goal", "").lower()
                current_goal = goal.lower()

                # Simple similarity check (could be enhanced with embeddings)
                if any(word in past_goal for word in current_goal.split()):
                    success_rate = plan_data.get("success_rate")
                    if success_rate is not None:
                        # Aggregate success rates for similar goals
                        key = f"similar_goal_{hash(past_goal) % 1000}"
                        if key not in insights['success_rates']:
                            insights['success_rates'][key] = []
                        insights['success_rates'][key].append(success_rate)

            # Average the success rates
            for key, rates in insights['success_rates'].items():
                insights['success_rates'][key] = sum(rates) / len(rates)

        except Exception as e:
            logger.warning(f"[Kalki.Planner] Failed to retrieve past plans: {e}")

        return insights

    async def _query_similar_failures(self, task_description: str) -> List[Dict[str, Any]]:
        """
        Query for similar past task failures.

        Args:
            task_description: Description of failed task

        Returns:
            List of similar failure cases
        """
        try:
            # Query for failed task events
            query = MemoryQuery(
                filter={"type": "task_failed"},
                limit=5
            )

            failures = self.episodic.store.query(query)
            similar_failures = []

            for failure_entry in failures:
                failure_data = failure_entry.value
                failed_task_desc = failure_data.get("task_description", "")

                # Simple similarity check
                if any(word in failed_task_desc.lower() for word in task_description.lower().split()):
                    similar_failures.append(failure_data)

            return similar_failures

        except Exception as e:
            logger.warning(f"[Kalki.Planner] Failed to query similar failures: {e}")
            return []

    async def _store_plan_template(self, goal: str, subtasks: List[Any]) -> None:
        """
        Store successful plan decomposition as template for future reuse.

        Args:
            goal: Original goal
            subtasks: Decomposed subtasks
        """
        try:
            template_text = f"Plan template for: {goal}"
            template_data = {
                "goal": goal,
                "subtasks": subtasks,
                "created_at": datetime.now().isoformat()
            }

            await self.semantic.add_document_async(
                template_text,
                metadata={"type": "plan_template", "goal": goal, "subtasks": subtasks}
            )

            logger.debug(f"[Kalki.Planner] Stored plan template for: {goal}")

        except Exception as e:
            logger.warning(f"[Kalki.Planner] Failed to store plan template: {e}")

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

        # Enhanced capability mapping
        capability_keywords = {
            "search": {"search", "query", "information_retrieval"},
            "analyze": {"analysis", "reasoning", "data_analysis"},
            "execute": {"execution", "task_execution"},
            "test": {"testing", "validation", "quality_assurance"},
            "verify": {"validation", "checking"},
            "gather": {"data_collection", "information_gathering"},
            "process": {"data_processing", "computation"},
            "synthesize": {"synthesis", "reasoning", "integration"},
            "store": {"storage", "data_management"},
            "report": {"reporting", "communication"},
            "plan": {"planning", "scheduling"},
            "schedule": {"scheduling", "time_management"},
            "coordinate": {"coordination", "management"},
        }

        for keyword, caps in capability_keywords.items():
            if keyword in desc_lower:
                capabilities.update(caps)

        # Always require at least basic execution capability
        if not capabilities:
            capabilities.add("execution")

        return capabilities

    def _create_subtasks_from_descriptions(self, descriptions: List[Any],
                                         parent_id: str, priority: int = 0) -> List[Task]:
        """
        Create Task objects from subtask descriptions.

        Args:
            descriptions: List of subtask descriptions or detailed info
            parent_id: Parent task ID
            priority: Base priority for subtasks

        Returns:
            List of Task objects
        """
        subtasks = []
        for i, desc_info in enumerate(descriptions):
            if isinstance(desc_info, dict):
                desc = desc_info.get('description', str(desc_info))
                task_priority = desc_info.get('priority', priority)
                capabilities = desc_info.get('capabilities', self._infer_capabilities(desc))
                due_time = desc_info.get('due_time')
                estimated_duration = desc_info.get('estimated_duration')
            else:
                desc = str(desc_info)
                task_priority = priority
                capabilities = self._infer_capabilities(desc)
                due_time = None
                estimated_duration = None

            task = Task(
                task_id=f"{parent_id}_refined_{i}",
                description=desc,
                required_capabilities=capabilities,
                parent_task=parent_id,
                priority=task_priority,
                due_time=due_time,
                estimated_duration=estimated_duration,
            )
            subtasks.append(task)

        return subtasks

    async def _log_planning_event_async(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Log planning event to episodic memory.

        Args:
            event_type: Type of planning event
            data: Event data
        """
        if self.episodic:
            try:
                await self.episodic.add_event_async(
                    event_type,
                    data,
                    metadata={"type": "planning_event", "event_type": event_type}
                )
            except Exception as e:
                logger.warning(f"[Kalki.Planner] Failed to log planning event: {e}")


class Scheduler:
    """Capability-based task scheduler with agent manager integration."""

    def __init__(self):
        """Initialize scheduler."""
        self.agent_capabilities: Dict[str, Set[str]] = {}
        self.agent_endpoints: Dict[str, str] = {}  # For live execution

    def register_agent(self, agent_id: str, capabilities: Set[str],
                      endpoint: Optional[str] = None) -> None:
        """
        Register an agent with its capabilities and execution endpoint.

        Args:
            agent_id: Agent identifier
            capabilities: Set of capability tags
            endpoint: Optional execution endpoint for live task assignment
        """
        self.agent_capabilities[agent_id] = capabilities
        if endpoint:
            self.agent_endpoints[agent_id] = endpoint

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
        Assign all ready tasks to agents with temporal/priority ordering.

        Args:
            graph: Task graph

        Returns:
            Dictionary mapping task_id to agent_id
        """
        assignments = {}

        ready_tasks = graph.get_ready_tasks(prioritize_temporal=True)

        for task in ready_tasks:
            agent_id = self.assign_task(task)
            if agent_id:
                assignments[task.task_id] = agent_id

        return assignments

    async def execute_task_live(self, task: Task) -> Dict[str, Any]:
        """
        Execute a task live via agent endpoint (stub for future implementation).

        Args:
            task: Task to execute

        Returns:
            Execution result
        """
        if not task.assigned_agent or task.assigned_agent not in self.agent_endpoints:
            return {"status": "error", "error": "No execution endpoint available"}

        # Stub for live execution - would integrate with agent manager
        logger.info(f"[Kalki.Planner] Would execute task {task.task_id} via {task.assigned_agent}")

        return {
            "status": "completed",
            "result": f"Task {task.task_id} executed via {task.assigned_agent}",
            "execution_time": 1.0
        }

# [Kalki v2.3 — planner/planner.py v1.0]