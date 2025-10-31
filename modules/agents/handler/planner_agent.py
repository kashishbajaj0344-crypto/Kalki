"""
Planner Agent - Creates and manages task plans with reasoning and memory integration
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from modules.agents.base_agent import BaseAgent, AgentCapability
from modules.config import get_config, CONFIG


class PlannerAgent(BaseAgent):
    """
    Creates detailed task plans with subtasks, dependencies, and resource requirements
    Enhanced with async execution, persistence, and comprehensive rule-based reasoning
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="PlannerAgent",
            capabilities=[AgentCapability.PLANNING, AgentCapability.REASONING],
            description="Creates detailed task plans with comprehensive rule-based reasoning",
            config=config
        )
        self.plans = {}
        self.planning_history = []

        # Persistence setup
        self.data_dir = Path(CONFIG.get("data_dir", "data")) / "plans"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Load existing plans
        self._load_plans()

    def _load_plans(self):
        """Load persisted plans from disk"""
        try:
            plans_file = self.data_dir / "plans.json"
            if plans_file.exists():
                with open(plans_file, 'r') as f:
                    data = json.load(f)
                    self.plans = data.get("plans", {})
                    self.planning_history = data.get("history", [])
                self.logger.info(f"Loaded {len(self.plans)} plans from disk")
        except Exception as e:
            self.logger.exception(f"Failed to load plans: {e}")

    def _save_plans(self):
        """Persist plans to disk"""
        try:
            plans_file = self.data_dir / "plans.json"
            data = {
                "plans": self.plans,
                "history": self.planning_history[-100:],  # Keep last 100 entries
                "last_updated": datetime.now().isoformat()
            }
            with open(plans_file, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.debug("Saved plans to disk")
        except Exception as e:
            self.logger.exception(f"Failed to save plans: {e}")

    async def create_plan(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a detailed plan with subtasks and dependencies using comprehensive rule-based reasoning
        """
        try:
            plan_id = f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Use comprehensive rule-based reasoning
            reasoning = await self._reason_comprehensively(task_description, context)

            # Create plan structure
            plan = {
                "id": plan_id,
                "description": task_description,
                "reasoning": reasoning,
                "subtasks": [],
                "dependencies": {},
                "resource_requirements": {},
                "estimated_duration": 0,
                "priority": context.get("priority", "normal") if context else "normal",
                "created_at": datetime.now().isoformat(),
                "status": "created"
            }

            # Generate subtasks based on reasoning
            subtasks = await self._decompose_task_comprehensively(task_description, reasoning, context)
            plan["subtasks"] = subtasks

            # Calculate dependencies and resources
            plan["dependencies"] = self._analyze_dependencies(subtasks)
            plan["resource_requirements"] = self._estimate_resources(subtasks)
            plan["estimated_duration"] = sum(subtask.get("estimated_duration", 0) for subtask in subtasks)

            self.plans[plan_id] = plan
            self.planning_history.append({
                "plan_id": plan_id,
                "action": "created",
                "timestamp": datetime.now().isoformat()
            })

            # Persist changes
            self._save_plans()

            self.logger.info(f"Created plan {plan_id} with {len(subtasks)} subtasks")
            return plan

        except Exception as e:
            self.logger.exception(f"Failed to create plan: {e}")
            raise

    def get_plan(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a plan by its ID
        """
        return self.plans.get(plan_id)

    async def _reason_comprehensively(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Apply comprehensive reasoning to understand task requirements and constraints
        """
        reasoning_parts = []

        # Analyze task type and complexity
        task_type = self._classify_task_type(task_description)
        complexity = self._assess_complexity(task_description, context)

        reasoning_parts.append(f"Task classified as: {task_type} with {complexity} complexity")

        # Identify key components and requirements
        components = self._identify_components(task_description)
        if components:
            reasoning_parts.append(f"Key components identified: {', '.join(components)}")

        # Assess resource needs
        resource_needs = self._assess_resource_needs(task_description, task_type)
        if resource_needs:
            reasoning_parts.append(f"Resource requirements: {resource_needs}")

        # Consider dependencies and sequencing
        dependencies = self._identify_dependencies(task_description, components)
        if dependencies:
            reasoning_parts.append(f"Task dependencies: {dependencies}")

        # Generate execution strategy
        strategy = self._generate_execution_strategy(task_type, complexity, components)
        reasoning_parts.append(f"Execution strategy: {strategy}")

        return ". ".join(reasoning_parts)

    def _classify_task_type(self, task_description: str) -> str:
        """Classify the type of task"""
        desc_lower = task_description.lower()

        if any(word in desc_lower for word in ["ingest", "upload", "import", "load", "process file"]):
            return "data_ingestion"
        elif any(word in desc_lower for word in ["query", "search", "find", "retrieve", "ask"]):
            return "information_retrieval"
        elif any(word in desc_lower for word in ["analyze", "examine", "review", "assess"]):
            return "data_analysis"
        elif any(word in desc_lower for word in ["generate", "create", "build", "develop"]):
            return "content_generation"
        elif any(word in desc_lower for word in ["optimize", "improve", "enhance", "tune"]):
            return "optimization"
        elif any(word in desc_lower for word in ["monitor", "watch", "track", "observe"]):
            return "monitoring"
        else:
            return "general_task"

    def _assess_complexity(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Assess task complexity"""
        complexity_score = 0

        # Length-based complexity
        if len(task_description) > 200:
            complexity_score += 2
        elif len(task_description) > 100:
            complexity_score += 1

        # Keyword-based complexity
        complex_keywords = ["multiple", "complex", "advanced", "integrate", "coordinate", "optimize"]
        desc_lower = task_description.lower()
        complexity_score += sum(1 for keyword in complex_keywords if keyword in desc_lower)

        # Context-based complexity
        if context:
            if context.get("priority") == "high":
                complexity_score += 1
            if context.get("dependencies"):
                complexity_score += len(context["dependencies"])

        if complexity_score >= 4:
            return "high"
        elif complexity_score >= 2:
            return "medium"
        else:
            return "low"

    def _identify_components(self, task_description: str) -> List[str]:
        """Identify key components of the task"""
        components = []

        # Data-related components
        if any(word in task_description.lower() for word in ["pdf", "document", "file", "data"]):
            components.append("data_processing")

        # Agent-related components
        if any(word in task_description.lower() for word in ["agent", "coordinate", "workflow"]):
            components.append("agent_coordination")

        # Resource-related components
        if any(word in task_description.lower() for word in ["resource", "cpu", "memory", "optimize"]):
            components.append("resource_management")

        # Analysis components
        if any(word in task_description.lower() for word in ["analyze", "review", "assess"]):
            components.append("analysis")

        # Communication components
        if any(word in task_description.lower() for word in ["communicate", "notify", "report"]):
            components.append("communication")

        return components

    def _assess_resource_needs(self, task_description: str, task_type: str) -> str:
        """Assess resource requirements for the task"""
        needs = []

        # Task-type specific resource needs
        if task_type == "data_ingestion":
            needs.extend(["file_io", "cpu_processing", "memory_buffering"])
        elif task_type == "information_retrieval":
            needs.extend(["database_access", "cpu_search", "memory_caching"])
        elif task_type == "data_analysis":
            needs.extend(["cpu_computation", "memory_analysis", "storage_temp"])
        elif task_type == "content_generation":
            needs.extend(["cpu_generation", "memory_large", "storage_output"])
        elif task_type == "optimization":
            needs.extend(["cpu_optimization", "memory_monitoring", "storage_logs"])
        elif task_type == "monitoring":
            needs.extend(["cpu_light", "memory_light", "network_light"])

        # Additional needs based on keywords
        desc_lower = task_description.lower()
        if "large" in desc_lower or "big" in desc_lower:
            needs.append("memory_large")
        if "fast" in desc_lower or "quick" in desc_lower:
            needs.append("cpu_high_priority")
        if "parallel" in desc_lower or "concurrent" in desc_lower:
            needs.append("cpu_multi_core")

        return ", ".join(set(needs)) if needs else "standard_resources"

    def _identify_dependencies(self, task_description: str, components: List[str]) -> str:
        """Identify task dependencies"""
        dependencies = []

        if "data_processing" in components:
            dependencies.append("input_validation")
        if "agent_coordination" in components:
            dependencies.append("agent_initialization")
        if "resource_management" in components:
            dependencies.append("resource_monitoring")
        if "analysis" in components:
            dependencies.append("data_availability")
        if "communication" in components:
            dependencies.append("network_connectivity")

        return ", ".join(dependencies) if dependencies else "minimal_dependencies"

    def _generate_execution_strategy(self, task_type: str, complexity: str, components: List[str]) -> str:
        """Generate execution strategy"""
        if complexity == "high":
            strategy = "parallel_execution_with_coordination"
        elif complexity == "medium":
            strategy = "sequential_execution_with_monitoring"
        else:
            strategy = "simple_sequential_execution"

        if "agent_coordination" in components:
            strategy += "_agent_orchestration"
        if "resource_management" in components:
            strategy += "_resource_optimization"

        return strategy

    async def _decompose_task_comprehensively(self, task_description: str, reasoning: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Decompose task into detailed, executable subtasks based on comprehensive analysis
        """
        task_type = self._classify_task_type(task_description)
        components = self._identify_components(task_description)

        subtasks = []

        # Generate subtasks based on task type and components
        if task_type == "data_ingestion":
            subtasks = self._generate_ingestion_subtasks(task_description, components)
        elif task_type == "information_retrieval":
            subtasks = self._generate_retrieval_subtasks(task_description, components)
        elif task_type == "data_analysis":
            subtasks = self._generate_analysis_subtasks(task_description, components)
        elif task_type == "content_generation":
            subtasks = self._generate_generation_subtasks(task_description, components)
        elif task_type == "optimization":
            subtasks = self._generate_optimization_subtasks(task_description, components)
        elif task_type == "monitoring":
            subtasks = self._generate_monitoring_subtasks(task_description, components)
        else:
            subtasks = self._generate_general_subtasks(task_description, components)

        # Add timing estimates and resource requirements
        for i, subtask in enumerate(subtasks):
            subtask["estimated_duration"] = self._estimate_subtask_duration(subtask, task_type)
            subtask["resource_requirements"] = self._estimate_subtask_resources(subtask, task_type)

        return subtasks

    def _generate_ingestion_subtasks(self, task_description: str, components: List[str]) -> List[Dict[str, Any]]:
        """Generate subtasks for data ingestion tasks"""
        subtasks = [
            {
                "id": "validate_input",
                "description": "Validate input files and data formats",
                "action": "validate",
                "dependencies": [],
                "priority": "high"
            },
            {
                "id": "prepare_processing",
                "description": "Set up processing environment and resources",
                "action": "prepare",
                "dependencies": ["validate_input"],
                "priority": "medium"
            },
            {
                "id": "extract_content",
                "description": "Extract content from input files",
                "action": "extract",
                "dependencies": ["prepare_processing"],
                "priority": "high"
            },
            {
                "id": "process_metadata",
                "description": "Process and store metadata",
                "action": "process_metadata",
                "dependencies": ["extract_content"],
                "priority": "medium"
            },
            {
                "id": "generate_embeddings",
                "description": "Generate vector embeddings for search",
                "action": "embed",
                "dependencies": ["extract_content"],
                "priority": "medium"
            },
            {
                "id": "store_results",
                "description": "Store processed data and update indices",
                "action": "store",
                "dependencies": ["process_metadata", "generate_embeddings"],
                "priority": "high"
            },
            {
                "id": "verify_ingestion",
                "description": "Verify successful ingestion and data integrity",
                "action": "verify",
                "dependencies": ["store_results"],
                "priority": "medium"
            }
        ]

        # Add agent coordination if needed
        if "agent_coordination" in components:
            subtasks.insert(1, {
                "id": "coordinate_agents",
                "description": "Coordinate with other agents for ingestion",
                "action": "coordinate",
                "dependencies": ["validate_input"],
                "priority": "medium"
            })

        return subtasks

    def _generate_retrieval_subtasks(self, task_description: str, components: List[str]) -> List[Dict[str, Any]]:
        """Generate subtasks for information retrieval tasks"""
        subtasks = [
            {
                "id": "parse_query",
                "description": "Parse and understand user query",
                "action": "parse",
                "dependencies": [],
                "priority": "high"
            },
            {
                "id": "prepare_search",
                "description": "Prepare search parameters and context",
                "action": "prepare_search",
                "dependencies": ["parse_query"],
                "priority": "medium"
            },
            {
                "id": "execute_search",
                "description": "Execute vector and text search",
                "action": "search",
                "dependencies": ["prepare_search"],
                "priority": "high"
            },
            {
                "id": "rank_results",
                "description": "Rank and filter search results",
                "action": "rank",
                "dependencies": ["execute_search"],
                "priority": "medium"
            },
            {
                "id": "format_response",
                "description": "Format results for presentation",
                "action": "format",
                "dependencies": ["rank_results"],
                "priority": "low"
            }
        ]

        return subtasks

    def _generate_analysis_subtasks(self, task_description: str, components: List[str]) -> List[Dict[str, Any]]:
        """Generate subtasks for data analysis tasks"""
        subtasks = [
            {
                "id": "load_data",
                "description": "Load relevant data for analysis",
                "action": "load",
                "dependencies": [],
                "priority": "high"
            },
            {
                "id": "preprocess_data",
                "description": "Preprocess and clean data",
                "action": "preprocess",
                "dependencies": ["load_data"],
                "priority": "medium"
            },
            {
                "id": "perform_analysis",
                "description": "Execute core analysis operations",
                "action": "analyze",
                "dependencies": ["preprocess_data"],
                "priority": "high"
            },
            {
                "id": "generate_insights",
                "description": "Generate insights and conclusions",
                "action": "generate_insights",
                "dependencies": ["perform_analysis"],
                "priority": "medium"
            },
            {
                "id": "create_report",
                "description": "Create analysis report",
                "action": "report",
                "dependencies": ["generate_insights"],
                "priority": "low"
            }
        ]

        return subtasks

    def _generate_generation_subtasks(self, task_description: str, components: List[str]) -> List[Dict[str, Any]]:
        """Generate subtasks for content generation tasks"""
        subtasks = [
            {
                "id": "analyze_requirements",
                "description": "Analyze generation requirements",
                "action": "analyze_requirements",
                "dependencies": [],
                "priority": "high"
            },
            {
                "id": "gather_context",
                "description": "Gather relevant context and data",
                "action": "gather_context",
                "dependencies": ["analyze_requirements"],
                "priority": "medium"
            },
            {
                "id": "generate_content",
                "description": "Generate primary content",
                "action": "generate",
                "dependencies": ["gather_context"],
                "priority": "high"
            },
            {
                "id": "refine_content",
                "description": "Refine and improve generated content",
                "action": "refine",
                "dependencies": ["generate_content"],
                "priority": "medium"
            },
            {
                "id": "validate_output",
                "description": "Validate generated content quality",
                "action": "validate_output",
                "dependencies": ["refine_content"],
                "priority": "medium"
            }
        ]

        return subtasks

    def _generate_optimization_subtasks(self, task_description: str, components: List[str]) -> List[Dict[str, Any]]:
        """Generate subtasks for optimization tasks"""
        subtasks = [
            {
                "id": "assess_current_state",
                "description": "Assess current system state and performance",
                "action": "assess",
                "dependencies": [],
                "priority": "high"
            },
            {
                "id": "identify_bottlenecks",
                "description": "Identify performance bottlenecks",
                "action": "identify_bottlenecks",
                "dependencies": ["assess_current_state"],
                "priority": "high"
            },
            {
                "id": "develop_optimization_plan",
                "description": "Develop detailed optimization plan",
                "action": "plan_optimization",
                "dependencies": ["identify_bottlenecks"],
                "priority": "medium"
            },
            {
                "id": "implement_optimizations",
                "description": "Implement identified optimizations",
                "action": "implement",
                "dependencies": ["develop_optimization_plan"],
                "priority": "high"
            },
            {
                "id": "measure_improvements",
                "description": "Measure and validate improvements",
                "action": "measure",
                "dependencies": ["implement_optimizations"],
                "priority": "medium"
            }
        ]

        return subtasks

    def _generate_monitoring_subtasks(self, task_description: str, components: List[str]) -> List[Dict[str, Any]]:
        """Generate subtasks for monitoring tasks"""
        subtasks = [
            {
                "id": "setup_monitoring",
                "description": "Set up monitoring infrastructure",
                "action": "setup_monitoring",
                "dependencies": [],
                "priority": "medium"
            },
            {
                "id": "define_metrics",
                "description": "Define monitoring metrics and thresholds",
                "action": "define_metrics",
                "dependencies": ["setup_monitoring"],
                "priority": "medium"
            },
            {
                "id": "collect_data",
                "description": "Continuously collect monitoring data",
                "action": "collect",
                "dependencies": ["define_metrics"],
                "priority": "high"
            },
            {
                "id": "analyze_trends",
                "description": "Analyze data trends and patterns",
                "action": "analyze_trends",
                "dependencies": ["collect_data"],
                "priority": "medium"
            },
            {
                "id": "generate_alerts",
                "description": "Generate alerts for threshold violations",
                "action": "alert",
                "dependencies": ["analyze_trends"],
                "priority": "high"
            }
        ]

        return subtasks

    def _generate_general_subtasks(self, task_description: str, components: List[str]) -> List[Dict[str, Any]]:
        """Generate subtasks for general tasks"""
        subtasks = [
            {
                "id": "analyze_task",
                "description": "Analyze task requirements and constraints",
                "action": "analyze",
                "dependencies": [],
                "priority": "high"
            },
            {
                "id": "prepare_resources",
                "description": "Prepare necessary resources and environment",
                "action": "prepare",
                "dependencies": ["analyze_task"],
                "priority": "medium"
            },
            {
                "id": "execute_task",
                "description": f"Execute main task: {task_description}",
                "action": "execute",
                "dependencies": ["prepare_resources"],
                "priority": "high"
            },
            {
                "id": "validate_results",
                "description": "Validate task execution results",
                "action": "validate",
                "dependencies": ["execute_task"],
                "priority": "medium"
            },
            {
                "id": "cleanup_resources",
                "description": "Clean up resources and temporary data",
                "action": "cleanup",
                "dependencies": ["validate_results"],
                "priority": "low"
            }
        ]

        return subtasks

    def _estimate_subtask_duration(self, subtask: Dict[str, Any], task_type: str) -> int:
        """Estimate duration for a subtask in minutes"""
        base_durations = {
            "validate": 5,
            "prepare": 10,
            "extract": 30,
            "process_metadata": 15,
            "embed": 45,
            "store": 20,
            "verify": 10,
            "parse": 5,
            "prepare_search": 10,
            "search": 25,
            "rank": 15,
            "format": 10,
            "load": 20,
            "preprocess": 25,
            "analyze": 40,
            "generate_insights": 30,
            "report": 20,
            "analyze_requirements": 15,
            "gather_context": 25,
            "generate": 60,
            "refine": 30,
            "validate_output": 15,
            "assess": 20,
            "identify_bottlenecks": 35,
            "plan_optimization": 25,
            "implement": 50,
            "measure": 20,
            "setup_monitoring": 15,
            "define_metrics": 20,
            "collect": 5,  # Ongoing
            "analyze_trends": 25,
            "alert": 5,
            "coordinate": 10,
            "cleanup": 5
        }

        action = subtask.get("action", "unknown")
        base_duration = base_durations.get(action, 30)

        # Adjust based on priority
        priority = subtask.get("priority", "medium")
        if priority == "high":
            base_duration = int(base_duration * 1.2)
        elif priority == "low":
            base_duration = int(base_duration * 0.8)

        # Adjust based on task type complexity
        if task_type in ["data_analysis", "optimization"]:
            base_duration = int(base_duration * 1.3)

        return max(base_duration, 5)  # Minimum 5 minutes

    def _analyze_dependencies(self, subtasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze dependencies between subtasks and detect potential issues
        """
        dependency_graph = {}
        all_subtask_ids = {subtask["id"] for subtask in subtasks}

        # Build dependency graph
        for subtask in subtasks:
            subtask_id = subtask["id"]
            dependencies = subtask.get("dependencies", [])

            # Validate dependencies exist
            invalid_deps = [dep for dep in dependencies if dep not in all_subtask_ids]
            if invalid_deps:
                raise ValueError(f"Subtask {subtask_id} has invalid dependencies: {invalid_deps}")

            dependency_graph[subtask_id] = {
                "depends_on": dependencies,
                "depended_by": []
            }

        # Build reverse dependencies
        for subtask_id, deps in dependency_graph.items():
            for dep in deps["depends_on"]:
                if dep in dependency_graph:
                    dependency_graph[dep]["depended_by"].append(subtask_id)

        # Detect cycles using DFS
        visited = set()
        rec_stack = set()

        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)

            for dep in dependency_graph[node]["depends_on"]:
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        cycles = []
        for subtask_id in dependency_graph:
            if subtask_id not in visited:
                if has_cycle(subtask_id):
                    cycles.append(subtask_id)

        # Calculate execution order (topological sort)
        execution_order = []
        temp_visited = set()

        def topological_sort(node):
            temp_visited.add(node)
            for dep in dependency_graph[node]["depends_on"]:
                if dep not in temp_visited:
                    topological_sort(dep)
            execution_order.append(node)

        # Only sort if no cycles
        if not cycles:
            for subtask_id in dependency_graph:
                if subtask_id not in temp_visited:
                    topological_sort(subtask_id)
            execution_order.reverse()

        return {
            "graph": dependency_graph,
            "cycles_detected": len(cycles) > 0,
            "cycle_nodes": cycles,
            "execution_order": execution_order if not cycles else [],
            "parallel_groups": self._identify_parallel_groups(dependency_graph)
        }

    def _identify_parallel_groups(self, dependency_graph: Dict[str, Any]) -> List[List[str]]:
        """Identify groups of subtasks that can be executed in parallel"""
        # Simple parallel group identification based on no dependencies between them
        processed = set()
        parallel_groups = []

        for subtask_id in dependency_graph:
            if subtask_id in processed:
                continue

            # Find subtasks with no dependencies that haven't been processed
            group = []
            for candidate_id in dependency_graph:
                if candidate_id not in processed:
                    # Check if this candidate can be added to current group
                    can_add = True
                    for existing in group:
                        # Check if they have dependencies on each other
                        if (candidate_id in dependency_graph[existing]["depends_on"] or
                            existing in dependency_graph[candidate_id]["depends_on"] or
                            candidate_id in dependency_graph[existing]["depended_by"] or
                            existing in dependency_graph[candidate_id]["depended_by"]):
                            can_add = False
                            break
                    if can_add:
                        group.append(candidate_id)

            if group:
                parallel_groups.append(group)
                processed.update(group)

        return parallel_groups

    def _estimate_resources(self, subtasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Estimate total resource requirements for all subtasks
        """
        total_resources = {
            "cpu_cores": 0,
            "memory_gb": 0.0,
            "max_concurrent_cpu": 0,
            "max_concurrent_memory": 0.0,
            "estimated_cost": 0.0
        }

        # Get parallel groups to estimate concurrent resource usage
        if subtasks:
            # Create a simple dependency graph for parallel analysis
            dep_graph = {}
            for subtask in subtasks:
                dep_graph[subtask["id"]] = {
                    "depends_on": subtask.get("dependencies", []),
                    "depended_by": []
                }

            # Build reverse dependencies
            for subtask_id, deps in dep_graph.items():
                for dep in deps["depends_on"]:
                    if dep in dep_graph:
                        dep_graph[dep]["depended_by"].append(subtask_id)

            parallel_groups = self._identify_parallel_groups(dep_graph)

            # Estimate peak concurrent resources
            for group in parallel_groups:
                group_cpu = 0
                group_memory = 0.0

                for subtask_id in group:
                    subtask = next(s for s in subtasks if s["id"] == subtask_id)
                    resources = subtask.get("resource_requirements", {})
                    group_cpu += resources.get("cpu_cores", 1)
                    group_memory += resources.get("memory_gb", 0.5)

                total_resources["max_concurrent_cpu"] = max(total_resources["max_concurrent_cpu"], group_cpu)
                total_resources["max_concurrent_memory"] = max(total_resources["max_concurrent_memory"], group_memory)

        # Sum all individual resource requirements
        for subtask in subtasks:
            resources = subtask.get("resource_requirements", {})
            total_resources["cpu_cores"] += resources.get("cpu_cores", 1)
            total_resources["memory_gb"] += resources.get("memory_gb", 0.5)

        # Estimate cost based on resource usage (simplified model)
        cpu_hours = total_resources["cpu_cores"] * 0.1  # Assume average 6 minutes per core
        memory_hours = total_resources["memory_gb"] * 0.1
        total_resources["estimated_cost"] = (cpu_hours * 0.05) + (memory_hours * 0.02)  # Cost per hour

        return total_resources

    def _estimate_subtask_resources(self, subtask: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """Estimate resource requirements for a subtask"""
        action = subtask.get("action", "unknown")

        # Base resource requirements by action type
        resource_map = {
            "validate": {"cpu_cores": 1, "memory_gb": 0.5},
            "prepare": {"cpu_cores": 1, "memory_gb": 1.0},
            "extract": {"cpu_cores": 2, "memory_gb": 2.0},
            "process_metadata": {"cpu_cores": 1, "memory_gb": 1.0},
            "embed": {"cpu_cores": 4, "memory_gb": 4.0},
            "store": {"cpu_cores": 2, "memory_gb": 1.5},
            "verify": {"cpu_cores": 1, "memory_gb": 0.5},
            "parse": {"cpu_cores": 1, "memory_gb": 0.5},
            "prepare_search": {"cpu_cores": 1, "memory_gb": 1.0},
            "search": {"cpu_cores": 2, "memory_gb": 2.0},
            "rank": {"cpu_cores": 1, "memory_gb": 1.0},
            "format": {"cpu_cores": 1, "memory_gb": 0.5},
            "load": {"cpu_cores": 2, "memory_gb": 2.0},
            "preprocess": {"cpu_cores": 2, "memory_gb": 3.0},
            "analyze": {"cpu_cores": 4, "memory_gb": 4.0},
            "generate_insights": {"cpu_cores": 2, "memory_gb": 2.0},
            "report": {"cpu_cores": 1, "memory_gb": 1.0},
            "analyze_requirements": {"cpu_cores": 1, "memory_gb": 1.0},
            "gather_context": {"cpu_cores": 2, "memory_gb": 2.0},
            "generate": {"cpu_cores": 4, "memory_gb": 6.0},
            "refine": {"cpu_cores": 2, "memory_gb": 3.0},
            "validate_output": {"cpu_cores": 1, "memory_gb": 1.0},
            "assess": {"cpu_cores": 2, "memory_gb": 2.0},
            "identify_bottlenecks": {"cpu_cores": 4, "memory_gb": 4.0},
            "plan_optimization": {"cpu_cores": 2, "memory_gb": 2.0},
            "implement": {"cpu_cores": 4, "memory_gb": 4.0},
            "measure": {"cpu_cores": 2, "memory_gb": 2.0},
            "setup_monitoring": {"cpu_cores": 1, "memory_gb": 1.0},
            "define_metrics": {"cpu_cores": 1, "memory_gb": 1.0},
            "collect": {"cpu_cores": 1, "memory_gb": 0.5},
            "analyze_trends": {"cpu_cores": 2, "memory_gb": 2.0},
            "alert": {"cpu_cores": 1, "memory_gb": 0.5},
            "coordinate": {"cpu_cores": 1, "memory_gb": 1.0},
            "cleanup": {"cpu_cores": 1, "memory_gb": 0.5}
        }

        base_resources = resource_map.get(action, {"cpu_cores": 1, "memory_gb": 1.0})

        # Adjust for task type
        if task_type in ["data_analysis", "optimization", "content_generation"]:
            base_resources["cpu_cores"] = min(base_resources["cpu_cores"] + 1, 8)
            base_resources["memory_gb"] = min(base_resources["memory_gb"] * 1.5, 16.0)

        return base_resources

    async def initialize(self) -> bool:
        """
        Initialize the planner agent
        """
        try:
            # Load existing plans
            self._load_plans()
            self.logger.info("PlannerAgent initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"Failed to initialize PlannerAgent: {e}")
            return False

    async def shutdown(self) -> bool:
        """
        Gracefully shutdown the planner agent
        """
        try:
            # Save current plans
            self._save_plans()
            self.logger.info("PlannerAgent shutdown successfully")
            return True
        except Exception as e:
            self.logger.exception(f"Failed to shutdown PlannerAgent: {e}")
            return False

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Async execute planning tasks"""
        try:
            action = task.get("action")

            if action == "create_plan":
                plan = await self.create_plan(task["description"], task.get("context"))
                return {"status": "success", "plan": plan}
            elif action == "get_plan":
                plan = self.get_plan(task["plan_id"])
                if plan:
                    return {"status": "success", "plan": plan}
                else:
                    return {"status": "error", "message": f"Plan {task['plan_id']} not found"}
            elif action == "list_plans":
                plans = self.list_plans()
                return {"status": "success", "plans": plans}
            elif action == "update_status":
                await self.update_plan_status(task["plan_id"], task["status"])
                return {"status": "success"}
            else:
                return {"status": "error", "message": f"Unknown action: {action}"}

        except Exception as e:
            self.logger.exception(f"Failed to execute planning task: {e}")
            return {"status": "error", "message": str(e)}