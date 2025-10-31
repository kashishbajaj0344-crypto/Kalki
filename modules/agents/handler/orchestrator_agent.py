"""
Orchestrator Agent - Coordinates workflow execution across multiple agents
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from modules.agents.base_agent import BaseAgent, AgentCapability
from modules.config import get_config, CONFIG


class OrchestratorAgent(BaseAgent):
    """
    Coordinates workflow execution by routing tasks to appropriate agents
    Enhanced with async execution, persistence, and improved agent routing
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="OrchestratorAgent",
            capabilities=[AgentCapability.ORCHESTRATION, AgentCapability.MEMORY],
            description="Coordinates workflow execution across multiple agents",
            config=config
        )
        self.workflows = {}
        self.registered_agents = {}

        # Persistence setup
        self.data_dir = Path(CONFIG.get("data_dir", "data")) / "workflows"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Load existing workflows
        self._load_workflows()

    def _load_workflows(self):
        """Load persisted workflows from disk"""
        try:
            workflows_file = self.data_dir / "workflows.json"
            if workflows_file.exists():
                with open(workflows_file, 'r') as f:
                    data = json.load(f)
                    self.workflows = data.get("workflows", {})
                self.logger.info(f"Loaded {len(self.workflows)} workflows from disk")
        except Exception as e:
            self.logger.exception(f"Failed to load workflows: {e}")

    def _save_workflows(self):
        """Persist workflows to disk"""
        try:
            workflows_file = self.data_dir / "workflows.json"
            data = {
                "workflows": self.workflows,
                "last_updated": datetime.now().isoformat()
            }
            with open(workflows_file, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.debug("Saved workflows to disk")
        except Exception as e:
            self.logger.exception(f"Failed to save workflows: {e}")

    def register_agent(self, agent_info: Dict[str, Any]):
        """Register an agent for workflow routing"""
        agent_name = agent_info.get("name")
        capabilities = agent_info.get("capabilities", [])
        agent_instance = agent_info.get("instance")

        self.registered_agents[agent_name] = {
            "capabilities": capabilities,
            "instance": agent_instance,
            "registered_at": datetime.now().isoformat()
        }

        self.logger.info(f"Registered agent: {agent_name} with capabilities: {capabilities}")

    async def create_workflow(self, workflow_id: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a workflow from a plan
        """
        try:
            workflow = {
                "id": workflow_id,
                "plan": plan,
                "status": "created",
                "results": {},
                "created_at": datetime.now().isoformat(),
                "progress": 0.0
            }

            self.workflows[workflow_id] = workflow
            self._save_workflows()

            self.logger.info(f"Created workflow {workflow_id}")
            return workflow

        except Exception as e:
            self.logger.exception(f"Failed to create workflow: {e}")
            raise

    async def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """
        Execute a workflow by coordinating agents asynchronously
        """
        try:
            workflow = self.workflows.get(workflow_id)
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")

            workflow["status"] = "running"
            plan = workflow["plan"]
            subtasks = plan.get("subtasks", [])

            # Execute subtasks in dependency order with concurrency where possible
            completed_tasks = set()
            pending_tasks = {subtask["id"]: subtask for subtask in subtasks}

            while pending_tasks:
                # Find tasks whose dependencies are met
                executable_tasks = []
                for task_id, subtask in pending_tasks.items():
                    deps = subtask.get("dependencies", [])
                    if all(dep in completed_tasks for dep in deps):
                        executable_tasks.append((task_id, subtask))

                if not executable_tasks:
                    raise ValueError(f"Deadlock detected in workflow {workflow_id}")

                # Execute executable tasks concurrently
                execution_tasks = []
                for task_id, subtask in executable_tasks:
                    task = asyncio.create_task(self._execute_subtask_async(subtask, workflow))
                    execution_tasks.append((task_id, task))

                # Wait for all concurrent tasks to complete
                results = await asyncio.gather(*[task for _, task in execution_tasks], return_exceptions=True)

                # Process results
                for (task_id, _), result in zip(execution_tasks, results):
                    if isinstance(result, Exception):
                        workflow["results"][task_id] = {
                            "status": "failed",
                            "error": str(result),
                            "timestamp": datetime.now().isoformat()
                        }
                    else:
                        workflow["results"][task_id] = result
                        completed_tasks.add(task_id)

                    del pending_tasks[task_id]

                # Update progress
                total_tasks = len(subtasks)
                completed_count = len(completed_tasks)
                workflow["progress"] = (completed_count / total_tasks) * 100

            workflow["status"] = "completed"
            workflow["completed_at"] = datetime.now().isoformat()
            self._save_workflows()

            self.logger.info(f"Completed workflow {workflow_id}")
            return workflow

        except Exception as e:
            self.logger.exception(f"Failed to execute workflow: {e}")
            if workflow_id in self.workflows:
                self.workflows[workflow_id]["status"] = "failed"
                self._save_workflows()
            raise

    async def _execute_subtask_async(self, subtask: Dict[str, Any], workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single subtask asynchronously"""
        try:
            action = subtask.get("action")

            # Route to appropriate agent based on action and capabilities
            target_agent = await self._route_to_agent(action, subtask)

            if target_agent:
                # Execute via agent
                task_payload = {
                    "action": action,
                    "subtask": subtask,
                    "workflow_context": workflow
                }
                result = await target_agent.execute(task_payload)
            else:
                # Fallback: execute directly
                result = await self._execute_subtask_direct(subtask)

            result.update({
                "subtask_id": subtask["id"],
                "action": action,
                "timestamp": datetime.now().isoformat()
            })

            self.logger.debug(f"Executed subtask {subtask['id']}: {action}")
            return result

        except Exception as e:
            self.logger.exception(f"Subtask execution failed: {e}")
            return {
                "subtask_id": subtask["id"],
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _route_to_agent(self, action: str, subtask: Dict[str, Any]) -> Optional[Any]:
        """Route subtask to appropriate agent based on capabilities"""
        # Simple routing logic - can be enhanced
        routing_map = {
            "validate": ["ValidationAgent", "SafetyAgent"],
            "extract": ["MultimodalAgent", "CognitiveAgent"],
            "embed": ["MultimodalAgent"],
            "parse": ["CognitiveAgent", "ReasoningAgent"],
            "search": ["MultimodalAgent"],
            "rank": ["ReasoningAgent"],
            "analyze": ["CognitiveAgent", "ReasoningAgent"],
            "execute": ["CoreAgent"],
        }

        candidate_agents = routing_map.get(action, [])

        for agent_name in candidate_agents:
            if agent_name in self.registered_agents:
                agent_info = self.registered_agents[agent_name]
                # Check if agent has required capabilities
                required_caps = subtask.get("required_capabilities", [])
                agent_caps = agent_info.get("capabilities", [])

                if not required_caps or any(cap in agent_caps for cap in required_caps):
                    return agent_info.get("instance")

        return None

    async def _execute_subtask_direct(self, subtask: Dict[str, Any]) -> Dict[str, Any]:
        """Execute subtask directly when no agent is available"""
        # Simulate execution with delay
        action = subtask.get("action", "unknown")
        await asyncio.sleep(0.1)  # Simulate processing time

        return {
            "status": "success",
            "output": f"Executed {action} directly",
            "method": "direct_execution"
        }

    def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a workflow by ID"""
        return self.workflows.get(workflow_id)

    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all workflows"""
        return list(self.workflows.values())

    async def update_workflow_status(self, workflow_id: str, status: str):
        """Update workflow status"""
        if workflow_id in self.workflows:
            self.workflows[workflow_id]["status"] = status
            self.workflows[workflow_id]["updated_at"] = datetime.now().isoformat()
            self._save_workflows()
            self.logger.info(f"Updated workflow {workflow_id} status to {status}")

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Async execute orchestration tasks"""
        try:
            action = task.get("action")

            if action == "register":
                self.register_agent(task["agent"])
                return {"status": "success"}
            elif action == "create_workflow":
                workflow = await self.create_workflow(task["workflow_id"], task["plan"])
                return {"status": "success", "workflow": workflow}
            elif action == "execute_workflow":
                workflow = await self.execute_workflow(task["workflow_id"])
                return {"status": "success", "workflow": workflow}
            elif action == "get_workflow":
                workflow = self.get_workflow(task["workflow_id"])
                if workflow:
                    return {"status": "success", "workflow": workflow}
                else:
                    return {"status": "error", "message": f"Workflow {task['workflow_id']} not found"}
            elif action == "list_workflows":
                workflows = self.list_workflows()
                return {"status": "success", "workflows": workflows}
            elif action == "update_status":
                await self.update_workflow_status(task["workflow_id"], task["status"])
                return {"status": "success"}
            else:
                return {"status": "error", "message": f"Unknown action: {action}"}

        except Exception as e:
            self.logger.exception(f"Failed to execute orchestration task: {e}")
            return {"status": "error", "message": str(e)}

    async def initialize(self) -> bool:
        """
        Initialize the orchestrator agent
        """
        try:
            # Load existing workflows
            self._load_workflows()
            self.logger.info("OrchestratorAgent initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"Failed to initialize OrchestratorAgent: {e}")
            return False

    async def shutdown(self) -> bool:
        """
        Gracefully shutdown the orchestrator agent
        """
        try:
            # Save current workflows
            self._save_workflows()
            self.logger.info("OrchestratorAgent shutdown successfully")
            return True
        except Exception as e:
            self.logger.exception(f"Failed to shutdown OrchestratorAgent: {e}")
            return False