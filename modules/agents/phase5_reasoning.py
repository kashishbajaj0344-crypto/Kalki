#!/usr/bin/env python3
"""
Phase 5: Reasoning, Planning & Multi-Agent Chaining
- PlannerAgent: Task decomposition and planning
- OrchestratorAgent: Multi-agent coordination
- ComputeOptimizerAgent: Resource allocation
- CopilotAgent: Interactive assistance
"""
import logging
import psutil
from typing import Dict, Any, Optional, List
from datetime import datetime
from .base_agent import BaseAgent

logger = logging.getLogger("kalki.agents.phase5")


class PlannerAgent(BaseAgent):
    """
    Decomposes complex tasks into subtasks with dependencies
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="PlannerAgent", config=config)
        self.plans = {}
    
    def create_plan(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create an execution plan for a task
        """
        try:
            plan_id = f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # Simple task decomposition (can be enhanced with LLM)
            subtasks = self._decompose_task(task, context)
            
            plan = {
                "plan_id": plan_id,
                "task": task,
                "context": context or {},
                "subtasks": subtasks,
                "status": "pending",
                "created_at": datetime.now().isoformat()
            }
            
            self.plans[plan_id] = plan
            self.logger.info(f"Created plan {plan_id} with {len(subtasks)} subtasks")
            return plan
        except Exception as e:
            self.logger.exception(f"Failed to create plan: {e}")
            raise
    
    def _decompose_task(self, task: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Decompose task into subtasks
        Can be enhanced with LLM-based decomposition
        """
        # Basic heuristic decomposition
        subtasks = []
        
        # Check if task involves ingestion
        if "ingest" in task.lower() or "pdf" in task.lower():
            subtasks.append({
                "id": "subtask_1",
                "action": "validate_input",
                "description": "Validate input files/paths",
                "dependencies": []
            })
            subtasks.append({
                "id": "subtask_2",
                "action": "ingest",
                "description": "Ingest documents",
                "dependencies": ["subtask_1"]
            })
        
        # Check if task involves query
        if "query" in task.lower() or "ask" in task.lower() or "search" in task.lower():
            subtasks.append({
                "id": "subtask_1",
                "action": "retrieve_context",
                "description": "Retrieve relevant context",
                "dependencies": []
            })
            subtasks.append({
                "id": "subtask_2",
                "action": "generate_response",
                "description": "Generate answer",
                "dependencies": ["subtask_1"]
            })
        
        # Default: single subtask
        if not subtasks:
            subtasks.append({
                "id": "subtask_1",
                "action": "execute",
                "description": task,
                "dependencies": []
            })
        
        return subtasks
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute planning tasks"""
        action = task.get("action")
        
        if action == "create_plan":
            plan = self.create_plan(task["task"], task.get("context"))
            return {"status": "success", "plan": plan}
        elif action == "get_plan":
            plan = self.plans.get(task["plan_id"])
            return {"status": "success", "plan": plan}
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}


class OrchestratorAgent(BaseAgent):
    """
    Coordinates multiple agents for complex workflows
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="OrchestratorAgent", config=config)
        self.agents_registry = {}
        self.workflows = {}
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent for orchestration"""
        self.agents_registry[agent.name] = agent
        self.logger.info(f"Registered agent: {agent.name}")
    
    def create_workflow(self, workflow_id: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a workflow from a plan
        """
        try:
            workflow = {
                "workflow_id": workflow_id,
                "plan": plan,
                "status": "created",
                "results": {},
                "created_at": datetime.now().isoformat()
            }
            self.workflows[workflow_id] = workflow
            self.logger.info(f"Created workflow {workflow_id}")
            return workflow
        except Exception as e:
            self.logger.exception(f"Failed to create workflow: {e}")
            raise
    
    def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """
        Execute a workflow by coordinating agents
        """
        try:
            workflow = self.workflows.get(workflow_id)
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            workflow["status"] = "running"
            plan = workflow["plan"]
            subtasks = plan.get("subtasks", [])
            
            # Execute subtasks in dependency order
            for subtask in subtasks:
                # Check dependencies
                deps_met = all(
                    workflow["results"].get(dep, {}).get("status") == "success"
                    for dep in subtask.get("dependencies", [])
                )
                
                if deps_met:
                    result = self._execute_subtask(subtask, workflow)
                    workflow["results"][subtask["id"]] = result
            
            workflow["status"] = "completed"
            self.logger.info(f"Completed workflow {workflow_id}")
            return workflow
        except Exception as e:
            self.logger.exception(f"Failed to execute workflow: {e}")
            if workflow_id in self.workflows:
                self.workflows[workflow_id]["status"] = "failed"
            raise
    
    def _execute_subtask(self, subtask: Dict[str, Any], workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single subtask"""
        try:
            action = subtask.get("action")
            # Route to appropriate agent based on action
            # This is a simplified routing; can be enhanced
            
            result = {
                "subtask_id": subtask["id"],
                "action": action,
                "status": "success",
                "output": f"Executed {action}",
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.debug(f"Executed subtask {subtask['id']}: {action}")
            return result
        except Exception as e:
            self.logger.exception(f"Subtask execution failed: {e}")
            return {
                "subtask_id": subtask["id"],
                "status": "failed",
                "error": str(e)
            }
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute orchestration tasks"""
        action = task.get("action")
        
        if action == "register":
            self.register_agent(task["agent"])
            return {"status": "success"}
        elif action == "create_workflow":
            workflow = self.create_workflow(task["workflow_id"], task["plan"])
            return {"status": "success", "workflow": workflow}
        elif action == "execute_workflow":
            workflow = self.execute_workflow(task["workflow_id"])
            return {"status": "success", "workflow": workflow}
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}


class ComputeOptimizerAgent(BaseAgent):
    """
    Dynamically allocates CPU/GPU/memory resources
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="ComputeOptimizerAgent", config=config)
        self.resource_allocations = {}
    
    def get_system_resources(self) -> Dict[str, Any]:
        """Get current system resource utilization"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            resources = {
                "cpu_percent": cpu_percent,
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": memory.total / (1024**3),
                "memory_available_gb": memory.available / (1024**3),
                "memory_percent": memory.percent,
                "disk_total_gb": disk.total / (1024**3),
                "disk_free_gb": disk.free / (1024**3),
                "disk_percent": disk.percent
            }
            
            return resources
        except Exception as e:
            self.logger.exception(f"Failed to get system resources: {e}")
            return {}
    
    def allocate_resources(self, task_id: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Allocate resources for a task
        """
        try:
            current_resources = self.get_system_resources()
            
            # Simple allocation strategy
            allocation = {
                "task_id": task_id,
                "cpu_cores": min(requirements.get("cpu_cores", 1), current_resources.get("cpu_count", 1)),
                "memory_gb": min(requirements.get("memory_gb", 1), current_resources.get("memory_available_gb", 1)),
                "priority": requirements.get("priority", "normal"),
                "allocated_at": datetime.now().isoformat()
            }
            
            self.resource_allocations[task_id] = allocation
            self.logger.info(f"Allocated resources for task {task_id}: {allocation}")
            return allocation
        except Exception as e:
            self.logger.exception(f"Failed to allocate resources: {e}")
            raise
    
    def release_resources(self, task_id: str):
        """Release resources allocated to a task"""
        if task_id in self.resource_allocations:
            del self.resource_allocations[task_id]
            self.logger.info(f"Released resources for task {task_id}")
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute compute optimization tasks"""
        action = task.get("action")
        
        if action == "get_resources":
            resources = self.get_system_resources()
            return {"status": "success", "resources": resources}
        elif action == "allocate":
            allocation = self.allocate_resources(task["task_id"], task["requirements"])
            return {"status": "success", "allocation": allocation}
        elif action == "release":
            self.release_resources(task["task_id"])
            return {"status": "success"}
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}


class CopilotAgent(BaseAgent):
    """
    Interactive assistance agent for user guidance
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="CopilotAgent", config=config)
        self.conversation_history = []
    
    def assist(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Provide interactive assistance
        """
        try:
            # Store in conversation history
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "user_input": user_input,
                "context": context or {}
            })
            
            # Generate assistance (simplified; can be enhanced with LLM)
            assistance = self._generate_assistance(user_input, context)
            
            self.conversation_history[-1]["assistance"] = assistance
            return assistance
        except Exception as e:
            self.logger.exception(f"Failed to provide assistance: {e}")
            return f"Error: {e}"
    
    def _generate_assistance(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate assistance response
        """
        # Simple keyword-based assistance (can be enhanced with LLM)
        lower_input = user_input.lower()
        
        if "help" in lower_input:
            return "I can help you with: querying documents, ingesting PDFs, managing sessions, and more. What would you like to do?"
        elif "ingest" in lower_input or "pdf" in lower_input:
            return "To ingest PDFs, you can use the ingest command or drag and drop files in the GUI. Would you like me to guide you through the process?"
        elif "query" in lower_input or "ask" in lower_input:
            return "To query the knowledge base, simply ask your question and I'll search for relevant information."
        else:
            return f"I understand you want to: {user_input}. Let me help you with that."
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute copilot tasks"""
        action = task.get("action")
        
        if action == "assist":
            assistance = self.assist(task["user_input"], task.get("context"))
            return {"status": "success", "assistance": assistance}
        elif action == "history":
            return {"status": "success", "history": self.conversation_history}
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}
