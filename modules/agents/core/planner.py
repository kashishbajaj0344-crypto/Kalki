from typing import Dict, Any
import logging

from ..base_agent import BaseAgent, AgentCapability, AgentStatus


class PlannerAgent(BaseAgent):
    """Task planning and decomposition agent"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="PlannerAgent",
            capabilities=[AgentCapability.PLANNING],
            description="Decomposes complex tasks into executable sub-tasks",
            config=config or {}
        )
        self.logger = logging.getLogger("kalki.agent.PlannerAgent")

    async def initialize(self) -> bool:
        try:
            self.logger.info(f"{self.name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"Failed to initialize {self.name}: {e}")
            return False

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        action = task.get("action")
        params = task.get("params", {})

        if action == "plan":
            return await self._create_plan(params)
        else:
            return {
                "status": "error",
                "error": f"Unknown action: {action}"
            }

    async def _create_plan(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            goal = params.get("goal", "")
            constraints = params.get("constraints", {})
            
            steps = [
                {"step": 1, "action": "analyze_goal", "description": f"Analyze: {goal}"},
                {"step": 2, "action": "gather_resources", "description": "Gather required resources"},
                {"step": 3, "action": "execute_plan", "description": "Execute planned actions"},
                {"step": 4, "action": "validate_results", "description": "Validate outcomes"}
            ]
            
            return {
                "status": "success",
                "goal": goal,
                "plan": steps,
                "constraints": constraints
            }
            
        except Exception as e:
            self.logger.exception(f"Planning error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def shutdown(self) -> bool:
        self.logger.info(f"{self.name} shutting down")
        self.update_status(AgentStatus.TERMINATED)
        return True
