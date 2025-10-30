from typing import Dict, Any
import logging

from ..base_agent import BaseAgent, AgentCapability, AgentStatus


class ReasoningAgent(BaseAgent):
    """Multi-step reasoning and inference agent"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="ReasoningAgent",
            capabilities=[AgentCapability.REASONING],
            description="Performs multi-step reasoning and logical inference",
            dependencies=["SearchAgent"],
            config=config or {}
        )
        self.logger = logging.getLogger("kalki.agent.ReasoningAgent")

    async def initialize(self) -> bool:
        try:
            from modules import llm
            self.llm = llm
            self.logger.info(f"{self.name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"Failed to initialize {self.name}: {e}")
            return False

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        action = task.get("action")
        params = task.get("params", {})

        if action == "reason":
            return await self._reason(params)
        else:
            return {"status": "error", "error": f"Unknown action: {action}"}

    async def _reason(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            query = params.get("query", "")
            steps = params.get("steps", 1)
            
            # Use LLM for reasoning
            answer = self.llm.ask_kalki(query)
            
            return {
                "status": "success",
                "query": query,
                "reasoning_steps": steps,
                "answer": answer
            }
            
        except Exception as e:
            self.logger.exception(f"Reasoning error: {e}")
            return {"status": "error", "error": str(e)}

    async def shutdown(self) -> bool:
        self.logger.info(f"{self.name} shutting down")
        self.update_status(AgentStatus.TERMINATED)
        return True
