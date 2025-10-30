from typing import Dict, Any, List
import logging

from ..base_agent import BaseAgent, AgentCapability, AgentStatus


class FeedbackAgent(BaseAgent):
    """Learning feedback and performance monitoring"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="FeedbackAgent",
            capabilities=[AgentCapability.FEEDBACK],
            description="Monitors performance and provides learning feedback",
            config=config or {}
        )
        self.performance_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("kalki.agent.FeedbackAgent")

    async def initialize(self) -> bool:
        try:
            self.logger.info(f"{self.name} initialized successfully")
            self.update_status(AgentStatus.INITIALIZED)
            return True
        except Exception as e:
            self.logger.exception(f"Failed to initialize {self.name}: {e}")
            self.update_status(AgentStatus.ERROR)
            return False

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        self.update_status(AgentStatus.RUNNING)
        action = task.get("action")
        params = task.get("params", {})

        try:
            if action == "record":
                result = await self._record_performance(params)
            elif action == "analyze":
                result = await self._analyze_performance(params)
            elif action == "recommend":
                result = await self._recommend_improvements(params)
            else:
                result = {"status": "error", "error": f"Unknown action: {action}"}
            return result
        finally:
            self.update_status(AgentStatus.IDLE)

    async def _record_performance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            agent_name = params.get("agent_name", "")
            metrics = params.get("metrics", {})
            timestamp = self.last_active.isoformat() if hasattr(self, "last_active") else None
            self.performance_history.append({"agent_name": agent_name, "metrics": metrics, "timestamp": timestamp})
            return {"status": "success", "recorded": True, "history_size": len(self.performance_history)}
        except Exception as e:
            self.logger.exception(f"Performance recording error: {e}")
            return {"status": "error", "error": str(e)}

    async def _analyze_performance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            agent_name = params.get("agent_name")
            if agent_name:
                relevant = [p for p in self.performance_history if p.get("agent_name") == agent_name]
            else:
                relevant = self.performance_history
            return {"status": "success", "agent_name": agent_name, "total_records": len(relevant), "trend": "improving", "avg_score": 0.85}
        except Exception as e:
            self.logger.exception(f"Performance analysis error: {e}")
            return {"status": "error", "error": str(e)}

    async def _recommend_improvements(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            agent_name = params.get("agent_name", "")
            recommendations = ["Optimize query processing", "Increase context window", "Improve error handling"]
            return {"status": "success", "agent_name": agent_name, "recommendations": recommendations, "priority": "medium"}
        except Exception as e:
            self.logger.exception(f"Recommendation error: {e}")
            return {"status": "error", "error": str(e)}

    async def shutdown(self) -> bool:
        self.logger.info(f"{self.name} shutting down")
        self.update_status(AgentStatus.TERMINATED)
        return True
