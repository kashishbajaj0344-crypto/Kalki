from typing import Dict, Any, List
import logging
import random

from ..base_agent import BaseAgent, AgentCapability, AgentStatus


class CreativeAgent(BaseAgent):
    """Creative synthesis and ideation agent"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="CreativeAgent",
            capabilities=[
                AgentCapability.CREATIVE_SYNTHESIS,
                AgentCapability.IDEA_FUSION
            ],
            description="Generates creative ideas through synthesis and fusion",
            config=config or {}
        )
        self.logger = logging.getLogger("kalki.agent.CreativeAgent")

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
            if action == "ideate":
                result = await self._ideate(params)
            elif action == "fuse":
                result = await self._fuse_ideas(params)
            elif action == "synthesize":
                result = await self._synthesize(params)
            else:
                result = {"status": "error", "error": f"Unknown action: {action}"}
            return result
        finally:
            self.update_status(AgentStatus.IDLE)

    async def _ideate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            topic = params.get("topic", "")
            count = params.get("count", 5)

            ideas = [f"Idea {i+1}: Creative approach to {topic}" for i in range(count)]

            return {"status": "success", "topic": topic, "ideas": ideas, "novelty_score": random.uniform(0.6, 0.95)}
        except Exception as e:
            self.logger.exception(f"Ideation error: {e}")
            return {"status": "error", "error": str(e)}

    async def _fuse_ideas(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            concepts = params.get("concepts", [])
            if len(concepts) < 2:
                return {"status": "error", "error": "Need at least 2 concepts to fuse"}
            fusion = f"Fusion of {' + '.join(concepts)}: A novel synthesis"
            return {"status": "success", "concepts": concepts, "fusion": fusion, "creativity_score": random.uniform(0.7, 0.98)}
        except Exception as e:
            self.logger.exception(f"Idea fusion error: {e}")
            return {"status": "error", "error": str(e)}

    async def _synthesize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            domains = params.get("domains", [])
            synthesis = {"insight": f"Cross-domain synthesis across {', '.join(domains)}", "applications": ["Application 1", "Application 2", "Application 3"], "patentability": random.uniform(0.5, 0.9)}
            return {"status": "success", "domains": domains, "synthesis": synthesis}
        except Exception as e:
            self.logger.exception(f"Synthesis error: {e}")
            return {"status": "error", "error": str(e)}

    async def shutdown(self) -> bool:
        self.logger.info(f"{self.name} shutting down")
        self.update_status(AgentStatus.TERMINATED)
        return True
