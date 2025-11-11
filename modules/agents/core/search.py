from typing import Dict, Any
import logging

from ..base_agent import BaseAgent, AgentCapability, AgentStatus
from modules.learning.vectordb import VectorDBManager


class SearchAgent(BaseAgent):
    """Semantic search agent with multi-domain support"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="SearchAgent",
            capabilities=[AgentCapability.SEARCH],
            description="Performs semantic search across knowledge base",
            config=config or {}
        )
        self.logger = logging.getLogger("kalki.agent.SearchAgent")

    async def initialize(self) -> bool:
        try:
            # Use the VectorDBManager instance for searches
            self.vectordb = VectorDBManager()
            self.logger.info(f"{self.name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"Failed to initialize {self.name}: {e}")
            return False

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        action = task.get("action")
        params = task.get("params", {})

        if action == "search":
            query = params.get("query") or task.get("query")
            top_k = params.get("top_k") if isinstance(params, dict) else None
            top_k = top_k if top_k is not None else task.get("top_k", 5)
            composed_params = {"query": query or "", "top_k": top_k}
            return await self._search(composed_params)
        else:
            return {
                "status": "error",
                "error": f"Unknown action: {action}"
            }

    async def _search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            query = params.get("query", "")
            top_k = params.get("top_k", 5)
            
            chunks = self.vectordb.get_top_k_chunks(query, top_k=top_k)
            
            return {
                "status": "success",
                "query": query,
                "results": chunks,
                "count": len(chunks)
            }
            
        except Exception as e:
            self.logger.exception(f"Search error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def shutdown(self) -> bool:
        self.logger.info(f"{self.name} shutting down")
        self.update_status(AgentStatus.TERMINATED)
        return True
