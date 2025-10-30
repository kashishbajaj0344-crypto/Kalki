from typing import Dict, Any, List
import logging

from ..base_agent import BaseAgent, AgentCapability, AgentStatus


class MemoryAgent(BaseAgent):
    """Episodic and semantic memory management"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="MemoryAgent",
            capabilities=[AgentCapability.MEMORY],
            description="Manages episodic and semantic memory",
            config=config or {}
        )
        self.episodic_memory: List[Dict[str, Any]] = []
        self.semantic_memory: Dict[str, Any] = {}
        self.logger = logging.getLogger("kalki.agent.MemoryAgent")

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

        if action == "store":
            return await self._store(params)
        elif action == "retrieve":
            return await self._retrieve(params)
        elif action == "update":
            return await self._update(params)
        else:
            return {"status": "error", "error": f"Unknown action: {action}"}

    async def _store(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            memory_type = params.get("memory_type", "episodic")
            key = params.get("key")
            value = params.get("value")
            
            if memory_type == "episodic":
                self.episodic_memory.append({
                    "key": key,
                    "value": value,
                    "timestamp": self.last_active.isoformat()
                })
            else:
                self.semantic_memory[key] = value
            
            return {"status": "success", "memory_type": memory_type, "stored": True}
            
        except Exception as e:
            self.logger.exception(f"Memory store error: {e}")
            return {"status": "error", "error": str(e)}

    async def _retrieve(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            memory_type = params.get("memory_type", "episodic")
            key = params.get("key")
            
            if memory_type == "episodic":
                results = [m for m in self.episodic_memory if m.get("key") == key]
                return {"status": "success", "memory_type": memory_type, "results": results}
            else:
                value = self.semantic_memory.get(key)
                return {"status": "success", "memory_type": memory_type, "value": value}
                
        except Exception as e:
            self.logger.exception(f"Memory retrieve error: {e}")
            return {"status": "error", "error": str(e)}

    async def _update(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            memory_type = params.get("memory_type", "semantic")
            key = params.get("key")
            value = params.get("value")
            
            if memory_type == "semantic":
                self.semantic_memory[key] = value
                return {"status": "success", "memory_type": memory_type, "updated": True}
            else:
                return {"status": "error", "error": "Episodic memory cannot be updated, only stored"}
                
        except Exception as e:
            self.logger.exception(f"Memory update error: {e}")
            return {"status": "error", "error": str(e)}

    async def shutdown(self) -> bool:
        self.logger.info(f"{self.name} shutting down")
        self.update_status(AgentStatus.TERMINATED)
        return True
