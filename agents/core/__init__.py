"""
Core foundational agents (Phase 1-5)
"""
from typing import Dict, Any, List
import logging
from ..base_agent import BaseAgent, AgentCapability, AgentStatus


class DocumentIngestAgent(BaseAgent):
    """Enhanced document ingestion agent with multi-format support"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="DocumentIngestAgent",
            capabilities=[
                AgentCapability.DOCUMENT_INGESTION,
                AgentCapability.VECTORIZATION
            ],
            description="Ingests documents from multiple formats and creates vector embeddings",
            config=config or {}
        )
        self.supported_formats = ['.pdf', '.txt', '.md', '.json', '.csv']
    
    async def initialize(self) -> bool:
        try:
            from ..modules import ingest
            self.ingest_module = ingest
            self.logger.info(f"{self.name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"Failed to initialize {self.name}: {e}")
            return False
    
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute document ingestion
        
        Task format:
        {
            "action": "ingest",
            "params": {
                "file_path": str,
                "domain": str,
                "metadata": dict
            }
        }
        """
        action = task.get("action")
        params = task.get("params", {})
        
        if action == "ingest":
            return await self._ingest_document(params)
        elif action == "list_formats":
            return {
                "status": "success",
                "formats": self.supported_formats
            }
        else:
            return {
                "status": "error",
                "error": f"Unknown action: {action}"
            }
    
    async def _ingest_document(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest a single document"""
        try:
            file_path = params.get("file_path")
            domain = params.get("domain", "general")
            metadata = params.get("metadata", {})
            
            # Use existing ingest functionality
            from pathlib import Path
            from modules.ingest import ingest_pdf_file
            
            success = ingest_pdf_file(
                Path(file_path),
                domain=domain,
                title=metadata.get("title"),
                author=metadata.get("author")
            )
            
            if success:
                await self.emit_event("document.ingested", {
                    "file_path": file_path,
                    "domain": domain
                })
                
                return {
                    "status": "success",
                    "file_path": file_path,
                    "domain": domain
                }
            else:
                return {
                    "status": "error",
                    "error": "Ingestion failed"
                }
                
        except Exception as e:
            self.logger.exception(f"Document ingestion error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def shutdown(self) -> bool:
        self.logger.info(f"{self.name} shutting down")
        self.update_status(AgentStatus.TERMINATED)
        return True


class SearchAgent(BaseAgent):
    """Semantic search agent with multi-domain support"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="SearchAgent",
            capabilities=[AgentCapability.SEARCH],
            description="Performs semantic search across knowledge base",
            config=config or {}
        )
    
    async def initialize(self) -> bool:
        try:
            from modules import vectordb
            self.vectordb = vectordb
            self.logger.info(f"{self.name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"Failed to initialize {self.name}: {e}")
            return False
    
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute search
        
        Task format:
        {
            "action": "search",
            "params": {
                "query": str,
                "top_k": int,
                "domain": str (optional)
            }
        }
        """
        action = task.get("action")
        params = task.get("params", {})
        
        if action == "search":
            return await self._search(params)
        else:
            return {
                "status": "error",
                "error": f"Unknown action: {action}"
            }
    
    async def _search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform semantic search"""
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


class PlannerAgent(BaseAgent):
    """Task planning and decomposition agent"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="PlannerAgent",
            capabilities=[AgentCapability.PLANNING],
            description="Decomposes complex tasks into executable sub-tasks",
            config=config or {}
        )
    
    async def initialize(self) -> bool:
        try:
            self.logger.info(f"{self.name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"Failed to initialize {self.name}: {e}")
            return False
    
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute planning
        
        Task format:
        {
            "action": "plan",
            "params": {
                "goal": str,
                "constraints": dict
            }
        }
        """
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
        """Create an execution plan"""
        try:
            goal = params.get("goal", "")
            constraints = params.get("constraints", {})
            
            # Simple planning logic - can be enhanced with LLM
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
        """
        Execute reasoning task
        
        Task format:
        {
            "action": "reason",
            "params": {
                "query": str,
                "context": str (optional),
                "steps": int (reasoning depth)
            }
        }
        """
        action = task.get("action")
        params = task.get("params", {})
        
        if action == "reason":
            return await self._reason(params)
        else:
            return {
                "status": "error",
                "error": f"Unknown action: {action}"
            }
    
    async def _reason(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform multi-step reasoning"""
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
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def shutdown(self) -> bool:
        self.logger.info(f"{self.name} shutting down")
        self.update_status(AgentStatus.TERMINATED)
        return True


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
    
    async def initialize(self) -> bool:
        try:
            self.logger.info(f"{self.name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"Failed to initialize {self.name}: {e}")
            return False
    
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute memory operation
        
        Task format:
        {
            "action": "store|retrieve|update",
            "params": {
                "memory_type": "episodic|semantic",
                "key": str,
                "value": any,
                "query": str
            }
        }
        """
        action = task.get("action")
        params = task.get("params", {})
        
        if action == "store":
            return await self._store(params)
        elif action == "retrieve":
            return await self._retrieve(params)
        elif action == "update":
            return await self._update(params)
        else:
            return {
                "status": "error",
                "error": f"Unknown action: {action}"
            }
    
    async def _store(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Store memory"""
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
            
            return {
                "status": "success",
                "memory_type": memory_type,
                "stored": True
            }
            
        except Exception as e:
            self.logger.exception(f"Memory store error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _retrieve(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve memory"""
        try:
            memory_type = params.get("memory_type", "episodic")
            key = params.get("key")
            
            if memory_type == "episodic":
                results = [m for m in self.episodic_memory if m.get("key") == key]
                return {
                    "status": "success",
                    "memory_type": memory_type,
                    "results": results
                }
            else:
                value = self.semantic_memory.get(key)
                return {
                    "status": "success",
                    "memory_type": memory_type,
                    "value": value
                }
                
        except Exception as e:
            self.logger.exception(f"Memory retrieve error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _update(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Update memory"""
        try:
            memory_type = params.get("memory_type", "semantic")
            key = params.get("key")
            value = params.get("value")
            
            if memory_type == "semantic":
                self.semantic_memory[key] = value
                return {
                    "status": "success",
                    "memory_type": memory_type,
                    "updated": True
                }
            else:
                return {
                    "status": "error",
                    "error": "Episodic memory cannot be updated, only stored"
                }
                
        except Exception as e:
            self.logger.exception(f"Memory update error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def shutdown(self) -> bool:
        self.logger.info(f"{self.name} shutting down")
        self.update_status(AgentStatus.TERMINATED)
        return True
