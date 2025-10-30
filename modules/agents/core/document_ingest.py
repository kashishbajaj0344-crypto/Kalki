from typing import Dict, Any
from pathlib import Path
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
        self.logger = logging.getLogger("kalki.agent.DocumentIngestAgent")
    
    async def initialize(self) -> bool:
        try:
            # Import the top-level ingest module (not relative to agents)
            from modules import ingest
            self.ingest_module = ingest
            self.logger.info(f"{self.name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"Failed to initialize {self.name}: {e}")
            return False
    
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
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
        try:
            file_path = params.get("file_path")
            domain = params.get("domain", "general")
            metadata = params.get("metadata", {})
            
            # Use existing ingest functionality
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
