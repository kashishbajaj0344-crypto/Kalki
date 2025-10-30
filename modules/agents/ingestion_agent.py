"""
KALKI v2.3 — Ingestion Agent Module v6.2
------------------------------------------------------------
Enterprise-grade agent for document ingestion and orchestration.
- File, text, directory ingestion (async-safe)
- Chunking, metadata, tagging, vector DB insertion
- Logging, audit, status, query, version registry
- Skips hidden/system files (like .DS_Store)
- Only processes supported file types
- Robust error handling for file reading and metadata
- Extensible for LLM/agent workflows
- DEFENSIVE METADATA VALIDATION: always passes a dict (never a string/object with .metadata)
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from modules.logger import get_logger
from modules.config import CONFIG, register_module_version
from modules.ingest import DocumentIngestor
from modules.vectordb import VectorDBManager

try:
    from modules.ocr import extract_text_file as ocr_extract_text
except ImportError:
    ocr_extract_text = None

__version__ = "KALKI v2.3 — ingestion_agent.py v6.2"
register_module_version("ingestion_agent.py", __version__)
logger = get_logger("IngestionAgent")

# Allowed file extensions for ingestion
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}

class IngestionAgent:
    def __init__(
        self,
        name: str = "IngestionAgent",
        config: Optional[Dict[str, Any]] = None,
        chunker: Optional[Callable] = None,
        tagger: Optional[Callable] = None,
        metadata_fn: Optional[Callable] = None,
    ):
        self.name = name
        self.config = config or CONFIG
        self.vectordb = VectorDBManager()
        self.ingestor = DocumentIngestor()
        self.state: Dict[str, Any] = {"runs": [], "errors": [], "results": []}
        self.chunker = chunker
        self.tagger = tagger
        self.metadata_fn = metadata_fn
        logger.info(f"Agent '{self.name}' initialized.")

    async def ingest_file(self, file_path: Path) -> Dict:
        """
        Ingest a single file. Skips hidden/system files and unsupported extensions.
        Returns chunk-level status/results.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warning("File not found: %s", file_path)
            return {"file": str(file_path), "status": "missing"}
        if not file_path.is_file():
            logger.warning("Path is not a file (skipping): %s", file_path)
            return {"file": str(file_path), "status": "not_a_file"}
        # Skip hidden/system files and unsupported extensions
        if file_path.name.startswith(".") or file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            logger.info("Skipping hidden/system/unsupported file: %s", file_path)
            return {"file": str(file_path), "status": "skipped"}

        logger.info("Ingesting file: %s", file_path)
        text = ""
        ext = file_path.suffix.lower()
        try:
            if ext in [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"] and ocr_extract_text:
                logger.debug("Using OCR for %s", file_path)
                text = await asyncio.to_thread(ocr_extract_text, file_path)
            else:
                text = await asyncio.to_thread(file_path.read_text, "utf-8")
        except Exception as e:
            logger.error("Failed to read file %s: %s", file_path, e)
            return {"file": str(file_path), "status": "read_error", "error": str(e)}

        return await self.ingest_text(text, source=str(file_path))

    async def ingest_text(self, text: str, source: Optional[str] = None) -> Dict:
        """
        Ingest raw text. Chunks, tags, generates metadata, inserts into vector DB.
        Returns chunk-level and overall status/results.
        """
        if not text.strip():
            logger.warning("Empty text received for ingestion: %s", source)
            return {"source": source, "status": "empty"}

        chunk_fn = self.chunker or self.ingestor.process_chunks
        chunks = chunk_fn(text, {}, "raw") if self.chunker else self.ingestor.process_chunks(text, {}, "raw")[1]
        results = []
        for idx, chunk_text_data in enumerate(chunks):
            chunk_id = f"{source}_chunk_{idx}"
            metadata = self.metadata_fn(chunk_text_data, source, chunk_id) if self.metadata_fn else {"source": source, "chunk_id": chunk_id}
            tags = self.tagger(chunk_text_data) if self.tagger else []
            metadata["tags"] = tags
            # Defensive: ensure metadata is always a dict, never a string
            if not isinstance(metadata, dict):
                logger.warning(f"Metadata for chunk {chunk_id} is not a dict (got {type(metadata)}). Converting to empty dict.")
                metadata = {}
            self.vectordb.add_document(source, [chunk_text_data], [metadata])
            results.append({"chunk_id": chunk_id, "status": "inserted", "metadata": metadata})
        logger.info("Ingested %d chunks from source %s", len(chunks), source)
        return {"source": source, "status": "success", "chunks": len(chunks), "details": results}

    async def ingest_directory(self, directory: Path, recursive: bool = True, extensions: Optional[List[str]] = None) -> List[Dict]:
        """
        Ingest all files in a directory (optionally filtered by extensions).
        Only processes files, skips directories and hidden/system files.
        """
        directory = Path(directory)
        if not directory.exists() or not directory.is_dir():
            logger.warning("Directory not found: %s", directory)
            return []

        files = list(directory.rglob("*") if recursive else directory.glob("*"))
        # Skip anything that's not a file, or is hidden/system, or unsupported extension
        files = [f for f in files if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in ALLOWED_EXTENSIONS]
        if extensions:
            extensions = [ext.lower() for ext in extensions]
            files = [f for f in files if f.suffix.lower() in extensions]
        results = []
        for file_path in files:
            result = await self.ingest_file(file_path)
            results.append(result)
        logger.info("Ingested directory %s: %d files", directory, len(results))
        return results

    async def ingest_documents(self, paths: List[Path]) -> int:
        """
        Batch ingestion pipeline for a list of documents using DocumentIngestor.
        Tracks progress, errors, and results.
        """
        logger.info(f"Agent '{self.name}' starting ingestion for {len(paths)} files.")
        try:
            count = await self.ingestor.ingest_all_async(paths)
            self.state["runs"].append({"action": "ingest", "count": count, "paths": [str(p) for p in paths]})
            logger.info(f"Agent '{self.name}' completed ingestion: {count} files.")
            return count
        except Exception as e:
            logger.error(f"Agent '{self.name}' failed ingestion: {e}")
            self.state["errors"].append(str(e))
            return 0

    def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the vector database through the agent.
        """
        logger.info(f"Agent '{self.name}' running query: {query}")
        try:
            results = self.vectordb.query(query, k=top_k)
            self.state["results"].append({"query": query, "results": results})
            return results
        except Exception as e:
            logger.error(f"Agent '{self.name}' query failed: {e}")
            self.state["errors"].append(str(e))
            return []

    def get_status(self) -> Dict[str, Any]:
        """
        Returns the current agent state for monitoring/audit.
        """
        status = {
            "name": self.name,
            "config": self.config,
            "runs": len(self.state["runs"]),
            "errors": self.state["errors"],
            "last_result": self.state["results"][-1] if self.state["results"] else None
        }
        logger.info(f"Agent '{self.name}' status: {status}")
        return status

    def reset(self):
        """Reset agent state."""
        self.state = {"runs": [], "errors": [], "results": []}
        logger.info(f"Agent '{self.name}' state reset.")

# ------------------------------------------------------------
# Self-test CLI for agent module
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kalki Ingestion Agent Orchestrator")
    parser.add_argument("--ingest", nargs="+", help="Paths to files or folders for ingestion via DocumentIngestor")
    parser.add_argument("--ingest-dir", type=str, help="Directory for file-by-file ingestion")
    parser.add_argument("--query", type=str, help="Run a query against the knowledge base")
    parser.add_argument("--status", action="store_true", help="Show agent status")
    args = parser.parse_args()

    agent = IngestionAgent()
    if args.ingest:
        asyncio.run(agent.ingest_documents([Path(p) for p in args.ingest]))
    if args.ingest_dir:
        asyncio.run(agent.ingest_directory(Path(args.ingest_dir)))
    if args.query:
        results = agent.query(args.query)
        print(f"Query Results: {results}")
    if args.status:
        print("Agent Status:", agent.get_status())