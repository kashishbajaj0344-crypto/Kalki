"""
KALKI v2.3 — CLI Module v1.4
------------------------------------------------------------
Enterprise-grade command-line interface for Kalki ingestion pipeline.
- Launch ingestion, chunking, vectorization, and query pipelines
- Inspect modules and agent status
- Interactive and direct command execution
- Modular, extensible, and robust logging/error handling
- Registers version for audit/trace
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
from modules.logger import get_logger
from modules.ingest import DocumentIngestor, run_cli
from modules.chunker import chunk_text
from modules.vectordb import VectorDBManager
from modules.config import CONFIG, register_module_version, get_module_versions

__version__ = "KALKI v2.3 — cli.py v1.4"
register_module_version("cli.py", __version__)
logger = get_logger("CLI")

# ------------------------------------------------------------
# CLI Functions
# ------------------------------------------------------------
def cli_ingest(paths: list[str], async_mode: bool = False):
    """Run ingestion pipeline on a list of file paths or folders."""
    logger.info(f"Starting ingestion for {len(paths)} path(s). Async: {async_mode}")
    ingestor = DocumentIngestor()
    files = ingestor.discover_files(paths)
    if not files:
        logger.warning("No files found for ingestion.")
        return
    if async_mode:
        import asyncio
        asyncio.run(ingestor.ingest_all_async(paths))
    else:
        ingestor.ingest_all(paths)

def cli_chunk(file_path: str, mode: str = "semantic"):
    """Run chunking on a file and display summary."""
    p = Path(file_path)
    if not p.exists():
        logger.error(f"File not found: {p}")
        return
    text = p.read_text(encoding="utf-8")
    chunks = chunk_text(text, mode=mode)
    logger.info(f"Chunked {len(chunks)} pieces from {p}")
    for chunk in chunks[:5]:  # Display first 5 for preview
        logger.debug(f"Chunk {chunk['chunk_id']}: {chunk['text'][:50]}...")

def cli_query(query: str, top_k: int = 5):
    """Query the vector database."""
    db = VectorDBManager()
    results = db.query(query, k=top_k)
    logger.info(f"Query: {query}")
    for i, r in enumerate(results):
        logger.info(f"{i+1}. Score: {r['score']}, Text: {r['text'][:100]}...")


def cli_safe_query(query: str, session_context: Optional[dict] = None):
    """Query with full safety orchestration."""
    try:
        from kalki_agent_integration import safe_ask_kalki_sync
        result = safe_ask_kalki_sync(query, session_context)

        if result["status"] == "success":
            logger.info(f"✅ Safe Query Result: {result['answer'][:200]}...")
            safety = result.get("safety_assessment", {})
            logger.info(f"   Safety: Ethical={safety.get('ethical_score', 0):.2f}, Risk={safety.get('risk_level', 'unknown')}")
        elif result["status"] == "blocked":
            logger.warning(f"🚫 Query blocked: {result.get('reason', 'Unknown reason')}")
        elif result["status"] == "deferred":
            logger.info(f"⏳ Query deferred: {result.get('message', 'Resources unavailable')}")
        else:
            logger.error(f"❌ Query failed: {result.get('error', 'Unknown error')}")

        return result
    except Exception as e:
        logger.exception(f"Safe query failed: {e}")
        return {"status": "error", "error": str(e)}


def cli_safe_ingest(paths: list[str], domain: str = "general", context: Optional[dict] = None):
    """Safe ingestion with full safety orchestration."""
    try:
        from kalki_agent_integration import safe_ingest_pdf_file_sync

        for path_str in paths:
            path = Path(path_str)
            if path.is_file() and path.suffix.lower() == '.pdf':
                logger.info(f"Processing PDF: {path}")
                result = safe_ingest_pdf_file_sync(str(path), domain, context)

                if result["status"] == "success":
                    logger.info(f"✅ Safe ingestion successful: {path.name}")
                    safety = result.get("safety_assessment", {})
                    logger.info(f"   Safety: Ethical={safety.get('ethical_score', 0):.2f}, Risk={safety.get('risk_score', 0):.2f}")
                elif result["status"] == "blocked":
                    logger.warning(f"🚫 Ingestion blocked: {result.get('reason', 'Unknown reason')}")
                elif result["status"] == "deferred":
                    logger.info(f"⏳ Ingestion deferred: {result.get('message', 'Resources unavailable')}")
                else:
                    logger.error(f"❌ Ingestion failed: {result.get('error', 'Unknown error')}")
            else:
                logger.warning(f"Skipping non-PDF file: {path}")

    except Exception as e:
        logger.exception(f"Safe ingestion failed: {e}")

def cli_status():
    """Display status of core modules."""
    logger.info("Kalki CLI Status Check:")
    for k, v in CONFIG.items():
        logger.info(f"{k}: {v}")
    logger.info(f"Registered module versions: {get_module_versions()}")


async def cli_safety_status():
    """Display safety system status."""
    try:
        from kalki_agent_integration import get_global_integration
        integration = await get_global_integration()
        status = await integration.get_system_safety_status()

        if status["status"] == "success":
            logger.info("🛡️ Safety System Status:")
            health = status["system_health"]
            logger.info(f"   Ethics Agent: {health['ethics_agent']['evaluations_count']} evaluations")
            logger.info(f"   Risk Agent: {health['risk_agent']['patterns_tracked']} patterns")
            logger.info(f"   Simulation Agent: {health['simulation_agent']['simulations_run']} simulations")
            logger.info("   Status: ✅ All systems operational")
        else:
            logger.error(f"❌ Safety system error: {status.get('error', 'Unknown')}")
    except Exception as e:
        logger.exception(f"Failed to get safety status: {e}")


def cli_safety_status_sync():
    """Synchronous wrapper for safety status."""
    try:
        import asyncio
        asyncio.run(cli_safety_status())
    except Exception as e:
        logger.exception(f"Safety status check failed: {e}")

# ------------------------------------------------------------
# Argument Parsing
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Kalki v2.3 CLI — Ingestion, Chunking, VectorDB, and Agent Operations"
    )
    parser.add_argument("--ingest", nargs="+", help="Paths to files or folders for ingestion")
    parser.add_argument("--chunk", help="File path to chunk")
    parser.add_argument("--chunk-mode", default="semantic", help="Chunking mode: semantic, paragraph, fixed")
    parser.add_argument("--query", help="Query the vector DB")
    parser.add_argument("--safe-query", help="Query with full safety orchestration")
    parser.add_argument("--safe-ingest", nargs="+", help="Safe ingestion with full safety orchestration")
    parser.add_argument("--domain", default="general", help="Domain for safe operations (general, academic, technical, etc.)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top results for query")
    parser.add_argument("--status", action="store_true", help="Display system status")
    parser.add_argument("--safety-status", action="store_true", help="Display safety system status")
    parser.add_argument("--version", action="store_true", help="Show CLI version")
    parser.add_argument("--async", dest="use_async", action="store_true", help="Use async ingestion")
    parser.add_argument("--interactive", action="store_true", help="Run interactive drag-drop CLI mode")
    
    args = parser.parse_args()
    
    if args.version:
        print(f"Kalki CLI Version: {__version__}")
        sys.exit(0)
    
    if args.status:
        cli_status()
    
    if args.safety_status:
        cli_safety_status_sync()
    
    if args.interactive:
        run_cli()

    if args.ingest:
        cli_ingest(args.ingest, async_mode=args.use_async)
    
    if args.safe_ingest:
        cli_safe_ingest(args.safe_ingest, domain=args.domain)
    
    if args.chunk:
        cli_chunk(args.chunk, mode=args.chunk_mode)
    
    if args.query:
        cli_query(args.query, top_k=args.top_k)
    
    if args.safe_query:
        cli_safe_query(args.safe_query, session_context={"domain": args.domain})

# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    main()