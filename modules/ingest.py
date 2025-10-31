"""
KALKI v2.3 — Unified Document Ingestor v1.2
------------------------------------------------------------
Unified, production-grade document ingestion pipeline for KALKI.
- Discovers files by extension, drag & drop, or folder
- Extracts metadata, chunks text, tags chunks, deduplicates
- Batch ingestion, async/sync, retries, logging
- CLI and API entrypoints
- Pipeline ready for agent/LLM/advanced enrichment
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm

from modules.logger import get_logger
from modules.config import CONFIG, register_module_version
from modules.utils import safe_read, compute_sha256, ensure_dir
from modules.metadata import extract_metadata, enrich_chunk_metadata
from modules.chunker import chunk_text
from modules.tagger import generate_tags, tag_chunk
from modules.vectordb import VectorDBManager

__version__ = "KALKI v2.3 — Unified Document Ingestor v1.2"
register_module_version("ingest.py", __version__)

logger = get_logger("ingest")


class DocumentIngestor:
    """
    Orchestrates full ingestion pipeline:
    discover → read → chunk → tag → enrich → deduplicate → store.
    """

    SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".md", ".docx"]

    def __init__(
        self,
        ingest_dir: Optional[Path] = None,
        batch_size: int = 8,
        chunk_mode: str = "semantic",
        tag_method: str = "keywords",
        retry: int = 3,
        retry_delay: float = 2.0,
        vector_db: Optional[VectorDBManager] = None
    ):
        self.ingest_dir = Path(ingest_dir) if ingest_dir else Path(CONFIG.get("ingest_dir", "data/ingest"))
        ensure_dir(self.ingest_dir)
        self.vectordb = vector_db or VectorDBManager()
        self.known_hashes = self.vectordb.known_hashes
        self.batch_size = batch_size
        self.chunk_mode = chunk_mode
        self.tag_method = tag_method
        self.retry = retry
        self.retry_delay = retry_delay

    def refresh_known_hashes(self):
        self.known_hashes = self.vectordb._load_known_hashes()

    def discover_files(self, paths: Optional[List[Union[Path, str]]] = None, extensions=None) -> List[Path]:
        """Find all files from user paths, drag & drop, or recursively from ingest_dir."""
        exts = [ext.lower() for ext in (extensions or self.SUPPORTED_EXTENSIONS)]
        files = []
        if paths:
            for p in paths:
                p = Path(str(p).strip('"'))
                if p.is_file() and p.suffix.lower() in exts:
                    files.append(p)
                elif p.is_dir():
                    for e in exts:
                        files.extend(p.rglob(f"*{e}"))
        else:
            files = [f for f in self.ingest_dir.rglob("*") if f.suffix.lower() in exts]
        logger.info("Discovered %d files for ingestion.", len(files))
        return files

    def extract_text(self, file_path: Path) -> str:
        """Extracts text from supported file types (with PDF fallback)."""
        text = ""
        try:
            if file_path.suffix.lower() == ".pdf":
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            elif file_path.suffix.lower() in [".txt", ".md"]:
                text = safe_read(file_path)
            elif file_path.suffix.lower() == ".docx":
                from docx import Document
                doc = Document(file_path)
                text = "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            logger.error(f"[Ingestor] Extraction failed for {file_path.name}: {e}")
        return text.strip()

    def process_chunks(self, text: str, file_meta: Dict[str, Any], file_hash: str) -> (List[Dict[str, Any]], List[str]):
        """Chunk, tag, and enrich document text."""
        chunks = chunk_text(text, mode=self.chunk_mode)
        metadatas, texts = [], []
        for chunk in chunks:
            chunk_id = f"{file_hash}_{chunk['chunk_id']}"
            if chunk_id in self.known_hashes:
                continue
            base_meta = enrich_chunk_metadata(file_meta, chunk['chunk_id'], chunk["text"])
            tags = generate_tags(chunk, method=self.tag_method)
            chunk_meta = base_meta.copy()
            chunk_meta.update({"tags": tags})
            chunk_meta["chunk_id"] = chunk_id
            metadatas.append(chunk_meta)
            texts.append(chunk["text"])
        return metadatas, texts

    def ingest_file(self, file_path: Path) -> bool:
        """Ingests a single document with full pipeline."""
        file_hash = compute_sha256(file_path)
        if file_hash in {"error_hash", None}:
            logger.warning("Skipping file (hash error): %s", file_path)
            return False

        text = self.extract_text(file_path)
        if not text or not text.strip():
            logger.warning("No text extracted from %s", file_path)
            return False

        file_meta = extract_metadata(file_path)
        metadatas, texts = self.process_chunks(text, file_meta, file_hash)

        if not texts:
            logger.info("All chunks already exist for %s", file_path)
            return False

        for attempt in range(1, self.retry + 1):
            try:
                for i in range(0, len(texts), self.batch_size):
                    batch_texts = texts[i:i + self.batch_size]
                    batch_metas = metadatas[i:i + self.batch_size]
                    self.vectordb.add_document(file_path, batch_texts, batch_metas)
                    for meta in batch_metas:
                        self.known_hashes.add(meta["chunk_id"])
                logger.info("Ingested %s (%d chunks)", file_path, len(texts))
                return True
            except Exception as e:
                logger.exception("Attempt %d failed for %s: %s", attempt, file_path, e)
                if attempt < self.retry:
                    import time
                    time.sleep(self.retry_delay * attempt)
        return False

    def ingest_all(self, paths: Optional[List[Path]] = None, extensions=None) -> int:
        """Sync ingestion of all files or directory."""
        self.refresh_known_hashes()
        files = self.discover_files(paths, extensions)
        count = 0
        for f in tqdm(files, desc="Ingesting files"):
            try:
                if self.ingest_file(f):
                    count += 1
            except Exception as e:
                logger.exception("Error ingesting %s: %s", f, e)
        logger.info("Ingestion complete: %d/%d files", count, len(files))
        return count

    async def ingest_file_async(self, file_path: Path) -> bool:
        """Async wrapper for ingest_file."""
        return await asyncio.to_thread(self.ingest_file, file_path)

    async def ingest_all_async(self, paths: Optional[List[Path]] = None, extensions=None) -> int:
        """Async ingestion for batch files."""
        self.refresh_known_hashes()
        files = self.discover_files(paths, extensions)
        results = await asyncio.gather(*[self.ingest_file_async(f) for f in files])
        count = sum(1 for r in results if r)
        logger.info("Async ingestion complete: %d/%d files", count, len(files))
        return count


# Global ingestor instance
_ingestor = None

def get_ingestor():
    """Get or create global DocumentIngestor instance"""
    global _ingestor
    if _ingestor is None:
        _ingestor = DocumentIngestor()
    return _ingestor

def ingest_pdf_file(file_path: str, domain: str = "general") -> bool:
    """Simplified PDF ingestion function for external use"""
    ingestor = get_ingestor()
    return ingestor.ingest_file(Path(file_path))


def run_cli():
    print("\n=== Kalki Document Ingest CLI ===")
    print("Drag & drop files or folders here, then press Enter:")
    user_input = input().strip()
    if not user_input:
        print("No input detected.")
        return
    paths = user_input.split()
    ingestor = DocumentIngestor()
    files = ingestor.discover_files(paths)
    if not files:
        print("No valid files found.")
        return
    print(f"Found {len(files)} files to ingest...")
    results = {}
    for f in files:
        results[str(f)] = ingestor.ingest_file(f)
    print("\n=== Ingestion Summary ===")
    for f, ok in results.items():
        print(f"{f}: {'✅ Success' if ok else '❌ Failed'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kalki Unified Ingestor")
    parser.add_argument("--folder", type=str, help="Folder to ingest recursively")
    parser.add_argument("--file", type=str, help="Single file to ingest")
    parser.add_argument("--cli", action="store_true", help="Run interactive drag-drop CLI mode")
    parser.add_argument("--async", dest="use_async", action="store_true", help="Use async ingestion")
    parser.add_argument("--chunk_mode", type=str, default="semantic", help="Chunking mode (semantic, paragraph, sentence, fixed)")
    parser.add_argument("--tag_method", type=str, default="keywords", help="Tagging method (keywords, llm, etc.)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for ingestion")
    parser.add_argument("--retry", type=int, default=3, help="Number of ingestion retries")
    parser.add_argument("--retry_delay", type=float, default=2.0, help="Delay between retries (seconds)")
    args = parser.parse_args()

    ingestor = DocumentIngestor(
        batch_size=args.batch_size,
        chunk_mode=args.chunk_mode,
        tag_method=args.tag_method,
        retry=args.retry,
        retry_delay=args.retry_delay,
    )

    if args.cli:
        run_cli()
    else:
        if args.folder:
            if args.use_async:
                asyncio.run(ingestor.ingest_all_async(paths=[Path(args.folder)]))
            else:
                ingestor.ingest_all(paths=[Path(args.folder)])
        elif args.file:
            if args.use_async:
                asyncio.run(ingestor.ingest_file_async(Path(args.file)))
            else:
                ingestor.ingest_file(Path(args.file))
        else:
            run_cli()