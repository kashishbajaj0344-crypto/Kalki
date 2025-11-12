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
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from modules.utils.logger import get_logger
from modules.utils.config import CONFIG, register_module_version
from modules.utils import safe_read, compute_sha256, ensure_dir
from modules.metadata import extract_metadata, enrich_chunk_metadata
from modules.chunker import chunk_text
from modules.tagger import generate_tags, tag_chunk
from modules.learning.vectordb import VectorDBManager

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
        """Extracts text from supported file types (with PDF fallback, OCR, and table extraction)."""
        text = ""
        try:
            if file_path.suffix.lower() == ".pdf":
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    page_texts = []
                    for page in pdf.pages:
                        page_text = page.extract_text() or ""
                        
                        # Extract tables from this page
                        tables = page.extract_tables()
                        if tables:
                            table_texts = []
                            for table in tables:
                                if table and self._is_meaningful_table(table):
                                    # Format table as readable text
                                    table_text = self._format_table_as_text(table)
                                    table_texts.append(table_text)
                            
                            if table_texts:
                                page_text += "\n\nTABLES:\n" + "\n\n".join(table_texts)
                        
                        page_texts.append(page_text)
                    
                    text = "\n".join(page_texts)
                
                # OCR fallback for scanned PDFs (if extracted text is too short)
                if len(text.strip()) < 500:  # Threshold for scanned PDFs
                    logger.info(f"Text extraction yielded only {len(text.strip())} chars, attempting OCR for {file_path.name}")
                    ocr_text = self._extract_text_with_ocr(file_path)
                    if ocr_text and len(ocr_text.strip()) > len(text.strip()):
                        text = ocr_text
                        logger.info(f"OCR extracted {len(text.strip())} chars from {file_path.name}")
            
            elif file_path.suffix.lower() in [".txt", ".md"]:
                text = safe_read(file_path)
            elif file_path.suffix.lower() == ".docx":
                from docx import Document
                doc = Document(file_path)
                text = "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            logger.error(f"[Ingestor] Extraction failed for {file_path.name}: {e}")
        return text.strip()

    def _format_table_as_text(self, table: List[List[str]]) -> str:
        """Format a table as readable text with proper alignment."""
        if not table:
            return ""
        
        # Clean table data
        cleaned_table = []
        for row in table:
            if row:
                cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
                cleaned_table.append(cleaned_row)
        
        if not cleaned_table:
            return ""
        
        # Calculate column widths
        col_widths = []
        for col_idx in range(len(cleaned_table[0])):
            max_width = 0
            for row in cleaned_table:
                if col_idx < len(row):
                    max_width = max(max_width, len(row[col_idx]))
            col_widths.append(max_width)
        
        # Format table rows
        formatted_rows = []
        for row in cleaned_table:
            formatted_cells = []
            for col_idx, cell in enumerate(row):
                if col_idx < len(col_widths):
                    formatted_cells.append(cell.ljust(col_widths[col_idx]))
                else:
                    formatted_cells.append(cell)
            formatted_rows.append(" | ".join(formatted_cells))
        
        return "\n".join(formatted_rows)

    def _is_meaningful_table(self, table: List[List[str]]) -> bool:
        """Check if a table contains meaningful content (not just formatting artifacts)"""
        if not table or len(table) < 2:  # Need at least header + 1 data row
            return False
        
        # Flatten all cells and check content
        all_cells = []
        for row in table:
            if row:
                all_cells.extend([str(cell).strip() for cell in row if cell is not None])
        
        if not all_cells:
            return False
        
        # Check for table artifacts (patterns that indicate formatting issues)
        artifact_patterns = [
            r'^>\w+$',  # Single characters like ">d", ">s"
            r'^\|\s*\|\s*$',  # Empty table cells "|  |"
            r'^\s*\|\s*\|\s*$',  # Just separators
        ]
        
        # If more than 50% of cells match artifact patterns, skip the table
        artifact_count = 0
        for cell in all_cells:
            if any(re.match(pattern, cell) for pattern in artifact_patterns):
                artifact_count += 1
        
        if artifact_count / len(all_cells) > 0.5:
            return False
        
        # Check for meaningful engineering content
        engineering_keywords = [
            'dimension', 'tolerance', 'material', 'force', 'load', 'stress', 'strain',
            'velocity', 'acceleration', 'power', 'torque', 'pressure', 'temperature',
            'steel', 'aluminum', 'titanium', 'plastic', 'composite', 'bearing', 'shaft',
            'gear', 'motor', 'sensor', 'actuator', 'weld', 'bolt', 'screw', 'rivet'
        ]
        
        # If table contains engineering keywords, it's likely meaningful
        table_text = ' '.join(all_cells).lower()
        if any(keyword in table_text for keyword in engineering_keywords):
            return True
        
        # If table has structured data (numbers, measurements), keep it
        measurement_pattern = r'\d+(\.\d+)?\s*(mm|cm|m|kg|lb|psi|mpa|rpm|hz|°|deg)'
        if re.search(measurement_pattern, table_text):
            return True
        
        # Default: if table is small and clean, keep it; if large and messy, skip
        total_cells = len(all_cells)
        avg_cell_length = sum(len(cell) for cell in all_cells) / total_cells
        
        # Keep small tables with reasonable content, skip large ones with short cells
        if total_cells <= 20 and avg_cell_length > 2:
            return True
        elif total_cells > 20 and avg_cell_length < 3:
            return False
        
        return True

    def _extract_text_with_ocr(self, file_path: Path) -> str:
        """Extract text from PDF using OCR (Tesseract)."""
        try:
            import pytesseract
            from PIL import Image
            import pdfplumber
            import io
            
            text_parts = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    # Convert PDF page to image
                    page_image = page.to_image(resolution=300).original
                    
                    # Convert to PIL Image if needed
                    if not isinstance(page_image, Image.Image):
                        # Handle pdfplumber's image format
                        img_byte_arr = io.BytesIO()
                        page_image.save(img_byte_arr, format='PNG')
                        page_image = Image.open(img_byte_arr)
                    
                    # Preprocessing for better OCR
                    page_image = page_image.convert('L')  # Convert to grayscale
                    
                    # Extract text with OCR
                    page_text = pytesseract.image_to_string(page_image, config='--psm 1')
                    text_parts.append(page_text)
            
            return "\n".join(text_parts)
        except ImportError as e:
            logger.warning(f"OCR dependencies not available: {e}. Install pytesseract and tesseract-ocr")
            return ""
        except Exception as e:
            logger.error(f"OCR extraction failed for {file_path.name}: {e}")
            return ""

    def process_chunks(self, text: str, file_meta: Dict[str, Any], file_hash: str) -> tuple[List[Dict[str, Any]], List[str]]:
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

    def ingest_all_parallel(self, paths: Optional[List[Path]] = None, extensions=None, max_workers: int = 4) -> int:
        """Parallel ingestion of all files using ThreadPoolExecutor."""
        self.refresh_known_hashes()
        files = self.discover_files(paths, extensions)
        
        if not files:
            logger.info("No files found for ingestion")
            return 0
        
        logger.info(f"Starting parallel ingestion of {len(files)} files with {max_workers} workers")
        count = 0
        success_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {executor.submit(self.ingest_file, f): f for f in files}
            
            # Process results as they complete
            with tqdm(total=len(files), desc="Ingesting files (parallel)") as pbar:
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        if result:
                            success_count += 1
                        count += 1
                    except Exception as e:
                        logger.exception("Error ingesting %s: %s", file_path, e)
                        count += 1
                    pbar.update(1)
        
        logger.info("Parallel ingestion complete: %d/%d files successfully ingested", success_count, len(files))
        return success_count

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
    parser = argparse.ArgumentParser(description="Kalki Ingestor")
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