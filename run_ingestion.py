#!/usr/bin/env python3
"""
KALKI Full Ingestion Pipeline
Ingests PDFs into both Vector DB and Knowledge Databases
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.learning.ingest import DocumentIngestor
from modules.hybrid_learning_system import KnowledgeExtractor
from modules.learning.vectordb import VectorDBManager
from modules.utils.logging_config import setup_logging, get_logger

setup_logging(log_level="INFO")
logger = get_logger("Kalki.Ingestion")


async def ingest_pdf_full(pdf_path: Path, knowledge_extractor: KnowledgeExtractor) -> Dict[str, Any]:
    """
    Full ingestion: Vector DB + Knowledge Extraction
    """
    result = {
        "pdf": str(pdf_path),
        "success": False,
        "vector_chunks": 0,
        "knowledge_extracted": {},
        "error": None
    }
    
    try:
        # Step 1: Extract text (needed for both)
        logger.info(f"📄 Extracting text from: {pdf_path.name}")
        ingestor = DocumentIngestor()
        text = ingestor.extract_text(pdf_path)
        
        if not text or len(text.strip()) < 100:
            logger.warning(f"⚠️  Minimal text extracted from {pdf_path.name} ({len(text)} chars)")
            result["error"] = f"Minimal text extracted ({len(text)} chars)"
            # Still try knowledge extraction with what we have
        else:
            logger.info(f"✅ Extracted {len(text)} characters")
        
        # Step 2: Ingest into Vector DB
        logger.info(f"📥 Ingesting into Vector DB: {pdf_path.name}")
        success = ingestor.ingest_file(pdf_path)
        
        if success:
            # Count chunks
            if text:
                chunks = ingestor.process_chunks(text, {}, "")[1]
                result["vector_chunks"] = len(chunks)
                logger.info(f"✅ Vector DB: {len(chunks)} chunks ingested")
            else:
                logger.warning(f"⚠️  Could not count chunks (no text)")
        else:
            logger.warning(f"⚠️  Vector DB ingestion returned False (may already exist)")
        
        # Step 3: Extract structured knowledge
        logger.info(f"🔍 Extracting knowledge from: {pdf_path.name}")
        extraction_results = knowledge_extractor.extract_from_pdf(
            str(pdf_path),
            text if text else "",
            use_llm_enhancements=True,
            extract_images=False  # Skip images for now (faster)
        )
        
        # Map results to our format
        result["knowledge_extracted"] = {
            "formulas": extraction_results.get("formulas", 0),
            "materials": extraction_results.get("materials", 0),
            "design_rules": extraction_results.get("rules", 0),
            "code_requirements": extraction_results.get("codes", 0),
            "span_tables": extraction_results.get("span_tables", 0),
            "procedures": extraction_results.get("procedures", 0),
            "inspection_criteria": extraction_results.get("inspection_criteria", 0),
            "cost_data": extraction_results.get("cost_data", 0),
            "load_parameters": extraction_results.get("load_parameters", 0),
            "decision_trees": extraction_results.get("decision_trees", 0),
        }
        
        # Consider success if either vector DB or knowledge extraction worked
        result["success"] = success or sum(result["knowledge_extracted"].values()) > 0
        logger.info(f"✅ Knowledge extracted: {result['knowledge_extracted']}")
        
    except Exception as e:
        logger.exception(f"❌ Error ingesting {pdf_path.name}: {e}")
        result["error"] = str(e)
    
    return result


async def main():
    """Main ingestion process"""
    print("=" * 80)
    print("🚀 KALKI Full Ingestion Pipeline")
    print("=" * 80)
    print()
    
    # Find PDFs
    pdf_archive = Path("data/pdf_archive")
    if not pdf_archive.exists():
        print(f"❌ PDF archive not found: {pdf_archive}")
        return
    
    pdf_files = list(pdf_archive.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDFs found in {pdf_archive}")
        return
    
    print(f"📁 Found {len(pdf_files)} PDFs in {pdf_archive}")
    print()
    
    # Initialize knowledge extractor
    print("🔧 Initializing systems...")
    knowledge_extractor = KnowledgeExtractor()
    print("✅ Knowledge extractor ready")
    print()
    
    # Process each PDF
    results = []
    start_time = time.time()
    
    for idx, pdf_path in enumerate(pdf_files, 1):
        print(f"[{idx}/{len(pdf_files)}] Processing: {pdf_path.name}")
        print("-" * 80)
        
        pdf_start = time.time()
        result = await ingest_pdf_full(pdf_path, knowledge_extractor)
        pdf_elapsed = time.time() - pdf_start
        
        results.append(result)
        
        if result["success"]:
            print(f"✅ Success ({pdf_elapsed:.1f}s)")
            print(f"   Vector DB: {result['vector_chunks']} chunks")
            print(f"   Knowledge: {sum(result['knowledge_extracted'].values())} items extracted")
            for k, v in result['knowledge_extracted'].items():
                if v > 0:
                    print(f"      • {k}: {v}")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
        
        print()
    
    # Summary
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r["success"])
    
    print("=" * 80)
    print("📊 Ingestion Summary")
    print("=" * 80)
    print(f"✅ Successful: {successful}/{len(pdf_files)}")
    print(f"❌ Failed: {len(pdf_files) - successful}/{len(pdf_files)}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print()
    
    # Knowledge extraction totals
    total_knowledge = {}
    for result in results:
        if result["success"]:
            for k, v in result["knowledge_extracted"].items():
                total_knowledge[k] = total_knowledge.get(k, 0) + v
    
    if total_knowledge:
        print("📚 Total Knowledge Extracted:")
        for k, v in sorted(total_knowledge.items(), key=lambda x: x[1], reverse=True):
            if v > 0:
                print(f"   • {k}: {v}")
        print()
    
    # Vector DB stats
    total_chunks = sum(r["vector_chunks"] for r in results if r["success"])
    print(f"📦 Vector DB: {total_chunks} total chunks ingested")
    print()
    
    # Verify ingestion
    print("🔍 Verifying ingestion...")
    try:
        vectordb = VectorDBManager()
        # Try to get collection count
        print(f"✅ Vector DB ready")
    except Exception as e:
        print(f"⚠️  Could not verify vector DB: {e}")
    
    # Check knowledge databases
    print("\n📊 Knowledge Database Status:")
    for db_name in ["formulas", "materials", "design_rules", "code_requirements", 
                    "span_tables", "procedures", "inspection_criteria", 
                    "cost_data", "load_parameters", "decision_trees"]:
        db_path = Path(f"data/knowledge/{db_name}.db")
        if db_path.exists():
            import sqlite3
            try:
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                cursor.execute(f"SELECT COUNT(*) FROM {db_name}")
                count = cursor.fetchone()[0]
                conn.close()
                status = "✅" if count > 0 else "⚠️"
                print(f"   {status} {db_name}: {count} items")
            except Exception as e:
                print(f"   ⚠️  {db_name}: Error checking ({e})")
        else:
            print(f"   ❌ {db_name}: Database not found")
    
    print()
    print("=" * 80)
    print("🎉 Ingestion Complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

