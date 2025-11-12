#!/usr/bin/env python3
"""
KALKI Simple Ingestion Script
Uses HybridKnowledgeSystem which handles both vector DB and knowledge extraction
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.hybrid_learning_system import get_hybrid_system

try:
    import fitz  # PyMuPDF
except ImportError:
    print("⚠️  PyMuPDF not available, trying pdfplumber...")
    fitz = None


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF"""
    text = ""
    use_fitz = fitz is not None
    
    if use_fitz:
        try:
            doc = fitz.open(str(pdf_path))
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            print(f"   ⚠️  PyMuPDF failed: {e}, trying pdfplumber...")
            use_fitz = False
    
    if not use_fitz:
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
            return text
        except Exception as e:
            print(f"   ❌ PDF extraction failed: {e}")
            return ""
    
    return text


async def main():
    """Main ingestion process"""
    print("=" * 80)
    print("🚀 KALKI Ingestion Pipeline")
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
    
    print(f"📁 Found {len(pdf_files)} PDFs")
    print()
    
    # Initialize hybrid system
    print("🔧 Initializing Hybrid Knowledge System...")
    hybrid_system = get_hybrid_system()
    print("✅ System ready")
    print()
    
    # Also need to ingest into vector DB separately
    print("📥 Setting up Vector DB ingestion...")
    try:
        from modules.learning.vectordb import VectorDBManager
        vectordb = VectorDBManager()
        print("✅ Vector DB ready")
    except Exception as e:
        print(f"⚠️  Vector DB setup warning: {e}")
        vectordb = None
    
    print()
    
    # Process each PDF
    results = []
    start_time = time.time()
    
    for idx, pdf_path in enumerate(pdf_files, 1):
        print(f"[{idx}/{len(pdf_files)}] 📄 {pdf_path.name}")
        print("-" * 80)
        
        pdf_start = time.time()
        
        try:
            # Extract text
            print("   📄 Extracting text...")
            pdf_content = extract_text_from_pdf(pdf_path)
            
            if not pdf_content or len(pdf_content.strip()) < 100:
                print(f"   ⚠️  Minimal text extracted ({len(pdf_content)} chars)")
                results.append({
                    "pdf": pdf_path.name,
                    "success": False,
                    "error": "Minimal text"
                })
                continue
            
            print(f"   ✅ Extracted {len(pdf_content)} characters")
            
            # Ingest into vector DB
            if vectordb:
                print("   📥 Ingesting into Vector DB...")
                try:
                    # Simple chunking and direct vector DB add
                    # Chunk text into ~512 token chunks
                    chunk_size = 512
                    words = pdf_content.split()
                    chunks = []
                    current_chunk = []
                    current_size = 0
                    
                    for word in words:
                        word_size = len(word.split())
                        if current_size + word_size > chunk_size and current_chunk:
                            chunks.append(" ".join(current_chunk))
                            current_chunk = [word]
                            current_size = word_size
                        else:
                            current_chunk.append(word)
                            current_size += word_size
                    
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                    
                    # Add to vector DB
                    texts = chunks
                    metadatas = [{"source": str(pdf_path), "chunk_id": f"{pdf_path.name}_chunk_{i}", 
                                "page": "unknown"} for i in range(len(texts))]
                    
                    vectordb.add_document(str(pdf_path), texts, metadatas)
                    print(f"   ✅ Vector DB: Added {len(texts)} chunks")
                    
                except Exception as e:
                    print(f"   ⚠️  Vector DB ingestion failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Extract knowledge
            print("   🔍 Extracting structured knowledge...")
            extraction_results = hybrid_system.ingest_pdf(
                str(pdf_path),
                pdf_content,
                archive=False,  # Already in archive
                use_llm_enhancements=True
            )
            
            pdf_elapsed = time.time() - pdf_start
            
            # Show results
            print(f"   ✅ Complete ({pdf_elapsed:.1f}s)")
            print(f"   📊 Knowledge extracted:")
            for k, v in extraction_results.items():
                if isinstance(v, int) and v > 0:
                    print(f"      • {k}: {v}")
            
            results.append({
                "pdf": pdf_path.name,
                "success": True,
                "extraction": extraction_results,
                "time": pdf_elapsed
            })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "pdf": pdf_path.name,
                "success": False,
                "error": str(e)
            })
        
        print()
    
    # Summary
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r.get("success"))
    
    print("=" * 80)
    print("📊 Ingestion Summary")
    print("=" * 80)
    print(f"✅ Successful: {successful}/{len(pdf_files)}")
    print(f"❌ Failed: {len(pdf_files) - successful}/{len(pdf_files)}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print()
    
    # Knowledge totals
    total_knowledge = {}
    for result in results:
        if result.get("success") and "extraction" in result:
            for k, v in result["extraction"].items():
                if isinstance(v, int):
                    total_knowledge[k] = total_knowledge.get(k, 0) + v
    
    if total_knowledge:
        print("📚 Total Knowledge Extracted:")
        for k, v in sorted(total_knowledge.items(), key=lambda x: x[1], reverse=True):
            if v > 0:
                print(f"   • {k}: {v}")
        print()
    
    # Verify databases
    print("🔍 Verifying databases...")
    try:
        import sqlite3
        knowledge_path = Path("data/knowledge")
        
        for db_name in ["formulas", "materials", "design_rules", "code_requirements"]:
            db_path = knowledge_path / f"{db_name}.db"
            if db_path.exists():
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {db_name}")
                    count = cursor.fetchone()[0]
                    status = "✅" if count > 0 else "⚠️"
                    print(f"   {status} {db_name}: {count} items")
                except:
                    print(f"   ⚠️  {db_name}: Error reading")
                conn.close()
    except Exception as e:
        print(f"   ⚠️  Could not verify: {e}")
    
    print()
    print("=" * 80)
    print("🎉 Ingestion Complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

