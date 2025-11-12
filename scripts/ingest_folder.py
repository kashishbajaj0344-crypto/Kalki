#!/usr/bin/env python3
"""
KALKI - Single Folder PDF Ingestion
Processes all PDFs in a specific folder
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime

def find_pdfs_in_folder(folder_path):
    """Find all PDFs in specified folder (recursive)"""
    pdf_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return sorted(pdf_files)

def ingest_folder(folder_path, use_llm=True):
    """
    Ingest all PDFs in a folder
    
    Args:
        folder_path: Path to folder containing PDFs
        use_llm: Use LLM validation (default True)
    """
    
    print(f"📁 Processing Folder: {folder_path}")
    print("=" * 80)
    
    # Find all PDFs
    pdf_files = find_pdfs_in_folder(folder_path)
    
    if not pdf_files:
        print("❌ No PDF files found in folder")
        return
    
    print(f"✅ Found {len(pdf_files)} PDFs")
    print()
    
    # Process each PDF
    success_count = 0
    error_count = 0
    start_time = time.time()
    
    for idx, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{idx}/{len(pdf_files)}] Processing: {Path(pdf_path).name}")
        print("-" * 80)
        
        # Build command
        cmd = ['python3', 'kalki_cli.py', 'learn', 'ingest', pdf_path]
        if not use_llm:
            cmd.append('--no-llm')
        
        try:
            # Run ingestion
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per PDF
            )
            
            if result.returncode == 0:
                print(f"✅ Success")
                success_count += 1
            else:
                print(f"⚠️  Warning: Non-zero exit code")
                print(f"STDERR: {result.stderr[:200]}")
                error_count += 1
                
        except subprocess.TimeoutExpired:
            print(f"⏱️  Timeout (5 minutes)")
            error_count += 1
        except Exception as e:
            print(f"❌ Error: {e}")
            error_count += 1
        
        # Progress update
        elapsed = time.time() - start_time
        avg_time = elapsed / idx
        remaining = avg_time * (len(pdf_files) - idx)
        
        print(f"Progress: {success_count} success, {error_count} errors")
        print(f"Elapsed: {elapsed/60:.1f}m | Est. remaining: {remaining/60:.1f}m")
    
    # Final summary
    print("\n" + "=" * 80)
    print("📊 FOLDER INGESTION COMPLETE")
    print("=" * 80)
    print(f"Folder: {folder_path}")
    print(f"Total PDFs: {len(pdf_files)}")
    print(f"✅ Success: {success_count}")
    print(f"❌ Errors: {error_count}")
    print(f"⏱️  Total time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"📈 Average: {(time.time() - start_time)/len(pdf_files):.1f} seconds per PDF")
    print("=" * 80)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 ingest_folder.py <folder_path> [--no-llm]")
        print("\nExample:")
        print('  python3 ingest_folder.py "pdfs/building designs/Academia.edu_Bundle_-_BIM_handbook"')
        sys.exit(1)
    
    folder_path = sys.argv[1]
    use_llm = '--no-llm' not in sys.argv
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        sys.exit(1)
    
    ingest_folder(folder_path, use_llm)
