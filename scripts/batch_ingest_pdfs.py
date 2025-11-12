#!/usr/bin/env python3
"""
KALKI - Batch PDF Ingestion Script
Processes all PDFs in pdfs/ directory with progress tracking
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

def find_all_pdfs(base_dir='pdfs'):
    """Find all PDF files recursively"""
    pdf_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return sorted(pdf_files)

def categorize_pdf(pdf_path):
    """Categorize PDF by directory and filename"""
    path_lower = pdf_path.lower()
    
    if 'building' in path_lower or 'code' in path_lower or 'ibc' in path_lower or 'irc' in path_lower:
        return 'building_codes', 1  # Highest priority
    elif 'cad' in path_lower or 'design' in path_lower or 'engineering' in path_lower:
        return 'cad_engineering', 2
    elif 'construction' in path_lower or 'structural' in path_lower:
        return 'construction', 1
    elif 'computer' in path_lower or 'game' in path_lower:
        return 'computer_science', 3
    else:
        return 'general', 2

def batch_ingest_pdfs(priority_first=True, max_pdfs=None, resume_from=None):
    """
    Batch ingest PDFs with progress tracking
    
    Args:
        priority_first: Process high-priority PDFs first (building codes, construction)
        max_pdfs: Maximum number of PDFs to process (None = all)
        resume_from: Resume from specific PDF index
    """
    
    print("🚀 KALKI Batch PDF Ingestion")
    print("=" * 60)
    
    # Find all PDFs
    print("\n📁 Scanning for PDFs...")
    all_pdfs = find_all_pdfs()
    print(f"✅ Found {len(all_pdfs)} PDF files")
    
    # Categorize and sort by priority
    if priority_first:
        print("\n🎯 Sorting by priority (building codes first)...")
        categorized = [(pdf, *categorize_pdf(pdf)) for pdf in all_pdfs]
        categorized.sort(key=lambda x: (x[2], x[0]))  # Sort by priority, then name
        all_pdfs = [item[0] for item in categorized]
        
        # Show distribution
        categories = {}
        for pdf, cat, priority in categorized:
            categories[cat] = categories.get(cat, 0) + 1
        
        print("\n📊 PDF Distribution:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            print(f"   • {cat}: {count} PDFs")
    
    # Apply limits
    start_idx = resume_from or 0
    end_idx = min(len(all_pdfs), start_idx + max_pdfs) if max_pdfs else len(all_pdfs)
    pdfs_to_process = all_pdfs[start_idx:end_idx]
    
    print(f"\n🎯 Processing {len(pdfs_to_process)} PDFs (starting at #{start_idx + 1})")
    print(f"⏱️  Estimated time: {len(pdfs_to_process) * 3 / 60:.1f} minutes")
    print()
    
    # Create progress log
    log_file = f"data/ingestion_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    progress = {
        'start_time': datetime.now().isoformat(),
        'total_pdfs': len(pdfs_to_process),
        'processed': [],
        'failed': [],
        'stats': {}
    }
    
    # Process PDFs
    for idx, pdf_path in enumerate(pdfs_to_process, start=start_idx + 1):
        filename = os.path.basename(pdf_path)
        
        print(f"[{idx}/{len(all_pdfs)}] 📄 {filename[:50]}")
        print(f"         📂 {os.path.dirname(pdf_path)}")
        
        start_time = time.time()
        
        try:
            # Run ingestion command
            cmd = f'python3 kalki_cli.py learn ingest "{pdf_path}" 2>&1 | tail -20'
            result = os.popen(cmd).read()
            
            elapsed = time.time() - start_time
            
            # Extract results
            if '✅ PDF ingested successfully' in result:
                print(f"         ✅ Success ({elapsed:.1f}s)")
                
                # Try to extract extraction counts
                extraction_counts = {}
                for line in result.split('\n'):
                    if 'Formulas:' in line:
                        extraction_counts['formulas'] = int(line.split(':')[1].strip())
                    elif 'Materials:' in line:
                        extraction_counts['materials'] = int(line.split(':')[1].strip())
                    elif 'Design Rules:' in line:
                        extraction_counts['design_rules'] = int(line.split(':')[1].strip())
                
                if extraction_counts:
                    print(f"         📊 Extracted: {extraction_counts}")
                
                progress['processed'].append({
                    'pdf': pdf_path,
                    'time': elapsed,
                    'extractions': extraction_counts
                })
            else:
                print(f"         ⚠️  Completed with warnings ({elapsed:.1f}s)")
                progress['processed'].append({
                    'pdf': pdf_path,
                    'time': elapsed,
                    'status': 'warning'
                })
            
        except Exception as e:
            print(f"         ❌ Failed: {str(e)}")
            progress['failed'].append({
                'pdf': pdf_path,
                'error': str(e)
            })
        
        print()
        
        # Save progress every 10 PDFs
        if idx % 10 == 0:
            with open(log_file, 'w') as f:
                json.dump(progress, f, indent=2)
    
    # Final stats
    progress['end_time'] = datetime.now().isoformat()
    progress['duration_minutes'] = (time.time() - time.mktime(datetime.fromisoformat(progress['start_time']).timetuple())) / 60
    
    with open(log_file, 'w') as f:
        json.dump(progress, f, indent=2)
    
    print("\n" + "=" * 60)
    print("🎉 Batch Ingestion Complete!")
    print(f"✅ Successfully processed: {len(progress['processed'])}")
    print(f"❌ Failed: {len(progress['failed'])}")
    print(f"⏱️  Total time: {progress['duration_minutes']:.1f} minutes")
    print(f"📝 Log saved to: {log_file}")
    print()
    
    # Show knowledge base stats
    print("📊 Checking final knowledge base stats...")
    os.system('python3 kalki_cli.py learn stats 2>&1 | grep -A 10 "Knowledge Base"')

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch ingest PDFs into KALKI')
    parser.add_argument('--max', type=int, help='Maximum PDFs to process')
    parser.add_argument('--resume', type=int, help='Resume from PDF index')
    parser.add_argument('--no-priority', action='store_true', help='Disable priority sorting')
    
    args = parser.parse_args()
    
    batch_ingest_pdfs(
        priority_first=not args.no_priority,
        max_pdfs=args.max,
        resume_from=args.resume
    )
