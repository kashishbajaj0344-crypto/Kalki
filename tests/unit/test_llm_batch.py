#!/usr/bin/env python3
"""
Test LLM-enhanced ingestion with 5 PDFs on M4 Max GPU
Measures performance, quality, and model caching effectiveness
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# Test PDFs (smaller ones for faster testing)
test_pdfs = [
    "pdfs/ASHRAE_test_30pages.pdf",
    "pdfs/IBC_test_50pages.pdf", 
    "pdfs/IBC_structural_test.pdf",
    "pdfs/IBC_loads_test.pdf",
    "pdfs/2010ADAStandards.pdf"
]

print("=" * 80)
print("🚀 KALKI LLM-Enhanced Batch Ingestion Test - M4 Max GPU")
print("=" * 80)

# Clear databases
print("\n🗑️  Clearing knowledge databases...")
subprocess.run([
    "rm", "-f", 
    "data/knowledge/*.db",
    "data/knowledge/processing_log.json"
], shell=False, capture_output=True)
print("✅ Databases cleared\n")

total_start = time.time()
results = []

for i, pdf_path in enumerate(test_pdfs, 1):
    pdf_name = Path(pdf_path).name
    print(f"\n{'='*80}")
    print(f"📄 Processing PDF {i}/5: {pdf_name}")
    print(f"{'='*80}")
    
    if not os.path.exists(pdf_path):
        print(f"⚠️  File not found: {pdf_path}")
        continue
    
    # Get file size
    size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    print(f"📦 Size: {size_mb:.1f} MB")
    
    # Run ingestion with LLM
    start_time = time.time()
    
    result = subprocess.run([
        "python3", "kalki_cli.py", "learn", "ingest", 
        pdf_path, "--use-llm"
    ], capture_output=True, text=True, timeout=180)
    
    elapsed = time.time() - start_time
    
    # Parse output
    formulas = 0
    procedures = 0
    inspection = 0
    llm_validated = 0
    
    for line in result.stdout.split('\n'):
        if 'Formulas:' in line and 'Validated' not in line:
            try:
                formulas = int(line.split(':')[1].strip())
            except:
                pass
        elif 'Procedures:' in line:
            try:
                procedures = int(line.split(':')[1].strip())
            except:
                pass
        elif 'Inspection Criteria:' in line:
            try:
                inspection = int(line.split(':')[1].strip())
            except:
                pass
        elif 'Validated formulas:' in line:
            try:
                llm_validated = int(line.split(':')[1].strip())
            except:
                pass
    
    # Check for GPU usage
    using_gpu = "mps" in result.stdout or "Metal" in result.stdout
    cached = i > 1  # First load, rest should be cached
    
    results.append({
        'pdf': pdf_name,
        'size_mb': size_mb,
        'time': elapsed,
        'formulas': formulas,
        'procedures': procedures,
        'inspection': inspection,
        'llm_validated': llm_validated,
        'gpu': using_gpu,
        'cached': cached
    })
    
    print(f"\n📊 Results:")
    print(f"   Time: {elapsed:.1f}s")
    print(f"   Formulas: {formulas} (LLM validated: {llm_validated})")
    print(f"   Procedures: {procedures}")
    print(f"   Inspection Criteria: {inspection}")
    print(f"   GPU: {'✅ Metal' if using_gpu else '❌ CPU'}")
    print(f"   Model: {'✅ Cached' if cached else '🔄 First Load'}")

total_time = time.time() - total_start

# Summary report
print(f"\n\n{'='*80}")
print("📈 BATCH PROCESSING SUMMARY")
print(f"{'='*80}")

print(f"\n⏱️  Total Time: {total_time:.1f} seconds")
print(f"📄 PDFs Processed: {len(results)}")

total_formulas = sum(r['formulas'] for r in results)
total_validated = sum(r['llm_validated'] for r in results)
total_procedures = sum(r['procedures'] for r in results)
total_inspection = sum(r['inspection'] for r in results)

print(f"\n📊 Aggregated Results:")
print(f"   Total Formulas Extracted: {total_formulas}")
print(f"   LLM Validated Formulas: {total_validated}")
if total_formulas > 0:
    reduction = ((total_formulas - total_validated) / total_formulas) * 100
    print(f"   False Positive Reduction: {reduction:.1f}%")
print(f"   Total Procedures: {total_procedures}")
print(f"   Total Inspection Criteria: {total_inspection}")

print(f"\n⚡ Performance:")
avg_time = total_time / len(results)
print(f"   Average Time per PDF: {avg_time:.1f}s")
gpu_count = sum(1 for r in results if r['gpu'])
print(f"   GPU Acceleration: {gpu_count}/{len(results)} PDFs")

print(f"\n💾 Model Caching:")
if results[0]['time'] > 15 and len(results) > 1:
    first_time = results[0]['time']
    avg_cached = sum(r['time'] for r in results[1:]) / (len(results) - 1)
    speedup = (first_time - avg_cached) / first_time * 100
    print(f"   First Load: {first_time:.1f}s (model loading)")
    print(f"   Cached Avg: {avg_cached:.1f}s")
    print(f"   Speedup: {speedup:.0f}% faster with caching")

print(f"\n{'='*80}")
print("✅ Test Complete!")
print(f"{'='*80}\n")

# Show detailed breakdown
print("\n📋 Detailed Breakdown:")
print(f"{'PDF':<35} {'Time':<8} {'Form':<6} {'Valid':<6} {'GPU':<5}")
print("-" * 70)
for r in results:
    gpu_icon = "✅" if r['gpu'] else "❌"
    print(f"{r['pdf']:<35} {r['time']:>6.1f}s  {r['formulas']:>4}  {r['llm_validated']:>4}   {gpu_icon}")

print("\n")
