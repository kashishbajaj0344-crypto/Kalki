#!/usr/bin/env python3
"""
Process building design PDFs one folder at a time
"""

import os
import subprocess
import sys
from pathlib import Path

def get_all_folders(base_path):
    """Get all subdirectories"""
    folders = []
    for item in sorted(os.listdir(base_path)):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and 'Academia.edu_Bundle' in item:
            folders.append(item_path)
    return folders

def count_pdfs_in_folder(folder_path):
    """Count PDFs in folder"""
    count = 0
    for root, dirs, files in os.walk(folder_path):
        count += sum(1 for f in files if f.lower().endswith('.pdf'))
    return count

def main():
    base_path = "pdfs/building designs"
    
    print("📚 KALKI Building Designs Ingestion")
    print("=" * 80)
    
    # Get all folders
    folders = get_all_folders(base_path)
    
    if not folders:
        print("❌ No Academia.edu_Bundle folders found")
        return
    
    print(f"✅ Found {len(folders)} folders to process")
    print()
    
    # Show folder list
    total_pdfs = 0
    for idx, folder in enumerate(folders, 1):
        pdf_count = count_pdfs_in_folder(folder)
        total_pdfs += pdf_count
        folder_name = Path(folder).name
        print(f"  [{idx:2d}] {folder_name[:60]:60s} ({pdf_count:3d} PDFs)")
    
    print()
    print(f"📊 Total: {total_pdfs} PDFs across {len(folders)} folders")
    print("=" * 80)
    print()
    
    # Process each folder
    for idx, folder in enumerate(folders, 1):
        folder_name = Path(folder).name
        pdf_count = count_pdfs_in_folder(folder)
        
        print(f"\n{'='*80}")
        print(f"📁 FOLDER {idx}/{len(folders)}: {folder_name[:50]}")
        print(f"📄 PDFs: {pdf_count}")
        print(f"{'='*80}\n")
        
        # Run ingestion
        cmd = ['python3', 'ingest_folder.py', folder]
        
        try:
            result = subprocess.run(cmd, check=False)
            
            if result.returncode == 0:
                print(f"\n✅ Folder {idx} complete")
            else:
                print(f"\n⚠️  Folder {idx} finished with errors (code {result.returncode})")
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            print(f"Resume from folder {idx} by running:")
            print(f'  python3 process_all_folders.py --start {idx}')
            return
        except Exception as e:
            print(f"\n❌ Error processing folder {idx}: {e}")
            continue
    
    print("\n" + "=" * 80)
    print("🎉 ALL FOLDERS PROCESSED!")
    print("=" * 80)

if __name__ == '__main__':
    main()
