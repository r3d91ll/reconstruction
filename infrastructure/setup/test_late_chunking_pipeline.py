#!/usr/bin/env python3
"""
Test script for PDF processing with late chunking
Tests the optimized pipeline with Jina's late chunking feature
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Test late chunking PDF processing"""
    print("=" * 80)
    print("PDF Processing with Late Chunking Test")
    print("=" * 80)
    print("\nKey improvements:")
    print("- No intermediate disk writes (all in memory)")
    print("- Late chunking: send full markdown to Jina")
    print("- Jina returns multiple chunk embeddings in one pass")
    print("- More efficient GPU utilization")
    print("=" * 80)
    
    # Check for password
    if not os.environ.get('ARANGO_PASSWORD'):
        print("ERROR: ARANGO_PASSWORD environment variable not set")
        sys.exit(1)
    
    pdf_dir = Path('/mnt/data-cold/arxiv_data/pdf')
    
    # First check if directory exists and has PDFs
    if not pdf_dir.exists():
        print(f"ERROR: PDF directory not found: {pdf_dir}")
        sys.exit(1)
        
    pdf_count = len(list(pdf_dir.glob("*.pdf")))
    print(f"\nFound {pdf_count} PDF files in {pdf_dir}")
    
    if pdf_count == 0:
        print("\nNo PDFs found. Please place PDFs in the directory first.")
        sys.exit(0)
        
    # Show first few PDFs
    print("\nFirst 5 PDFs found:")
    for i, pdf in enumerate(sorted(pdf_dir.glob("*.pdf"))[:5]):
        print(f"  {i+1}. {pdf.name}")
        
    print("\n" + "-" * 40)
    print("Running DRY RUN first")
    print("-" * 40)
    
    # Check if script exists
    script_path = 'process_pdfs_directory_late_chunking.py'
    if not os.path.exists(script_path):
        print(f"ERROR: Script '{script_path}' not found in current directory")
        return False
        
    # Run dry run with just 5 PDFs
    dry_run_params = [
        'python3', script_path,
        '--dry-run',
        '--max-pdfs', '5',
        '--batch-size', '2',
        '--chunk-size', '512',
        '--chunk-stride', '256'
    ]
    
    try:
        result = subprocess.run(
            dry_run_params,
            capture_output=True,
            text=True,
            check=True
        )
        
        print(result.stdout)
        
        if result.stderr:
            print("\nLog output:")
            print(result.stderr[:1000])
            
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Dry run failed with exit code {e.returncode}")
        print(e.stdout)
        print(e.stderr)
        sys.exit(1)
        
    # Ask if user wants to proceed
    print("\n" + "=" * 80)
    response = input("\nProceed with ACTUAL processing of 5 PDFs? (yes/no): ")
    
    if response.lower() != 'yes':
        print("Aborted.")
        sys.exit(0)
        
    print("\n" + "-" * 40)
    print("Running actual processing with late chunking")
    print("-" * 40)
    
    # Run actual processing
    process_params = [
        'python3', 'process_pdfs_directory_late_chunking.py',
        '--max-pdfs', '5',
        '--batch-size', '2',
        '--chunk-size', '512',
        '--chunk-stride', '256'
    ]
    
    try:
        result = subprocess.run(
            process_params,
            capture_output=True,
            text=True,
            check=True
        )
        
        print(result.stdout)
        
        if result.stderr:
            print("\nProcessing details:")
            # Show more of the log for actual processing
            print(result.stderr[:3000])
            
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Processing failed with exit code {e.returncode}")
        print(e.stdout)
        print(e.stderr)
        sys.exit(1)
        
    print("\n" + "=" * 80)
    print("Test complete!")
    print("\nLate chunking advantages:")
    print("1. Single forward pass through Jina model")
    print("2. Better semantic understanding (full context)")
    print("3. No manual chunking overhead")
    print("4. Potential for NVLink optimization")
    print("\nCheck the results and run full processing when ready.")
    print("=" * 80)

if __name__ == '__main__':
    main()