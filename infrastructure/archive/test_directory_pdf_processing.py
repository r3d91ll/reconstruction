#!/usr/bin/env python3
"""
Test the directory-based PDF processing approach
First runs a dry-run to see what would happen
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Test directory PDF processing"""
    print("=" * 80)
    print("Directory-based PDF Processing Test")
    print("=" * 80)
    
    # Check for password
    if not os.environ.get('ARANGO_PASSWORD'):
        print("ERROR: ARANGO_PASSWORD environment variable not set")
        sys.exit(1)
    
    pdf_dir = Path('/mnt/data-cold/arxiv_data/pdf')
    
    # First check if directory exists and has PDFs
    if not pdf_dir.exists():
        print(f"ERROR: PDF directory not found: {pdf_dir}")
        print("\nPlease ensure PDFs are placed in this directory for processing")
        sys.exit(1)
        
    pdf_count = len(list(pdf_dir.glob("*.pdf")))
    print(f"Found {pdf_count} PDF files in {pdf_dir}")
    
    if pdf_count == 0:
        print("\nNo PDFs found. Please place PDFs in the directory first.")
        sys.exit(0)
        
    # Show first few PDFs
    print("\nFirst 10 PDFs found:")
    for i, pdf in enumerate(sorted(pdf_dir.glob("*.pdf"))[:10]):
        print(f"  {i+1}. {pdf.name}")
        
    print("\n" + "-" * 40)
    print("Running DRY RUN first (no actual processing)")
    print("-" * 40)
    
    # Run dry run
    dry_run_params = [
        'python3', 'process_pdfs_from_directory.py',
        '--dry-run',
        '--max-pdfs', '100',  # Limit for test
        '--batch-size', '10'
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
            print("\nWarnings/Info:")
            print(result.stderr[:1000])
            
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Dry run failed with exit code {e.returncode}")
        print(e.stdout)
        print(e.stderr)
        sys.exit(1)
        
    # Ask if user wants to proceed with actual processing
    print("\n" + "=" * 80)
    response = input("\nDo you want to proceed with ACTUAL processing? (yes/no): ")
    
    if response.lower() != 'yes':
        print("Aborted.")
        sys.exit(0)
        
    print("\n" + "-" * 40)
    print("Running ACTUAL processing (will modify files)")
    print("-" * 40)
    
    # Run actual processing
    process_params = [
        'python3', 'process_pdfs_from_directory.py',
        '--max-pdfs', '10',  # Start with just 10 for safety
        '--batch-size', '5'
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
            print("\nProcessing log:")
            print(result.stderr[:2000])
            
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Processing failed with exit code {e.returncode}")
        print(e.stdout)
        print(e.stderr)
        sys.exit(1)
        
    print("\n" + "=" * 80)
    print("Test complete!")
    print("\nNext steps:")
    print("1. Check the results above")
    print("2. Verify processed PDFs were deleted from directory")
    print("3. Check that orphaned PDFs (no metadata) remain")
    print("4. Run full processing without --max-pdfs limit when ready")
    print("=" * 80)

if __name__ == '__main__':
    main()