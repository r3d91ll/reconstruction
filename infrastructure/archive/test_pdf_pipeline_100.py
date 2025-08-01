#!/usr/bin/env python3
"""
Test script for PDF processing pipeline with 100 documents
Validates dual-GPU approach before full production run
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def main():
    """Run 100 document PDF processing test"""
    print("=" * 80)
    print("PDF Processing Pipeline Test - 100 Documents")
    print("Strategy: Dual GPU (Docling on GPU0, Jina on GPU1)")
    print("Expected rate: ~2 PDFs/second (limited by Docling)")
    print("=" * 80)
    
    # Check for password
    if not os.environ.get('ARANGO_PASSWORD'):
        print("ERROR: ARANGO_PASSWORD environment variable not set")
        sys.exit(1)
    
    # Check if the script exists
    script_path = Path('process_pdfs_dual_gpu.py')
    if not script_path.exists():
        print(f"ERROR: Script '{script_path}' not found in current directory: {Path.cwd()}")
        sys.exit(1)
    
    # First, we need to mark some PDFs as unprocessed for testing
    print("\nPreparing test documents...")
    print("This will mark the first 100 documents with PDF tar sources for processing")
    
    # Test parameters
    test_params = [
        'python3', 'process_pdfs_dual_gpu.py',
        '--max-pdfs', '100',
        '--db-name', 'arxiv_single_collection',  # Use production DB
        '--batch-size', '10',  # Smaller batches for testing
        '--docling-gpu', '0',
        '--embedding-gpu', '1',
        '--working-dir', '/tmp/arxiv_pdf_test_100'
    ]
    
    print("\nTest configuration:")
    print(f"  Documents: 100")
    print(f"  Batch size: 10 PDFs")
    print(f"  GPU 0: Docling PDF→Markdown")
    print(f"  GPU 1: Jina embeddings")
    print(f"  Expected time: ~50 seconds @ 2 PDFs/sec")
    
    print("\nStarting test run...")
    start_time = time.time()
    
    try:
        # Run the pipeline with timeout
        result = subprocess.run(
            test_params,
            capture_output=True,
            text=True,
            check=True,
            timeout=300  # 5 minute timeout for 100 PDFs
        )
        
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 80)
        print("TEST COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Total time: {elapsed:.2f} seconds")
        print(f"Average rate: {100/elapsed:.2f} PDFs/sec")
        
        # Check for GPU errors in output
        if "GPU OOM error" in result.stderr:
            oom_count = result.stderr.count("GPU OOM error")
            print(f"\n⚠️  Found {oom_count} GPU OOM errors - may need to adjust batch sizes")
        else:
            print("\n✓ No GPU OOM errors detected")
            
        # Display output
        print("\nPipeline output:")
        print("-" * 40)
        print(result.stdout[-2000:])  # Last 2000 chars
        
        if result.stderr:
            print("\nErrors/Warnings:")
            print("-" * 40)
            print(result.stderr[:2000])
            
    except subprocess.TimeoutExpired:
        print("\nERROR: Pipeline timed out after 5 minutes")
        print("This suggests the pipeline may be hanging or running too slowly")
        sys.exit(1)
        
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Pipeline failed with exit code {e.returncode}")
        print("\nStdout:")
        print(e.stdout[:1000])
        print("\nStderr:")
        print(e.stderr[:2000])
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
        
    print("\n" + "=" * 80)
    print("Analysis:")
    print("- Docling on GPU 0 for PDF→Markdown conversion")
    print("- Jina on GPU 1 for chunk embeddings")
    print("- Pipeline should achieve ~2 PDFs/sec (Docling bottleneck)")
    print("- Full 2.7M PDFs would take ~16 days at this rate")
    print("=" * 80)

if __name__ == '__main__':
    main()