#!/usr/bin/env python3
"""
Test script for single collection approach with 1000 documents
Validates performance improvements and metrics collection
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def main():
    """Run 1000 document test"""
    print("=" * 80)
    print("Single Collection Proof of Concept Test")
    print("Target: Process 1000 documents at 25-30 docs/sec")
    print("=" * 80)
    
    # Check for password
    if not os.environ.get('ARANGO_PASSWORD'):
        print("ERROR: ARANGO_PASSWORD environment variable not set")
        sys.exit(1)
    
    # Check if the script exists
    script_path = Path('process_abstracts_single_collection.py')
    if not script_path.exists():
        print(f"ERROR: Script '{script_path}' not found in current directory")
        print(f"Current directory: {Path.cwd()}")
        print("Please run this test from the same directory as process_abstracts_single_collection.py")
        sys.exit(1)
    
    # Test parameters
    test_params = [
        'python3', 'process_abstracts_single_collection.py',
        '--max-abstracts', '1000',
        '--db-name', 'arxiv_test_single_collection',
        '--clean-start',  # Fresh start for testing
        '--metadata-workers', '2',  # Reduced for test
        '--embedding-workers', '1',
        '--batch-size', '100',  # Smaller batches for 1K test
        '--embedding-gpu', '0'
    ]
    
    print("\nTest configuration:")
    print(f"  Documents: 1,000")
    print(f"  Database: arxiv_test_single_collection")
    print(f"  Batch size: 100")
    print(f"  Workers: 2 metadata, 1 embedding")
    print(f"  GPU: 0")
    
    print("\nStarting test run...")
    start_time = time.time()
    
    try:
        # Run the pipeline
        result = subprocess.run(
            test_params,
            capture_output=True,
            text=True,
            check=True
        )
        
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 80)
        print("TEST COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Total time: {elapsed:.2f} seconds")
        print(f"Average rate: {1000/elapsed:.1f} docs/sec")
        
        # Check if we met our target
        if 1000/elapsed >= 20:
            print("\n✓ Performance target MET (≥20 docs/sec for proof of concept)")
        else:
            print("\n✗ Performance target NOT MET")
            
        # Display output
        print("\nPipeline output:")
        print("-" * 40)
        print(result.stdout)
        
        if result.stderr:
            print("\nErrors/Warnings:")
            print("-" * 40)
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Pipeline failed with exit code {e.returncode}")
        print("\nStdout:")
        print(e.stdout)
        print("\nStderr:")
        print(e.stderr)
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
        
    print("\n" + "=" * 80)
    print("Next steps:")
    print("1. Review the metrics in abstracts_single_collection.log")
    print("2. Verify single collection structure in ArangoDB")
    print("3. If performance target met, proceed to 10K document test")
    print("=" * 80)

if __name__ == '__main__':
    main()