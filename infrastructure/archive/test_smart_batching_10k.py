#!/usr/bin/env python3
"""
Test script for smart batching approach with 10K documents
Uses 100-doc batches with max 7 concurrent = 700 texts total (< 849 limit)
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def main():
    """Run 10K document test with smart batching"""
    print("=" * 80)
    print("Smart Batching Test - 10,000 Documents")
    print("Strategy: 100 docs/batch, max 7 concurrent batches")
    print("Maximum GPU load: 700 texts (safe under 849 limit)")
    print("=" * 80)
    
    # Check for password
    if not os.environ.get('ARANGO_PASSWORD'):
        print("ERROR: ARANGO_PASSWORD environment variable not set")
        sys.exit(1)
    
    # Check if the script exists
    script_path = Path('process_abstracts_single_collection.py')
    if not script_path.exists():
        print(f"ERROR: Script '{script_path}' not found in current directory: {Path.cwd()}")
        sys.exit(1)
    
    # Test parameters
    test_params = [
        'python3', 'process_abstracts_single_collection.py',
        '--max-abstracts', '10000',
        '--db-name', 'arxiv_test_smart_batch_10k',
        '--clean-start',
        '--metadata-workers', '4',
        '--embedding-workers', '1',
        '--batch-size', '100',  # Override if needed
        '--embedding-gpu', '0'
    ]
    
    print("\nTest configuration:")
    print(f"  Documents: 10,000")
    print(f"  Batch size: 100")
    print(f"  Max concurrent batches: 7")
    print(f"  Workers: 4 metadata, 1 embedding")
    print(f"  Expected behavior: No OOM errors")
    
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
        print(f"Average rate: {10000/elapsed:.1f} docs/sec")
        
        # Check for OOM errors in output
        if "GPU OOM error" in result.stderr:
            oom_count = result.stderr.count("GPU OOM error")
            print(f"\n⚠️  Found {oom_count} OOM errors - strategy needs adjustment")
        else:
            print("\n✓ No OOM errors detected - batching strategy working well")
            
        # Display output
        print("\nPipeline output:")
        print("-" * 40)
        print(result.stdout)
        
        if result.stderr:
            print("\nErrors/Warnings:")
            print("-" * 40)
            # Only show first 2000 chars to avoid spam
            print(result.stderr[:2000])
            if len(result.stderr) > 2000:
                print(f"... (truncated, total {len(result.stderr)} chars)")
            
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
    print("- With 100-doc batches and 7 queue depth, max GPU load is 700 texts")
    print("- This is 82.6% of the 849 hard limit, leaving good safety margin")
    print("- Small batches process faster and clear memory more frequently")
    print("=" * 80)

if __name__ == '__main__':
    main()