#!/usr/bin/env python3
"""
Test script for multi-worker PDF processing
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Test multi-worker PDF processing"""
    print("=" * 80)
    print("Multi-Worker PDF Processing Test")
    print("=" * 80)
    print("\nKey features:")
    print("- 4 Docling workers on GPU 0 (parallel PDF conversion)")
    print("- 1 Jina worker on GPU 1 (late chunking)")
    print("- Each Docling worker gets ~10GB GPU memory")
    print("- Expected ~4x speedup over single worker")
    print("=" * 80)
    
    if not os.environ.get('ARANGO_PASSWORD'):
        print("ERROR: ARANGO_PASSWORD environment variable not set")
        sys.exit(1)
    
    # Test with 10 PDFs
    test_params = [
        'python3', 'process_pdfs_multi_worker.py',
        '--max-pdfs', '10',
        '--docling-workers', '4'
    ]
    
    print("\nRunning test with 10 PDFs and 4 Docling workers...")
    print("Expected time: ~15-20 seconds (vs 60 seconds with single worker)")
    print("-" * 40)
    
    try:
        # Run with real-time output
        process = subprocess.Popen(
            test_params,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Print output in real-time
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line.rstrip())
                
        process.wait()
        
        if process.returncode == 0:
            print("\n✓ Test completed successfully!")
        else:
            print(f"\n✗ Test failed with exit code {process.returncode}")
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        process.terminate()
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()