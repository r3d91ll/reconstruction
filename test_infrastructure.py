#!/usr/bin/env python3
"""
Test Infrastructure Module

Quick test to verify the infrastructure components work correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from irec_infrastructure import DocumentProcessor
from irec_infrastructure.data import ArxivLoader
from irec_infrastructure.embeddings import LateChucker
from irec_infrastructure.database import ArangoClient
from irec_infrastructure.monitoring import ProgressTracker


def test_components():
    """Test individual infrastructure components."""
    
    print("Testing Infrastructure Components")
    print("=" * 60)
    
    # Test 1: ArxivLoader
    print("\n1. Testing ArxivLoader...")
    try:
        loader = ArxivLoader("/mnt/data/arxiv_data")
        papers = loader.load_papers(num_papers=10, sampling_strategy="random")
        print(f"   ✓ Loaded {len(papers)} papers")
        if papers:
            print(f"   ✓ Sample paper: {papers[0]['id']}")
    except Exception as e:
        print(f"   ✗ ArxivLoader failed: {e}")
    
    # Test 2: LateChucker
    print("\n2. Testing LateChucker...")
    try:
        chunker = LateChucker(use_gpu=False)  # CPU for quick test
        print("   ✓ LateChucker initialized")
    except Exception as e:
        print(f"   ✗ LateChucker failed: {e}")
    
    # Test 3: ArangoClient
    print("\n3. Testing ArangoClient...")
    try:
        client = ArangoClient()
        print("   ✓ ArangoClient initialized")
        # Don't actually connect in test
    except Exception as e:
        print(f"   ✗ ArangoClient failed: {e}")
    
    # Test 4: ProgressTracker
    print("\n4. Testing ProgressTracker...")
    try:
        with ProgressTracker(total=100, desc="Test progress", disable=True) as tracker:
            for i in range(100):
                tracker.update(1)
        print("   ✓ ProgressTracker works")
    except Exception as e:
        print(f"   ✗ ProgressTracker failed: {e}")
    
    # Test 5: DocumentProcessor integration
    print("\n5. Testing DocumentProcessor...")
    try:
        from irec_infrastructure.embeddings import JinaConfig
        from irec_infrastructure.data import ProcessingConfig
        
        # Initialize without Jina API key for test
        processor = DocumentProcessor(
            config=ProcessingConfig(use_gpu=False)
        )
        print("   ✓ DocumentProcessor initialized")
    except Exception as e:
        print(f"   ✗ DocumentProcessor failed: {e}")
    
    print("\n" + "=" * 60)
    print("Infrastructure test complete!")


if __name__ == "__main__":
    test_components()