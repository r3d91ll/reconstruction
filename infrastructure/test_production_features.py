#!/usr/bin/env python3
"""
Test script for production pipeline features.
Tests memory estimation, smart batching, and load balancing.
"""

import sys
import torch
from pathlib import Path

# Add infrastructure directory to path
infrastructure_dir = Path(__file__).parent
sys.path.append(str(infrastructure_dir))

from setup.process_abstracts_production import (
    AdvancedMemoryManager, SmartBatcher, DocumentInfo,
    EnhancedCheckpoint, PredictiveLoadBalancer, BatchInfo
)


def test_memory_estimation():
    """Test memory estimation for late chunking."""
    print("\n1. Testing Memory Estimation")
    print("-" * 40)
    
    manager = AdvancedMemoryManager()
    
    # Test cases
    test_cases = [
        ([100], "Single short doc"),
        ([1000], "Single medium doc"),
        ([5000], "Single long doc"),
        ([100, 100, 100, 100], "4 short docs"),
        ([1000, 2000, 1500, 1000], "4 mixed docs"),
        ([5000, 5000], "2 long docs (quadratic scaling test)")
    ]
    
    for tokens, description in test_cases:
        memory_gb = manager.estimate_batch_memory(tokens)
        print(f"{description}: {tokens} tokens -> {memory_gb:.3f} GB")
    
    # Test GPU fit check
    if torch.cuda.is_available():
        gpu_id = 0
        current = manager.get_current_usage(gpu_id)
        total = manager.get_total_memory(gpu_id)
        print(f"\nGPU {gpu_id} status: {current:.2f}/{total:.2f} GB used ({current/total*100:.1f}%)")
        
        can_fit = manager.can_fit_batch(gpu_id, [1000, 1000, 1000])
        print(f"Can fit 3x1000 token batch: {can_fit}")


def test_smart_batching():
    """Test smart batching with memory constraints."""
    print("\n2. Testing Smart Batching")
    print("-" * 40)
    
    manager = AdvancedMemoryManager()
    batcher = SmartBatcher(manager)
    
    # Create test documents
    documents = []
    for i in range(20):
        # Vary document sizes
        if i % 5 == 0:
            text = "Long abstract " * 500  # Long doc
        elif i % 3 == 0:
            text = "Medium abstract " * 200  # Medium doc
        else:
            text = "Short abstract " * 50  # Short doc
            
        doc = DocumentInfo(
            file_path=f"test_{i}.json",
            arxiv_id=f"test.{i}",
            abstract=text,
            char_count=len(text),
            estimated_tokens=batcher.estimate_tokens(text),
            priority=1.0 if i % 2 == 0 else 0.5
        )
        documents.append(doc)
    
    # Create batches
    max_gpu_memory = 48.0  # A6000 has 48GB
    batches = batcher.create_optimal_batches(documents, max_gpu_memory)
    
    print(f"Created {len(batches)} batches from {len(documents)} documents")
    for i, batch in enumerate(batches[:5]):  # Show first 5
        print(f"  Batch {i}: {len(batch.documents)} docs, "
              f"{batch.total_tokens} tokens, "
              f"{batch.estimated_memory_mb:.1f} MB, "
              f"priority={batch.priority:.2f}")


def test_checkpoint_system():
    """Test enhanced checkpoint with validation."""
    print("\n3. Testing Enhanced Checkpoint")
    print("-" * 40)
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = EnhancedCheckpoint(Path(tmpdir))
        
        # Add some data
        for i in range(10):
            checkpoint.data['processed_files'].add(f"file_{i}.json")
            if i % 3 == 0:
                checkpoint.mark_failed(f"failed_{i}.json", "Test error")
        
        # Record performance
        checkpoint.record_performance(0, 100, 2.5)
        checkpoint.record_performance(1, 120, 2.8)
        
        # Save
        checkpoint.save()
        print(f"Saved checkpoint with {len(checkpoint.data['processed_files'])} processed files")
        
        # Test loading
        checkpoint2 = EnhancedCheckpoint(Path(tmpdir))
        print(f"Loaded checkpoint with {len(checkpoint2.data['processed_files'])} processed files")
        print(f"Checksum match: {checkpoint.data['metadata']['checksum'] == checkpoint2.data['metadata']['checksum']}")
        
        # Test should_process method
        test_file = "new_file.json"
        print(f"\nShould process {test_file}: {checkpoint2.should_process(test_file)}")
        
        # Add file to processed and test again
        checkpoint2.data['processed_files'].add(test_file)
        checkpoint2.save()
        
        # Reload and verify
        checkpoint3 = EnhancedCheckpoint(Path(tmpdir))
        print(f"Should process {test_file} after processing: {checkpoint3.should_process(test_file)}")
        
        # Test with failed file
        failed_file = "failed_0.json"
        print(f"Should process {failed_file} (failed file): {checkpoint3.should_process(failed_file)}")


def test_predictive_load_balancer():
    """Test predictive load balancing."""
    print("\n4. Testing Predictive Load Balancer")
    print("-" * 40)
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = EnhancedCheckpoint(Path(tmpdir))
        balancer = PredictiveLoadBalancer(checkpoint)
        
        # Simulate some performance history
        for i in range(20):
            # GPU 0 is faster
            balancer.update_performance(0, 100, 2.0)  # 50 docs/sec
            # GPU 1 is slower
            balancer.update_performance(1, 100, 2.5)  # 40 docs/sec
        
        # Create test batch
        docs = [DocumentInfo(
            file_path=f"test_{i}.json",
            arxiv_id=f"test.{i}",
            abstract="Test abstract",
            char_count=100,
            estimated_tokens=50
        ) for i in range(10)]
        
        batch = BatchInfo(
            documents=docs,
            total_tokens=500,
            estimated_memory_mb=100,
            priority=1.0
        )
        
        # Test GPU selection
        selections = []
        for _ in range(10):
            gpu = balancer.select_gpu(batch)
            selections.append(gpu)
        
        print(f"GPU selections (10 batches): {selections}")
        print(f"GPU 0 selected: {selections.count(0)} times")
        print(f"GPU 1 selected: {selections.count(1)} times")
        print("(GPU 0 should be selected more often as it's faster)")


def main():
    """Run all tests."""
    print("PRODUCTION PIPELINE FEATURE TESTS")
    print("=" * 60)
    
    tests = [
        test_memory_estimation,
        test_smart_batching,
        test_checkpoint_system,
        test_predictive_load_balancer
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"\nTest failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("All tests completed!")


if __name__ == "__main__":
    main()