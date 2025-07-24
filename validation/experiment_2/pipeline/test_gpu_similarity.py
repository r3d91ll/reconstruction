#!/usr/bin/env python3
"""
Test script to verify GPU acceleration for similarity computation
"""

import torch
import numpy as np
import time
import sys
import os

def test_gpu_similarity():
    # Check GPU availability
    if not torch.cuda.is_available():
        print("CUDA is not available!")
        sys.exit(1)
    
    gpu_count = torch.cuda.device_count()
    print(f"Found {gpu_count} GPUs:")
    
    for i in range(gpu_count):
        name = torch.cuda.get_device_name(i)
        memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {name} ({memory:.1f} GB)")
    
    # Test different batch sizes
    embedding_dim = 1024  # Jina v4 dimension
    batch_sizes = [100, 1000, 5000, 10000]
    
    print("\nBenchmarking CPU vs GPU similarity computation:")
    print("-" * 60)
    
    for n in batch_sizes:
        print(f"\nBatch size: {n} x {n}")
        
        # Generate random embeddings
        embeddings = np.random.randn(n, embedding_dim).astype(np.float32)
        
        # CPU computation
        start_time = time.time()
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_norm_cpu = embeddings / norms
        # Compute similarity
        similarity_cpu = np.dot(embeddings_norm_cpu, embeddings_norm_cpu.T)
        cpu_time = time.time() - start_time
        print(f"  CPU time: {cpu_time:.3f} seconds")
        
        # GPU computation (single GPU)
        torch.cuda.synchronize()
        start_time = time.time()
        embeddings_gpu = torch.from_numpy(embeddings).cuda()
        embeddings_norm_gpu = torch.nn.functional.normalize(embeddings_gpu, p=2, dim=1)
        similarity_gpu = torch.mm(embeddings_norm_gpu, embeddings_norm_gpu.t())
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        print(f"  GPU time: {gpu_time:.3f} seconds")
        print(f"  Speedup: {cpu_time/gpu_time:.1f}x")
        
        # Check if multi-GPU would help
        if gpu_count > 1 and n >= 5000:
            print(f"  Multi-GPU available for large batches (using {gpu_count} GPUs)")
        
        # Memory usage
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"  GPU memory used: {allocated:.2f} GB")
        
        # Verify results match
        similarity_gpu_cpu = similarity_gpu.cpu().numpy()
        max_diff = np.max(np.abs(similarity_cpu - similarity_gpu_cpu))
        print(f"  Max difference (CPU vs GPU): {max_diff:.6f}")
        
        # Clear GPU memory
        del embeddings_gpu, embeddings_norm_gpu, similarity_gpu
        torch.cuda.empty_cache()
    
    # Test memory limits
    print("\nTesting GPU memory limits:")
    max_n = 20000
    try:
        embeddings_large = torch.randn(max_n, embedding_dim, dtype=torch.float16).cuda()
        embeddings_norm_large = torch.nn.functional.normalize(embeddings_large, p=2, dim=1)
        print(f"  Can handle {max_n} embeddings in float16")
        
        # Estimate max batch size for 48GB GPU
        memory_per_embedding = embedding_dim * 2  # float16
        similarity_matrix_memory = max_n * max_n * 2  # float16
        total_memory = (max_n * memory_per_embedding * 2 + similarity_matrix_memory) / 1024**3
        print(f"  Memory used for {max_n} embeddings: {total_memory:.2f} GB")
        
        # Get GPU memory utilization from environment variable with default fallback
        gpu_memory_utilization = float(os.environ.get('GPU_MEMORY_UTILIZATION', '0.7'))
        available_memory = 48 * gpu_memory_utilization  # Percentage of 48GB
        max_batch_estimate = int(np.sqrt(available_memory * 1024**3 / (embedding_dim * 4 + 2)))
        print(f"  Estimated max batch size for 48GB GPU: ~{max_batch_estimate}")
        
    except RuntimeError as e:
        print(f"  GPU out of memory at {max_n} embeddings")
    finally:
        torch.cuda.empty_cache()
    
    return 0

if __name__ == "__main__":
    sys.exit(test_gpu_similarity())