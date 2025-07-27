#!/usr/bin/env python3
"""Test script to verify GPU availability"""

import os
import torch

print("Testing GPU availability...")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch CUDA device count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Memory: {props.total_memory / 1024**3:.2f} GB")
else:
    print("No CUDA devices available!")

# Test setting device
try:
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print("\nSuccessfully set device to GPU 0")
        
        # Test tensor creation
        x = torch.randn(10, 10).cuda()
        print(f"Created tensor on: {x.device}")
except Exception as e:
    print(f"\nError setting device: {e}")