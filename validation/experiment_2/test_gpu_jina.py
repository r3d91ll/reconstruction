#!/usr/bin/env python3
"""Quick test to verify GPU and Jina V4 are working"""

import torch
from transformers import AutoModel

print("=" * 60)
print("GPU AND JINA V4 TEST")
print("=" * 60)

# Check GPU
print(f"\nGPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU 0 name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("No GPU available!")

# Test Jina loading
print("\nLoading Jina V4 model...")
try:
    model = AutoModel.from_pretrained(
        'jinaai/jina-embeddings-v4',
        trust_remote_code=True
    )
    print("✓ Model loaded successfully")
    print(f"Model config: {model.config.hidden_size} dimensions")
    
    # Test encoding
    print("\nTesting encoding...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    test_text = "This is a test sentence for Jina V4 embeddings."
    with torch.no_grad():
        embeddings = model.encode_text(
            texts=[test_text],
            task="retrieval",
            prompt_name="passage"
        )
    
    print(f"✓ Embedding shape: {embeddings.shape}")
    print(f"✓ Embedding dim: {embeddings.shape[1]}")
    
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)