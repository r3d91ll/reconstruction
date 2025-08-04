#!/usr/bin/env python3
"""Test flash attention performance"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import time
import torch
from transformers import AutoTokenizer, AutoModel

# Load model
print("Loading Jina model...")
device = torch.device("cuda:0")
tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
model = AutoModel.from_pretrained(
    "jinaai/jina-embeddings-v3",
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device)
model.eval()

# Test texts
test_texts = ["This is a test abstract about machine learning."] * 100

# Warm up
print("Warming up...")
with torch.no_grad():
    for _ in range(3):
        encoded = tokenizer(test_texts[:10], padding=True, truncation=True, 
                          max_length=512, return_tensors='pt').to(device)
        outputs = model(**encoded)
        _ = outputs.last_hidden_state.mean(dim=1)

# Benchmark
print("\nBenchmarking with 100 texts...")
torch.cuda.synchronize()
start = time.time()

with torch.no_grad():
    encoded = tokenizer(test_texts, padding=True, truncation=True, 
                      max_length=512, return_tensors='pt').to(device)
    outputs = model(**encoded)
    embeddings = outputs.last_hidden_state.mean(dim=1)

torch.cuda.synchronize()
end = time.time()

print(f"Time: {end - start:.3f} seconds")
print(f"Throughput: {100 / (end - start):.1f} texts/second")

# Check if flash attention was used
print("\nChecking model internals...")
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    # Run again to catch warnings
    outputs = model(**encoded)
    if any("flash_attn" in str(warning.message).lower() for warning in w):
        print("Flash attention warnings detected")
    else:
        print("Model likely using flash attention (no warnings)")