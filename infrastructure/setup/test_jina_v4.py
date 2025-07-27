#!/usr/bin/env python3
"""Test script to verify Jina v4 usage"""

import torch
from transformers import AutoTokenizer, AutoModel

# Test 1: Basic transformers usage
print("Testing Jina v4 with transformers...")
model_name = "jinaai/jina-embeddings-v4"

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

# Move model to appropriate device
if torch.cuda.is_available():
    model = model.cuda()
else:
    model = model.to(device)

# Test getting raw token embeddings for late chunking
texts = ["This is a test document for late chunking"]
inputs = tokenizer(texts, return_tensors='pt', padding=True).to(device)

with torch.no_grad():
    # Method 1: Direct forward pass for token embeddings
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    print(f"Token embeddings shape: {embeddings.shape}")
    
    # Mean pooling
    attention_mask = inputs['attention_mask']
    masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
    summed = masked_embeddings.sum(dim=1)
    counts = attention_mask.sum(dim=1, keepdim=True)
    mean_pooled = summed / counts
    print(f"Mean pooled shape: {mean_pooled.shape}")

print("\nTest successful with transformers!")

# Test 2: Check encode_text method
print("\nTesting encode_text method...")
try:
    # Test using encode_text with task parameter
    query_embeddings = model.encode_text(
        texts=texts,
        task="retrieval",
        prompt_name="passage"
    )
    print(f"encode_text embedding shape: {query_embeddings.shape}")
except Exception as e:
    print(f"encode_text error: {e}")

# Test 3: Check if sentence-transformers is better
try:
    from sentence_transformers import SentenceTransformer
    print("\nTesting with sentence-transformers...")
    
    model_st = SentenceTransformer(model_name, trust_remote_code=True)
    embeddings_st = model_st.encode(texts)
    print(f"Sentence-transformers embedding shape: {embeddings_st.shape}")
    
except ImportError:
    print("\nsentence-transformers not installed")