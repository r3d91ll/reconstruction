#!/usr/bin/env python3
"""Test to understand Jina v4 embeddings for late chunking"""

import torch
from transformers import AutoTokenizer, AutoModel

print("Testing Jina v4 embeddings structure...")
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

texts = ["This is a test document for understanding late chunking with Jina v4"]
inputs = tokenizer(texts, return_tensors='pt', padding=True).to(device)

with torch.no_grad():
    # Get outputs with retrieval task
    outputs = model(**inputs, task_label="retrieval")
    
    print(f"single_vec_emb shape: {outputs.single_vec_emb.shape}")
    print(f"multi_vec_emb shape: {outputs.multi_vec_emb.shape if outputs.multi_vec_emb is not None else 'None'}")
    print(f"vlm_last_hidden_states: {outputs.vlm_last_hidden_states}")
    
    # For late chunking, we need token-level embeddings
    # Let's see if we can get them with return_multivector
    print("\nTrying with return_multivector=True...")
    
    # Check if model has forward with different params
    outputs_multi = model(**inputs, task_label="retrieval", return_multivector=True)
    print(f"With multivector - single_vec_emb shape: {outputs_multi.single_vec_emb.shape}")
    print(f"With multivector - multi_vec_emb shape: {outputs_multi.multi_vec_emb.shape if outputs_multi.multi_vec_emb is not None else 'None'}")
    
    # Check model architecture
    print("\nModel architecture check:")
    print(f"Model type: {type(model)}")
    if hasattr(model, 'base_model'):
        print(f"Base model type: {type(model.base_model)}")
    
    # Try to access encoder directly
    if hasattr(model, 'encoder'):
        print("Has encoder")
    if hasattr(model, 'embeddings'):
        print("Has embeddings layer")