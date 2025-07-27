#!/usr/bin/env python3
"""Test to find valid task labels for Jina v4"""

import torch
from transformers import AutoTokenizer, AutoModel

print("Testing Jina v4 task labels...")
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

texts = ["This is a test document"]
inputs = tokenizer(texts, return_tensors='pt', padding=True).to(device)

# Test different task labels
task_labels = [
    "retrieval",
    "retrieval.query", 
    "retrieval.document",
    "text-matching",
    "code",
    "passage",
    "query"
]

with torch.no_grad():
    for task_label in task_labels:
        try:
            outputs = model(**inputs, task_label=task_label)
            # Check which output attribute exists in Jina v4
            if hasattr(outputs, 'single_vec_emb'):
                print(f"✓ Task label '{task_label}' works! Output shape: {outputs.single_vec_emb.shape}")
            elif hasattr(outputs, 'last_hidden_state'):
                print(f"✓ Task label '{task_label}' works! Output shape: {outputs.last_hidden_state.shape}")
            else:
                print(f"✓ Task label '{task_label}' works! Available attributes: {[attr for attr in dir(outputs) if not attr.startswith('_')]}")
        except Exception as e:
            print(f"✗ Task label '{task_label}' failed: {type(e).__name__}: {e}")

# Also test if model has any attribute that lists valid tasks
print("\nChecking model attributes...")
if hasattr(model, 'task_labels'):
    print(f"Available task labels: {model.task_labels}")
if hasattr(model, 'config'):
    if hasattr(model.config, 'task_labels'):
        print(f"Config task labels: {model.config.task_labels}")