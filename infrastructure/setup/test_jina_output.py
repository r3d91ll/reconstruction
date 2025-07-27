#!/usr/bin/env python3
"""Test to understand Jina v4 output structure"""

import torch
from transformers import AutoTokenizer, AutoModel

print("Testing Jina v4 output structure...")
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

texts = ["This is a test document for understanding the output"]
inputs = tokenizer(texts, return_tensors='pt', padding=True).to(device)

with torch.no_grad():
    # Use retrieval task
    outputs = model(**inputs, task_label="retrieval")
    
    print(f"Output type: {type(outputs)}")
    print(f"Output attributes: {dir(outputs)}")
    
    # Check common attributes
    if hasattr(outputs, 'embeddings'):
        print(f"embeddings shape: {outputs.embeddings.shape}")
    if hasattr(outputs, 'last_hidden_state'):
        print(f"last_hidden_state shape: {outputs.last_hidden_state.shape}")
    if hasattr(outputs, 'hidden_states'):
        print(f"hidden_states length: {len(outputs.hidden_states)}")
    if hasattr(outputs, 'attentions'):
        print(f"attentions available: {outputs.attentions is not None}")
        
    # Try to get token embeddings
    print("\nTrying to get token embeddings...")
    
    # Check if model has encoder
    try:
        if hasattr(model, 'encoder'):
            print("Model has encoder attribute")
            encoder_outputs = model.encoder(**inputs)
            print(f"Encoder output type: {type(encoder_outputs)}")
            if hasattr(encoder_outputs, 'last_hidden_state'):
                print(f"Encoder last_hidden_state shape: {encoder_outputs.last_hidden_state.shape}")
    except Exception as e:
        print(f"Error accessing model encoder: {e}")
        import traceback
        traceback.print_exc()