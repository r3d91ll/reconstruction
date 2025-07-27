#!/usr/bin/env python3
"""Test Jina v4 late chunking implementation"""

import torch
from transformers import AutoTokenizer, AutoModel

print("Testing Jina v4 late chunking...")
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

model.eval()

# Test document for late chunking
text = """This is a longer test document that we will use for late chunking. 
Late chunking means we encode the entire document at once to get token embeddings,
then extract chunk embeddings by mean pooling over sliding windows of tokens.
This approach preserves context across chunk boundaries."""

inputs = tokenizer(
    text,
    return_tensors='pt',
    max_length=512,
    truncation=True,
    return_offsets_mapping=True,
    padding=True
).to(device)

with torch.no_grad():
    try:
        # Get outputs with retrieval task
        outputs = model(
            **{k: v for k, v in inputs.items() if k != 'offset_mapping'},
            task_label='retrieval'
        )
        
        print(f"Output attributes: {[attr for attr in dir(outputs) if not attr.startswith('_')]}")
        print(f"single_vec_emb shape: {outputs.single_vec_emb.shape}")
        print(f"multi_vec_emb shape: {outputs.multi_vec_emb.shape}")
        
        # For late chunking, we use multi_vec_emb
        all_token_embeddings = outputs.multi_vec_emb[0]
        print(f"Token embeddings shape: {all_token_embeddings.shape}")
        
        # Extract chunks with sliding window
        chunk_size = 10  # tokens
        stride = 5       # tokens
        seq_len = all_token_embeddings.shape[0]
        
        chunks = []
        
        # Handle sequences shorter than chunk size
        if seq_len < chunk_size:
            # Create a single chunk from the entire sequence
            chunk_embedding = all_token_embeddings.mean(dim=0)
            chunks.append(chunk_embedding)
            print(f"\nSequence shorter than chunk size, created single chunk")
        else:
            # Use sliding window for longer sequences
            for start_idx in range(0, seq_len - chunk_size + 1, stride):
                end_idx = start_idx + chunk_size
                
                # Mean pool tokens in this chunk
                chunk_embedding = all_token_embeddings[start_idx:end_idx].mean(dim=0)
                chunks.append(chunk_embedding)
        
        print(f"\nCreated {len(chunks)} chunks")
        if chunks:
            print(f"Each chunk embedding shape: {chunks[0].shape}")
            
    except Exception as e:
        print(f"Error during model inference: {e}")
        import traceback
        traceback.print_exc()