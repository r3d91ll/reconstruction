#!/usr/bin/env python3
"""
Test if Jina model supports late chunking / multi-vector embeddings
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
from transformers import AutoTokenizer, AutoModel

def test_jina_capabilities():
    """Test what Jina model returns"""
    print("Testing Jina model capabilities...")
    
    model_name = "jinaai/jina-embeddings-v3"
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading model: {model_name}")
    print(f"Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device != 'cpu' else torch.float32
    ).to(device)
    
    model.eval()
    
    # Test text
    test_text = """
    This is a test document with multiple paragraphs.
    
    The first paragraph talks about testing the Jina embeddings model
    to see if it supports late chunking or multi-vector outputs.
    
    The second paragraph continues the discussion with more details
    about how late chunking works and why it's more efficient.
    
    The third paragraph concludes our test document.
    """
    
    print(f"\nTest text length: {len(test_text)} chars")
    
    # Tokenize
    inputs = tokenizer(
        test_text,
        return_tensors='pt',
        max_length=8192,
        truncation=True,
        padding=True
    ).to(device)
    
    print(f"Token count: {inputs['input_ids'].shape[1]}")
    
    # Get outputs
    with torch.no_grad():
        # Try different task types
        for task in ['retrieval.passage', 'retrieval', None]:
            print(f"\nTrying task: {task}")
            try:
                if task:
                    outputs = model(**inputs, task=task)
                else:
                    outputs = model(**inputs)
                    
                # Check what outputs are available
                print(f"Output attributes: {list(outputs.keys()) if hasattr(outputs, 'keys') else dir(outputs)}")
                
                if hasattr(outputs, 'last_hidden_state'):
                    print(f"last_hidden_state shape: {outputs.last_hidden_state.shape}")
                    
                if hasattr(outputs, 'pooler_output'):
                    print(f"pooler_output shape: {outputs.pooler_output.shape}")
                    
                if hasattr(outputs, 'multi_vec_emb'):
                    print(f"multi_vec_emb available! Shape: {outputs.multi_vec_emb.shape}")
                    
                # Check for embeddings
                if hasattr(outputs, 'embeddings'):
                    print(f"embeddings shape: {outputs.embeddings.shape}")
                    
            except Exception as e:
                print(f"Error with task {task}: {e}")
    
    print("\n" + "="*50)
    print("Summary:")
    print("- Check if 'multi_vec_emb' is available for late chunking")
    print("- Otherwise, we can use 'last_hidden_state' for token embeddings")
    print("- May need to use specific task parameter")
    print("="*50)

if __name__ == '__main__':
    test_jina_capabilities()