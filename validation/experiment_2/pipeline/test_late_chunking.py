#!/usr/bin/env python3
"""
Test Jina V4's late chunking on a single paper to verify the output format.
Run this first to understand what late chunking returns.
"""

import os
import json
import torch
from transformers import AutoModel
from glob import glob

def test_late_chunking():
    """Test late chunking on one paper to see the output format."""
    
    # Load model
    print("Loading Jina V4 model...")
    model = AutoModel.from_pretrained(
        'jinaai/jina-embeddings-v4',
        trust_remote_code=True
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    
    # Get a test paper
    papers_dir = "/home/todd/olympus/Erebus/unstructured/papers"
    json_files = glob(os.path.join(papers_dir, "*.json"))
    
    # Find a paper with PDF content
    test_paper = None
    for json_file in json_files[:10]:
        with open(json_file, 'r') as f:
            paper = json.load(f)
        if 'pdf_content' in paper:
            test_paper = paper
            test_file = json_file
            break
    
    if not test_paper:
        print("No paper with PDF content found!")
        return
    
    print(f"\nTesting with: {test_paper.get('title', 'Unknown')[:60]}...")
    
    # Build text
    if 'pdf_content' in test_paper and 'markdown' in test_paper['pdf_content']:
        markdown = test_paper['pdf_content']['markdown']
        full_text = f"Title: {test_paper.get('title', '')}\n\n{markdown[:10000]}"  # Limit for test
    else:
        full_text = f"Title: {test_paper.get('title', '')}\n\nAbstract: {test_paper.get('abstract', '')}"
    
    print(f"Text length: {len(full_text)} characters")
    
    # Test different approaches to find the right method
    print("\n" + "="*60)
    print("Testing late chunking approaches...")
    print("="*60)
    
    # Approach 1: Check if model has late chunking method
    if hasattr(model, 'encode_late_chunked'):
        print("\nApproach 1: model.encode_late_chunked()")
        try:
            results = model.encode_late_chunked(
                texts=[full_text],
                task="retrieval",
                prompt_name="passage"
            )
            print(f"Success! Result type: {type(results)}")
            print(f"Result shape/content: {results}")
        except Exception as e:
            print(f"Failed: {e}")
    
    # Approach 2: Check model config for late chunking parameters
    print("\nModel config attributes:")
    config_attrs = [attr for attr in dir(model.config) if 'chunk' in attr.lower()]
    for attr in config_attrs:
        print(f"  {attr}: {getattr(model.config, attr, 'N/A')}")
    
    # Approach 3: Try encode_text with different parameters
    print("\nApproach 3: encode_text with late_chunking parameter")
    try:
        # Check what parameters encode_text accepts
        import inspect
        sig = inspect.signature(model.encode_text)
        print(f"encode_text parameters: {list(sig.parameters.keys())}")
    except:
        pass
    
    # Approach 4: Look for chunk-related methods
    print("\nChunk-related methods in model:")
    chunk_methods = [method for method in dir(model) if 'chunk' in method.lower()]
    for method in chunk_methods:
        print(f"  {method}")
    
    # Test standard encoding for comparison
    print("\n" + "="*60)
    print("Standard encoding (for comparison):")
    print("="*60)
    
    embeddings = model.encode_text(
        texts=[full_text[:2000]],  # Shorter for test
        task="retrieval",
        prompt_name="passage",
    )
    
    print(f"Standard embedding shape: {embeddings[0].shape}")
    print(f"Embedding dimensions: {len(embeddings[0])}")
    
    # Check if there's documentation in the model
    if hasattr(model, '__doc__') and model.__doc__:
        print("\nModel documentation:")
        print(model.__doc__[:500])


if __name__ == "__main__":
    test_late_chunking()