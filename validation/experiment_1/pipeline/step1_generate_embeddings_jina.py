#!/usr/bin/env python3
"""
STEP 1: Generate embeddings using Jina Embeddings V4
Using local model with late chunking support
"""

import json
import os
import glob
import torch
from transformers import AutoModel
import numpy as np

def main():
    # Configuration
    papers_dir = "/home/todd/olympus/Erebus/unstructured/papers"
    output_dir = os.environ.get("VALIDATION_OUTPUT_DIR", "/home/todd/reconstructionism/validation/data/papers_with_embeddings")
    limit = 100  # Start small for testing
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Jina Embeddings V4 from local cache
    print("Loading Jina Embeddings V4 from local cache...")
    model_name = "jinaai/jina-embeddings-v4"
    
    # Load model (tokenizer is built into the model's encode methods)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16  # Use float16 as shown in docs
    )
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    print(f"Model config: {model.config.hidden_size} dimensions")
    print(f"Max sequence length: {model.config.max_position_embeddings}")
    
    # Get papers
    json_files = glob.glob(os.path.join(papers_dir, "*.json"))[:limit]
    print(f"\nProcessing {len(json_files)} papers...")
    
    success = 0
    for i, json_file in enumerate(json_files):
        try:
            # Load paper
            with open(json_file, 'r') as f:
                paper = json.load(f)
            
            # Create text for embedding - using more context
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')
            categories = ', '.join(paper.get('categories', []))
            
            # Jina V4 can handle long context - feed it everything
            text = f"Title: {title}\n\nCategories: {categories}\n\nAbstract: {abstract}"
            
            # For late chunking, we could also add:
            # - Paper sections if available
            # - References
            # - Any extracted text from PDF
            
            # Generate embedding using Jina V4's encode_text method
            # For academic papers, we treat them as passages in retrieval task
            embeddings = model.encode_text(
                texts=[text],
                task="retrieval",
                prompt_name="passage",
                # Could add these if needed:
                # max_length=16384,  # 16K context window
                # truncate_dim=None,  # Use full dimensionality
            )
            
            # encode_text returns numpy array, get first (and only) embedding
            embedding = embeddings[0]
            
            # Jina V4 produces embeddings (default 2048 dimensions unless truncated)
            print(f"Generated embedding shape: {embedding.shape}")
            
            # Add to paper data
            if 'dimensions' not in paper:
                paper['dimensions'] = {}
            if 'WHAT' not in paper['dimensions']:
                paper['dimensions']['WHAT'] = {}
                
            paper['dimensions']['WHAT']['embeddings'] = embedding.tolist()
            paper['dimensions']['WHAT']['embedding_dim'] = len(embedding)
            paper['dimensions']['WHAT']['embedding_method'] = 'jina-v4'
            paper['dimensions']['WHAT']['context_length'] = len(text)
            
            # Save
            output_file = os.path.join(output_dir, os.path.basename(json_file))
            with open(output_file, 'w') as f:
                json.dump(paper, f)
                
            success += 1
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(json_files)}")
                
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    print(f"\n✓ STEP 1 COMPLETE")
    print(f"Embedded {success}/{len(json_files)} papers with Jina V4")
    print(f"Output: {output_dir}")
    
    # Verify one paper
    sample_files = glob.glob(os.path.join(output_dir, "*.json"))
    if sample_files:
        with open(sample_files[0], 'r') as f:
            sample = json.load(f)
        
        embedding_exists = (
            'dimensions' in sample and 
            'WHAT' in sample['dimensions'] and 
            'embeddings' in sample['dimensions']['WHAT'] and 
            sample['dimensions']['WHAT']['embeddings'] is not None
        )
        print(f"\nVerification - Sample has embeddings: {embedding_exists}")
        if embedding_exists:
            print(f"Embedding dimension: {len(sample['dimensions']['WHAT']['embeddings'])}")
            print(f"Embedding method: {sample['dimensions']['WHAT']['embedding_method']}")
            print(f"Context length used: {sample['dimensions']['WHAT']['context_length']} chars")

if __name__ == "__main__":
    main()