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
    papers_dir = "/home/todd/olympus/Erebus/unstructured/papers"  # Source - DO NOT MODIFY
    output_dir = os.environ.get("VALIDATION_OUTPUT_DIR", "/home/todd/reconstructionism/validation/experiment_1/data/papers_with_embeddings")
    limit = 4000  # Pipeline run - 4000 papers
    
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
    
    # Get papers and sort by year (oldest first)
    all_json_files = glob.glob(os.path.join(papers_dir, "*.json"))
    
    # Load papers with years for sorting
    papers_with_years = []
    for json_file in all_json_files:
        try:
            with open(json_file, 'r') as f:
                paper = json.load(f)
            year = paper.get('year', 9999)  # Default to 9999 if no year
            papers_with_years.append((year, json_file))
        except:
            continue
    
    # Sort by year (oldest first)
    papers_with_years.sort(key=lambda x: x[0])
    
    # Take the first 'limit' papers
    json_files = [p[1] for p in papers_with_years[:limit]]
    
    print(f"\nProcessing {len(json_files)} papers (sorted by year, oldest first)...")
    if json_files:
        first_year = papers_with_years[0][0]
        last_year = papers_with_years[min(limit-1, len(papers_with_years)-1)][0]
        print(f"Year range: {first_year} - {last_year}")
    
    success = 0
    full_content_count = 0
    for i, json_file in enumerate(json_files):
        try:
            # Load paper
            with open(json_file, 'r') as f:
                paper = json.load(f)
            
            # Create text for embedding - using more context
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')
            categories = ', '.join(paper.get('categories', []))
            
            # Check if we have full PDF content
            has_full_content = False
            if 'pdf_content' in paper and 'markdown' in paper['pdf_content']:
                # Use full content if available
                markdown = paper['pdf_content']['markdown']
                text = f"Title: {title}\n\nCategories: {categories}\n\nAbstract: {abstract}\n\n# Full Paper Content\n\n{markdown}"
                has_full_content = True
                full_content_count += 1
                
                # Truncate if needed (128K token limit ~ 512K chars)
                max_chars = 512000
                if len(text) > max_chars:
                    text = text[:max_chars] + "\n\n[Content truncated...]"
                    print(f"Truncated content from {len(text)} to {max_chars} chars")
            else:
                # Fall back to abstract only
                text = f"Title: {title}\n\nCategories: {categories}\n\nAbstract: {abstract}"
            
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
    print(f"   - With full PDF content: {full_content_count}")
    print(f"   - Abstract only: {success - full_content_count}")
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