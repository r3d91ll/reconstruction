#!/usr/bin/env python3
"""
STEP 1: Generate embeddings for papers
Simple, no fancy features, just prove we can embed
"""

import json
import os
import glob
from sentence_transformers import SentenceTransformer
import numpy as np

def main():
    # Configuration
    papers_dir = "/home/todd/olympus/Erebus/unstructured/papers"
    output_dir = "/home/todd/reconstructionism/validation/data/papers_with_embeddings"
    limit = 100  # Start small
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model (SPECTER2 for scientific papers)
    print("Loading embedding model...")
    model = SentenceTransformer('allenai/specter2')
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    
    # Get papers
    json_files = glob.glob(os.path.join(papers_dir, "*.json"))[:limit]
    print(f"\nProcessing {len(json_files)} papers...")
    
    success = 0
    for i, json_file in enumerate(json_files):
        try:
            # Load paper
            with open(json_file, 'r') as f:
                paper = json.load(f)
            
            # Create text to embed
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')
            text = f"{title} {abstract}"
            
            # Generate embedding
            embedding = model.encode(text)
            
            # Add to paper data
            if 'dimensions' not in paper:
                paper['dimensions'] = {}
            if 'WHAT' not in paper['dimensions']:
                paper['dimensions']['WHAT'] = {}
                
            paper['dimensions']['WHAT']['embeddings'] = embedding.tolist()
            paper['dimensions']['WHAT']['embedding_dim'] = len(embedding)
            paper['dimensions']['WHAT']['embedding_method'] = 'specter2'
            
            # Save
            output_file = os.path.join(output_dir, os.path.basename(json_file))
            with open(output_file, 'w') as f:
                json.dump(paper, f)
                
            success += 1
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(json_files)}")
                
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"\n✓ STEP 1 COMPLETE")
    print(f"Embedded {success}/{len(json_files)} papers")
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

if __name__ == "__main__":
    main()