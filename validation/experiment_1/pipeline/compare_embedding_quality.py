#!/usr/bin/env python3
"""
Compare embedding quality between abstract-only and full-content embeddings
"""

import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from glob import glob

def analyze_embeddings():
    """Analyze the difference between abstract-only and full-content embeddings"""
    
    papers_dir = os.environ.get('VALIDATION_OUTPUT_DIR', './data/papers_with_embeddings')
    json_files = glob(os.path.join(papers_dir, "*.json"))
    
    print("Analyzing embedding quality...")
    print("=" * 60)
    
    abstract_only = []
    full_content = []
    
    for json_file in json_files[:100]:  # Sample 100 papers
        try:
            with open(json_file, 'r') as f:
                paper = json.load(f)
            
            if 'dimensions' in paper and 'WHAT' in paper['dimensions']:
                dim_data = paper['dimensions']['WHAT']
                
                if dim_data.get('has_full_content', False):
                    full_content.append({
                        'title': paper.get('title', 'Unknown'),
                        'context_length': dim_data.get('context_length', 0),
                        'embedding': np.array(dim_data['embeddings'])
                    })
                else:
                    abstract_only.append({
                        'title': paper.get('title', 'Unknown'),
                        'context_length': dim_data.get('context_length', 0),
                        'embedding': np.array(dim_data['embeddings'])
                    })
        except:
            continue
    
    print(f"\nFound {len(abstract_only)} abstract-only papers")
    print(f"Found {len(full_content)} full-content papers")
    
    if abstract_only:
        avg_abstract_len = np.mean([p['context_length'] for p in abstract_only])
        print(f"\nAbstract-only average length: {avg_abstract_len:.0f} characters")
    
    if full_content:
        avg_full_len = np.mean([p['context_length'] for p in full_content])
        print(f"Full-content average length: {avg_full_len:.0f} characters")
        print(f"Content increase: {avg_full_len/avg_abstract_len:.1f}x")
    
    # Compare similarity distributions
    if len(full_content) >= 2:
        print("\nAnalyzing similarity distributions...")
        
        # Sample similarities within full-content papers
        full_embeddings = np.array([p['embedding'] for p in full_content[:50]])
        full_similarities = cosine_similarity(full_embeddings)
        
        # Get upper triangle (excluding diagonal)
        upper_triangle = np.triu_indices_from(full_similarities, k=1)
        full_sim_values = full_similarities[upper_triangle]
        
        print(f"\nFull-content similarity statistics:")
        print(f"  Mean: {np.mean(full_sim_values):.3f}")
        print(f"  Std: {np.std(full_sim_values):.3f}")
        print(f"  Min: {np.min(full_sim_values):.3f}")
        print(f"  Max: {np.max(full_sim_values):.3f}")
        print(f"  Papers > 0.9 similarity: {(full_sim_values > 0.9).sum()}")
        print(f"  Papers > 0.8 similarity: {(full_sim_values > 0.8).sum()}")
        print(f"  Papers > 0.7 similarity: {(full_sim_values > 0.7).sum()}")

if __name__ == "__main__":
    analyze_embeddings()