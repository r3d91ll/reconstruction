#!/usr/bin/env python3
"""
Export ArangoDB graph data for Wolfram validation
Extracts real dimensional data and context scores for mathematical analysis
"""

import os
import json
import numpy as np
from arango import ArangoClient

def main():
    # Configuration
    db_name = "information_reconstructionism"
    output_dir = "/home/todd/reconstructionism/validation/wolfram/data"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Connect to ArangoDB
    arango_host = os.environ.get('ARANGO_HOST', 'http://192.168.1.69:8529')
    client = ArangoClient(hosts=arango_host)
    username = os.environ.get('ARANGO_USERNAME', 'root')
    password = os.environ.get('ARANGO_PASSWORD', '')
    db = client.db(db_name, username=username, password=password)
    
    print("Extracting data from ArangoDB...")
    
    # 1. Extract papers with embeddings
    papers_collection = db.collection('papers')
    papers = list(papers_collection.all())
    print(f"Found {len(papers)} papers")
    
    # 2. Extract semantic similarity edges
    similarity_collection = db.collection('semantic_similarity')
    edges = list(similarity_collection.all())
    print(f"Found {len(edges)} similarity edges")
    
    # 3. Prepare data for Wolfram validation
    
    # A. Zero propagation test data
    # Find papers with missing dimensions
    zero_propagation_data = []
    for paper in papers:
        dims = paper.get('dimensions', {})
        where = 1 if paper.get('arxiv_id') else 0
        what = 1 if dims.get('WHAT', {}).get('embeddings') else 0
        # For now, simulate CONVEYANCE and TIME
        conveyance = np.random.uniform(0.3, 0.9) if what else 0
        time = 1  # Present
        
        zero_propagation_data.append({
            'id': paper['_key'],
            'title': paper.get('title', '')[:50],
            'WHERE': where,
            'WHAT': what,
            'CONVEYANCE': conveyance,
            'TIME': time,
            'INFORMATION': where * what * conveyance * time
        })
    
    # B. Context amplification data
    context_scores = []
    amplified_scores = []
    for edge in edges:
        original = edge.get('context_original', edge.get('context', 0))
        amplified = edge.get('context', 0)
        if original > 0:
            context_scores.append(original)
            amplified_scores.append(amplified)
    
    # C. Theory-practice bridge candidates
    # Papers with high conveyance (simulated for now)
    bridge_candidates = []
    for i, paper in enumerate(papers[:20]):  # Top 20 for analysis
        bridge_candidates.append({
            'id': paper['_key'],
            'title': paper.get('title', '')[:50],
            'abstract_length': len(paper.get('abstract', '')),
            'category_count': len(paper.get('categories', [])),
            'embedding_norm': np.linalg.norm(
                paper.get('dimensions', {}).get('WHAT', {}).get('embeddings', [0])[:100]
            )
        })
    
    # 4. Export for Wolfram
    wolfram_data = {
        'metadata': {
            'paper_count': len(papers),
            'edge_count': len(edges),
            'avg_context': np.mean(context_scores) if context_scores else 0,
            'std_context': np.std(context_scores) if context_scores else 0
        },
        'zero_propagation': zero_propagation_data[:10],  # Sample
        'context_distribution': {
            'original': context_scores[:100],  # First 100
            'amplified': amplified_scores[:100]
        },
        'bridge_candidates': bridge_candidates
    }
    
    # Save as JSON
    with open(os.path.join(output_dir, 'graph_data.json'), 'w') as f:
        json.dump(wolfram_data, f, indent=2)
    
    # Save as Wolfram-friendly format
    with open(os.path.join(output_dir, 'graph_data.wl'), 'w') as f:
        f.write("(* Graph data from ArangoDB *)\n\n")
        
        f.write("graphMetadata = <|\n")
        f.write(f'  "paperCount" -> {len(papers)},\n')
        f.write(f'  "edgeCount" -> {len(edges)},\n')
        f.write(f'  "avgContext" -> {np.mean(context_scores) if context_scores else 0:.3f},\n')
        f.write(f'  "stdContext" -> {np.std(context_scores) if context_scores else 0:.3f}\n')
        f.write("|>;\n\n")
        
        f.write("contextScores = {")
        f.write(", ".join(f"{s:.3f}" for s in context_scores[:50]))
        f.write("};\n\n")
        
        f.write("amplifiedScores = {")
        f.write(", ".join(f"{s:.3f}" for s in amplified_scores[:50]))
        f.write("};\n")
    
    print(f"\nExported data to {output_dir}/")
    print("Files created:")
    print("  - graph_data.json (structured data)")
    print("  - graph_data.wl (Wolfram format)")
    
    # Summary statistics
    print("\nSummary:")
    print(f"  Papers with embeddings: {sum(1 for p in papers if p.get('dimensions', {}).get('WHAT', {}).get('embeddings'))}")
    print(f"  Context scores: min={min(context_scores):.3f}, max={max(context_scores):.3f}" if context_scores else "  No context scores")
    print(f"  Amplification effect: {np.mean(amplified_scores)/np.mean(context_scores):.3f}x" if context_scores else "")

if __name__ == "__main__":
    main()