#!/usr/bin/env python3
"""
Verify pipeline results quickly
"""

import os
import json
import glob
from arango import ArangoClient

def verify_embeddings(data_dir):
    """Check that embeddings were generated"""
    print("\n1. EMBEDDING VERIFICATION")
    print("-" * 40)
    
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    print(f"JSON files found: {len(json_files)}")
    
    if json_files:
        # Check first file
        with open(json_files[0], 'r') as f:
            paper = json.load(f)
        
        has_embeddings = (
            'dimensions' in paper and
            'WHAT' in paper['dimensions'] and
            'embeddings' in paper['dimensions']['WHAT']
        )
        
        if has_embeddings:
            emb_len = len(paper['dimensions']['WHAT']['embeddings'])
            print(f"✓ Embeddings present: {emb_len} dimensions")
            print(f"✓ Method: {paper['dimensions']['WHAT'].get('embedding_method', 'unknown')}")
        else:
            print("✗ No embeddings found!")
    
    return len(json_files)

def verify_arangodb():
    """Check ArangoDB state"""
    print("\n2. ARANGODB VERIFICATION")
    print("-" * 40)
    
    try:
        # Connect
        client = ArangoClient(hosts='http://localhost:8529')
        username = os.environ.get('ARANGO_USERNAME', 'root')
        password = os.environ.get('ARANGO_PASSWORD', '')
        db = client.db('information_reconstructionism', username=username, password=password)
        
        # Check papers
        papers_count = db.collection('papers').count()
        print(f"Papers in database: {papers_count}")
        
        # Check edges
        if db.has_collection('semantic_similarity'):
            edges_count = db.collection('semantic_similarity').count()
            print(f"Semantic similarity edges: {edges_count}")
            
            # Sample edge
            cursor = db.aql.execute("""
            FOR e IN semantic_similarity
                LIMIT 1
                RETURN {
                    context: e.context,
                    context_amplified: e.context_amplified,
                    has_alpha: HAS(e, "alpha_used")
                }
            """)
            
            sample = list(cursor)
            if sample:
                edge = sample[0]
                print(f"✓ Sample edge context: {edge['context']:.3f}")
                print(f"✓ Amplified (α=1.5): {edge['context_amplified']:.3f}")
                print(f"✓ Amplification applied: {edge['has_alpha']}")
        
        # Check for high-similarity pairs
        cursor = db.aql.execute("""
        FOR e IN semantic_similarity
            FILTER e.context_amplified > 0.8
            COLLECT WITH COUNT INTO high_sim_count
            RETURN high_sim_count
        """)
        
        high_sim = list(cursor)[0]
        print(f"\nHigh similarity pairs (>0.8): {high_sim}")
        
        return papers_count, edges_count
        
    except Exception as e:
        print(f"✗ ArangoDB error: {e}")
        return 0, 0

def quick_stats(data_dir):
    """Show quick statistics"""
    print("\n3. QUICK STATISTICS")
    print("-" * 40)
    
    # Find latest run
    if not data_dir:
        latest = "/home/todd/reconstructionism/validation/results/latest"
        if os.path.exists(latest):
            data_dir = os.path.join(latest, "data")
        else:
            print("No latest run found")
            return
    
    # Count files
    num_embeddings = verify_embeddings(data_dir)
    
    # Check DB
    num_papers, num_edges = verify_arangodb()
    
    # Summary
    print("\n" + "="*40)
    print("SUMMARY")
    print("="*40)
    print(f"Embedded files: {num_embeddings}")
    print(f"Papers in DB: {num_papers}")
    print(f"Semantic edges: {num_edges}")
    print(f"Avg edges per paper: {num_edges/num_papers:.1f}" if num_papers > 0 else "N/A")
    
    if num_embeddings == num_papers and num_edges > 0:
        print("\n✓ Pipeline appears successful!")
    else:
        print("\n⚠ Check pipeline - counts don't match")

def main():
    """Run verification"""
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = None
    
    quick_stats(data_dir)

if __name__ == "__main__":
    main()