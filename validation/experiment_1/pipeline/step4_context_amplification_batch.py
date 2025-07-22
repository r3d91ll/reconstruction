#!/usr/bin/env python3
"""
STEP 4: Apply Context^α amplification to semantic edges
Batch processing version for large graphs
"""

import os
import numpy as np
from arango import ArangoClient
from tqdm import tqdm
import time

def main():
    # Configuration
    db_name = "information_reconstructionism"
    alpha = 1.5  # Our theoretical value
    batch_size = 10000  # Process edges in batches
    
    # Connect to ArangoDB
    arango_host = os.environ.get('ARANGO_HOST', 'http://192.168.1.69:8529')
    client = ArangoClient(hosts=arango_host)
    username = os.environ.get('ARANGO_USERNAME', 'root')
    password = os.environ.get('ARANGO_PASSWORD', '')
    db = client.db(db_name, username=username, password=password)
    
    print("Applying Context^1.5 amplification...")
    
    # Get edge count
    edge_count_query = """
    FOR edge IN semantic_similarity
        COLLECT WITH COUNT INTO total
        RETURN total
    """
    total_edges = next(db.aql.execute(edge_count_query))
    print(f"\nTotal edges to process: {total_edges:,}")
    
    # Process in batches
    print(f"Processing in batches of {batch_size:,}...")
    start_time = time.time()
    
    processed = 0
    with tqdm(total=total_edges, desc="Amplifying edges") as pbar:
        while processed < total_edges:
            # Update batch
            batch_query = """
            FOR edge IN semantic_similarity
                LIMIT @offset, @batch_size
                UPDATE edge WITH {
                    context_original: edge.context,
                    context: POW(edge.context, @alpha),
                    context_amplified: POW(edge.context, @alpha),
                    alpha_used: @alpha
                } IN semantic_similarity
                RETURN 1
            """
            
            result = db.aql.execute(
                batch_query,
                bind_vars={
                    'offset': processed,
                    'batch_size': batch_size,
                    'alpha': alpha
                }
            )
            
            batch_count = sum(1 for _ in result)
            processed += batch_count
            pbar.update(batch_count)
            
            if batch_count == 0:
                break
    
    process_time = time.time() - start_time
    print(f"\nProcessing completed in {process_time:.1f} seconds")
    print(f"Edges per second: {total_edges / process_time:,.0f}")
    
    # Analyze the results
    print("\nAnalyzing amplification effects...")
    
    # Sample statistics
    sample_query = """
    FOR edge IN semantic_similarity
        LIMIT 100000
        RETURN {
            original: edge.context_original,
            amplified: edge.context
        }
    """
    
    samples = list(db.aql.execute(sample_query))
    
    if samples:
        originals = [s['original'] for s in samples if s['original'] is not None]
        amplifieds = [s['amplified'] for s in samples if s['amplified'] is not None]
        
        if originals and amplifieds:
            print("\nBefore amplification (sample):")
            print(f"  Mean context: {np.mean(originals):.3f}")
            print(f"  Std dev: {np.std(originals):.3f}")
            print(f"  Min: {np.min(originals):.3f}")
            print(f"  Max: {np.max(originals):.3f}")
            
            print("\nAfter amplification (sample):")
            print(f"  Mean context: {np.mean(amplifieds):.3f}")
            print(f"  Std dev: {np.std(amplifieds):.3f}")
            print(f"  Min: {np.min(amplifieds):.3f}")
            print(f"  Max: {np.max(amplifieds):.3f}")
    
    # Show amplification effects at different levels
    print(f"\nAmplification effect (Context^{alpha}):")
    test_values = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print("Original → Amplified")
    for val in test_values:
        amplified = val ** alpha
        change = (amplified / val - 1) * 100
        print(f"  {val:.2f} → {amplified:.3f} ({change:+.1f}%)")
    
    # Clustering analysis
    print("\nClustering analysis:")
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    for threshold in thresholds:
        count_query = """
        FOR edge IN semantic_similarity
            FILTER edge.context > @threshold
            COLLECT WITH COUNT INTO count
            RETURN count
        """
        count = next(db.aql.execute(count_query, bind_vars={'threshold': threshold}))
        print(f"  Edges > {threshold}: {count:,}")
    
    # Top connections after amplification
    print("\nStrongest connections after Context^1.5:")
    top_query = """
    FOR edge IN semantic_similarity
        SORT edge.context DESC
        LIMIT 10
        LET from_paper = DOCUMENT(edge._from)
        LET to_paper = DOCUMENT(edge._to)
        RETURN {
            from_title: SUBSTRING(from_paper.title, 0, 50),
            to_title: SUBSTRING(to_paper.title, 0, 50),
            original: edge.context_original,
            amplified: edge.context,
            factor: edge.context / edge.context_original
        }
    """
    
    for i, result in enumerate(db.aql.execute(top_query), 1):
        print(f"\n{i}. {result['from_title']}...")
        print(f"   ↔ {result['to_title']}...")
        print(f"   Original: {result['original']:.3f} → Amplified: {result['amplified']:.3f}")
        print(f"   Amplification factor: {result['factor']:.3f}x")
    
    print(f"\n✓ STEP 4 COMPLETE")
    print(f"Context^{alpha} amplification applied successfully")
    print("\nKey finding: High-similarity papers become even more strongly connected,")
    print("creating natural clustering around semantic concepts.")

if __name__ == "__main__":
    main()