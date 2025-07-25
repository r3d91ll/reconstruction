#!/usr/bin/env python3
"""
STEP 3: Compute semantic similarity between papers using GPU acceleration
Create edges with Context scores
"""

import os
import torch
import numpy as np
from arango import ArangoClient
from tqdm import tqdm
import time

def main():
    # Configuration
    db_name = "information_reconstructionism"
    similarity_threshold = 0.5  # Only create edges for Context > 0.5
    batch_size = 1000  # Process in batches to manage memory
    
    # Connect to ArangoDB
    arango_host = os.environ.get('ARANGO_HOST', 'http://192.168.1.69:8529')
    client = ArangoClient(hosts=arango_host)
    username = os.environ.get('ARANGO_USERNAME', 'root')
    password = os.environ.get('ARANGO_PASSWORD', '')
    db = client.db(db_name, username=username, password=password)
    
    # Create edge collection
    if not db.has_collection('semantic_similarity'):
        similarity_collection = db.create_collection('semantic_similarity', edge=True)
        print("Created edge collection: semantic_similarity")
    else:
        # Skip deletion for now - just use existing collection
        similarity_collection = db.collection('semantic_similarity')
        print("Using existing semantic_similarity collection")
    
    # Get all papers with embeddings
    papers_collection = db.collection('papers')
    
    print("Loading papers from ArangoDB...")
    papers_query = """
    FOR p IN papers
        FILTER p.embeddings != null
        RETURN {
            _id: p._id,
            _key: p._key,
            embeddings: p.embeddings,
            year: p.year,
            primary_category: p.primary_category,
            title: p.title
        }
    """
    
    papers = list(db.aql.execute(papers_query))
    total_papers = len(papers)
    print(f"Loaded {total_papers} papers with embeddings")
    
    if total_papers == 0:
        print("No papers with embeddings found!")
        return
    
    # Extract embeddings and metadata
    print("\nPreparing data for GPU...")
    embeddings_list = []
    paper_ids = []
    paper_metadata = {}
    
    for paper in papers:
        paper_ids.append(paper['_id'])
        embeddings_list.append(paper['embeddings'])
        paper_metadata[paper['_id']] = {
            'year': paper.get('year', 2024),
            'category': paper.get('primary_category', 'unknown'),
            'title': paper.get('title', 'Untitled')
        }
    
    # Convert to GPU tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    embeddings = torch.tensor(embeddings_list, dtype=torch.float32, device=device)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Normalize embeddings for cosine similarity
    print("\nNormalizing embeddings...")
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    # Compute similarity matrix in batches
    print(f"\nComputing similarities for {total_papers} papers...")
    print(f"Total comparisons: {total_papers * (total_papers - 1) // 2:,}")
    
    edges_to_insert = []
    total_edges = 0
    start_time = time.time()
    
    # Process in batches to manage memory
    for i in tqdm(range(0, total_papers, batch_size), desc="Computing similarities"):
        end_i = min(i + batch_size, total_papers)
        batch_embeddings = embeddings[i:end_i]
        
        # Compute similarities with all papers
        similarities = torch.matmul(batch_embeddings, embeddings.T)
        
        # Process results
        for local_idx in range(end_i - i):
            global_idx = i + local_idx
            
            # Only process upper triangle (avoid duplicates)
            for j in range(global_idx + 1, total_papers):
                sim_score = similarities[local_idx, j].item()
                
                if sim_score > similarity_threshold:
                    # Calculate metadata
                    from_id = paper_ids[global_idx]
                    to_id = paper_ids[j]
                    
                    year_gap = abs(
                        paper_metadata[from_id]['year'] - 
                        paper_metadata[to_id]['year']
                    )
                    
                    category_match = (
                        paper_metadata[from_id]['category'] == 
                        paper_metadata[to_id]['category']
                    )
                    
                    edge = {
                        '_from': from_id,
                        '_to': to_id,
                        'context': float(sim_score),
                        'year_gap': year_gap,
                        'category_match': category_match
                    }
                    
                    edges_to_insert.append(edge)
                    total_edges += 1
                    
                    # Insert in batches
                    if len(edges_to_insert) >= 10000:
                        similarity_collection.insert_many(edges_to_insert)
                        edges_to_insert = []
    
    # Insert remaining edges
    if edges_to_insert:
        similarity_collection.insert_many(edges_to_insert)
    
    compute_time = time.time() - start_time
    
    print(f"\n✓ STEP 3 COMPLETE")
    print(f"Created {total_edges:,} semantic similarity edges")
    print(f"Computation time: {compute_time:.1f} seconds")
    print(f"Comparisons per second: {(total_papers * (total_papers - 1) // 2) / compute_time:,.0f}")
    
    # Verification analysis
    print("\nVerification analysis:")
    
    # Context distribution
    context_dist_query = """
    FOR edge IN semantic_similarity
        COLLECT context_range = FLOOR(edge.context * 10) / 10
        WITH COUNT INTO count
        SORT context_range
        RETURN {
            range: CONCAT(TO_STRING(context_range), "-", TO_STRING(context_range + 0.1)),
            count: count
        }
    """
    
    print("\nContext score distribution:")
    for result in db.aql.execute(context_dist_query):
        if result['count'] > 0:
            print(f"  {result['range']}: {result['count']} edges")
    
    # Top similarity pairs
    top_pairs_query = """
    FOR edge IN semantic_similarity
        SORT edge.context DESC
        LIMIT 5
        LET from_paper = DOCUMENT(edge._from)
        LET to_paper = DOCUMENT(edge._to)
        RETURN {
            context: edge.context,
            from_title: SUBSTRING(from_paper.title, 0, 40),
            to_title: SUBSTRING(to_paper.title, 0, 40),
            category_match: edge.category_match
        }
    """
    
    print("\nHighest similarity pairs:")
    for i, result in enumerate(db.aql.execute(top_pairs_query), 1):
        print(f"  Context={result['context']:.3f}: {result['from_title']}... ↔ {result['to_title']}...")
        print(f"    Same category: {result['category_match']}")

    # GPU memory usage
    if device.type == 'cuda':
        print(f"\nGPU memory used: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()