#!/usr/bin/env python3
"""
Step 3: Compute chunk-level similarities using Jina V4 embeddings
"""

import os
import json
import numpy as np
from pathlib import Path
from arango import ArangoClient
from datetime import datetime
import logging
from tqdm import tqdm
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))

def main():
    # Configuration
    db_name = os.environ.get("EXP2_DB_NAME", "information_reconstructionism_exp2")
    results_dir = os.environ.get("EXP2_RESULTS_DIR", ".")
    batch_size = int(os.environ.get("EXP2_BATCH_SIZE", "1000"))
    similarity_threshold = float(os.environ.get("EXP2_SIM_THRESHOLD", "0.5"))
    
    # ArangoDB connection
    arango_host = os.environ.get('ARANGO_HOST', 'http://192.168.1.69:8529')
    username = os.environ.get('ARANGO_USERNAME', 'root')
    password = os.environ.get('ARANGO_PASSWORD', '')
    
    logger.info("=" * 60)
    logger.info("STEP 3: COMPUTE CHUNK SIMILARITIES")
    logger.info("=" * 60)
    logger.info(f"Database: {db_name}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Similarity threshold: {similarity_threshold}")
    
    # Connect to database
    client = ArangoClient(hosts=arango_host)
    db = client.db(db_name, username=username, password=password)
    
    # Get collections
    chunks_coll = db.collection('chunks_exp2')
    similarities_coll = db.collection('chunk_similarities_exp2')
    
    # Clear existing similarities
    logger.info("Clearing existing similarities...")
    similarities_coll.truncate()
    
    # Get total chunk count
    total_chunks = chunks_coll.count()
    logger.info(f"Total chunks to process: {total_chunks}")
    
    # Process chunks in batches
    logger.info("\nFetching all chunks...")
    all_chunks = []
    
    # Use cursor to fetch all chunks
    cursor = chunks_coll.all()
    for chunk in cursor:
        all_chunks.append(chunk)
    
    logger.info(f"Loaded {len(all_chunks)} chunks")
    
    # Compute similarities
    logger.info("\nComputing chunk similarities...")
    similarities_computed = 0
    similarities_above_threshold = 0
    
    # Progress tracking
    total_comparisons = (len(all_chunks) * (len(all_chunks) - 1)) // 2
    
    with tqdm(total=total_comparisons, desc="Computing similarities") as pbar:
        for i in range(len(all_chunks)):
            chunk1 = all_chunks[i]
            
            # Batch similarities for this chunk
            batch_similarities = []
            
            for j in range(i + 1, len(all_chunks)):
                chunk2 = all_chunks[j]
                
                # Skip if same paper
                if chunk1['paper_id'] == chunk2['paper_id']:
                    pbar.update(1)
                    continue
                
                # Compute similarity
                similarity = cosine_similarity(chunk1['embedding'], chunk2['embedding'])
                similarities_computed += 1
                
                if similarity >= similarity_threshold:
                    similarities_above_threshold += 1
                    
                    # Create similarity edge
                    similarity_doc = {
                        '_from': f"chunks_exp2/{chunk1['_key']}",
                        '_to': f"chunks_exp2/{chunk2['_key']}",
                        'similarity': similarity,
                        'from_paper': chunk1['paper_id'],
                        'to_paper': chunk2['paper_id'],
                        'computed_at': datetime.now().isoformat()
                    }
                    batch_similarities.append(similarity_doc)
                
                pbar.update(1)
                
                # Insert batch when it reaches size limit
                if len(batch_similarities) >= batch_size:
                    similarities_coll.insert_many(batch_similarities)
                    batch_similarities = []
            
            # Insert remaining similarities
            if batch_similarities:
                similarities_coll.insert_many(batch_similarities)
    
    # Verify results
    logger.info("\nVerifying computed similarities:")
    
    # Count similarities
    similarity_count = similarities_coll.count()
    logger.info(f"  Total similarities computed: {similarities_computed}")
    logger.info(f"  Similarities above threshold ({similarity_threshold}): {similarities_above_threshold}")
    logger.info(f"  Similarities stored in database: {similarity_count}")
    
    # Sample high-similarity pairs
    query = """
    FOR edge IN chunk_similarities_exp2
        SORT edge.similarity DESC
        LIMIT 10
        LET from_chunk = DOCUMENT(edge._from)
        LET to_chunk = DOCUMENT(edge._to)
        RETURN {
            similarity: edge.similarity,
            from_paper: edge.from_paper,
            to_paper: edge.to_paper,
            from_text: SUBSTRING(from_chunk.text, 0, 100),
            to_text: SUBSTRING(to_chunk.text, 0, 100)
        }
    """
    
    logger.info("\nTop 10 most similar chunk pairs:")
    for result in db.aql.execute(query):
        logger.info(f"\n  Similarity: {result['similarity']:.4f}")
        logger.info(f"  From: {result['from_paper']}")
        logger.info(f"  To: {result['to_paper']}")
        logger.info(f"  From text: {result['from_text']}...")
        logger.info(f"  To text: {result['to_text']}...")
    
    # Distribution analysis
    query = """
    FOR edge IN chunk_similarities_exp2
        COLLECT bucket = FLOOR(edge.similarity * 10) / 10
        WITH COUNT INTO count
        SORT bucket DESC
        RETURN {
            bucket: bucket,
            count: count
        }
    """
    
    logger.info("\nSimilarity distribution:")
    for result in db.aql.execute(query):
        bucket = result['bucket']
        count = result['count']
        logger.info(f"  [{bucket:.1f} - {bucket + 0.1:.1f}): {count} pairs")
    
    # Save summary statistics
    summary = {
        'total_chunks': total_chunks,
        'total_comparisons': similarities_computed,
        'similarities_above_threshold': similarities_above_threshold,
        'threshold': similarity_threshold,
        'timestamp': datetime.now().isoformat()
    }
    
    summary_path = os.path.join(results_dir, "chunk_similarity_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nSummary saved to: {summary_path}")
    logger.info("\n✓ STEP 3 COMPLETE")
    
    return 0

if __name__ == "__main__":
    exit(main())