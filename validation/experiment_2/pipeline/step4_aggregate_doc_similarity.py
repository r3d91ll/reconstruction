#!/usr/bin/env python3
"""
Step 4: Aggregate chunk similarities to document level
"""

import os
import json
import numpy as np
from pathlib import Path
from arango import ArangoClient
from datetime import datetime
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Configuration
    db_name = os.environ.get("EXP2_DB_NAME", "information_reconstructionism_exp2")
    results_dir = os.environ.get("EXP2_RESULTS_DIR", ".")
    aggregation_method = os.environ.get("EXP2_AGG_METHOD", "max")  # max, mean, weighted_mean
    
    # ArangoDB connection
    arango_host = os.environ.get('ARANGO_HOST', 'http://192.168.1.69:8529')
    username = os.environ.get('ARANGO_USERNAME', 'root')
    password = os.environ.get('ARANGO_PASSWORD', '')
    
    logger.info("=" * 60)
    logger.info("STEP 4: AGGREGATE DOCUMENT SIMILARITIES")
    logger.info("=" * 60)
    logger.info(f"Database: {db_name}")
    logger.info(f"Aggregation method: {aggregation_method}")
    
    # Connect to database
    client = ArangoClient(hosts=arango_host)
    db = client.db(db_name, username=username, password=password)
    
    # Get collections
    papers_coll = db.collection('papers_exp2')
    doc_similarities_coll = db.collection('document_similarities_exp2')
    
    # Clear existing document similarities
    logger.info("Clearing existing document similarities...")
    doc_similarities_coll.truncate()
    
    # Get all papers
    papers = list(papers_coll.all())
    logger.info(f"Processing {len(papers)} papers")
    
    # Query to get chunk similarities between papers
    chunk_sim_query = """
    FOR paper1 IN papers_exp2
        FOR paper2 IN papers_exp2
            FILTER paper1._key < paper2._key
            
            LET chunk_sims = (
                FOR edge IN chunk_similarities_exp2
                    FILTER (edge.from_paper == paper1.arxiv_id AND edge.to_paper == paper2.arxiv_id)
                        OR (edge.from_paper == paper2.arxiv_id AND edge.to_paper == paper1.arxiv_id)
                    RETURN edge.similarity
            )
            
            FILTER LENGTH(chunk_sims) > 0
            
            RETURN {
                paper1: paper1,
                paper2: paper2,
                chunk_similarities: chunk_sims,
                num_similarities: LENGTH(chunk_sims)
            }
    """
    
    logger.info("\nComputing document-level similarities...")
    doc_similarities_computed = 0
    
    for result in tqdm(db.aql.execute(chunk_sim_query), desc="Aggregating similarities"):
        paper1 = result['paper1']
        paper2 = result['paper2']
        chunk_sims = result['chunk_similarities']
        
        # Aggregate chunk similarities based on method
        if aggregation_method == "max":
            doc_similarity = max(chunk_sims)
        elif aggregation_method == "mean":
            doc_similarity = sum(chunk_sims) / len(chunk_sims)
        elif aggregation_method == "weighted_mean":
            # Weight by number of chunk comparisons (more chunks = more confidence)
            weight = min(len(chunk_sims) / 10.0, 1.0)  # Cap weight at 1.0
            doc_similarity = (sum(chunk_sims) / len(chunk_sims)) * weight
        else:
            doc_similarity = max(chunk_sims)  # Default to max
        
        # Create document similarity edge
        doc_sim_edge = {
            '_from': f"papers_exp2/{paper1['_key']}",
            '_to': f"papers_exp2/{paper2['_key']}",
            'similarity': doc_similarity,
            'aggregation_method': aggregation_method,
            'num_chunk_similarities': len(chunk_sims),
            'min_chunk_sim': min(chunk_sims),
            'max_chunk_sim': max(chunk_sims),
            'mean_chunk_sim': sum(chunk_sims) / len(chunk_sims),
            'std_chunk_sim': float(np.std(chunk_sims)) if len(chunk_sims) > 1 else 0.0,
            'computed_at': datetime.now().isoformat()
        }
        
        doc_similarities_coll.insert(doc_sim_edge)
        doc_similarities_computed += 1
    
    # Verify results
    logger.info("\nVerifying document similarities:")
    
    # Count document similarities
    doc_sim_count = doc_similarities_coll.count()
    logger.info(f"  Document similarities computed: {doc_similarities_computed}")
    logger.info(f"  Document similarities in database: {doc_sim_count}")
    
    # Top similar document pairs
    top_pairs_query = """
    FOR edge IN document_similarities_exp2
        SORT edge.similarity DESC
        LIMIT 10
        LET from_paper = DOCUMENT(edge._from)
        LET to_paper = DOCUMENT(edge._to)
        RETURN {
            similarity: edge.similarity,
            from_title: from_paper.title,
            to_title: to_paper.title,
            num_chunk_sims: edge.num_chunk_similarities,
            chunk_sim_range: CONCAT("[", edge.min_chunk_sim, " - ", edge.max_chunk_sim, "]")
        }
    """
    
    logger.info("\nTop 10 most similar document pairs:")
    for result in db.aql.execute(top_pairs_query):
        logger.info(f"\n  Document similarity: {result['similarity']:.4f}")
        logger.info(f"  From: {result['from_title'][:80]}...")
        logger.info(f"  To: {result['to_title'][:80]}...")
        logger.info(f"  Based on {result['num_chunk_sims']} chunk comparisons")
        logger.info(f"  Chunk similarity range: {result['chunk_sim_range']}")
    
    # Distribution analysis
    dist_query = """
    FOR edge IN document_similarities_exp2
        COLLECT bucket = FLOOR(edge.similarity * 10) / 10
        WITH COUNT INTO count
        SORT bucket DESC
        RETURN {
            bucket: bucket,
            count: count
        }
    """
    
    logger.info("\nDocument similarity distribution:")
    for result in db.aql.execute(dist_query):
        bucket = result['bucket']
        count = result['count']
        logger.info(f"  [{bucket:.1f} - {bucket + 0.1:.1f}): {count} pairs")
    
    # Paper connectivity analysis
    connectivity_query = """
    FOR paper IN papers_exp2
        LET outgoing = LENGTH(
            FOR edge IN document_similarities_exp2
                FILTER edge._from == CONCAT("papers_exp2/", paper._key)
                RETURN 1
        )
        LET incoming = LENGTH(
            FOR edge IN document_similarities_exp2
                FILTER edge._to == CONCAT("papers_exp2/", paper._key)
                RETURN 1
        )
        LET total_connections = outgoing + incoming
        FILTER total_connections > 0
        SORT total_connections DESC
        LIMIT 10
        RETURN {
            title: paper.title,
            connections: total_connections,
            outgoing: outgoing,
            incoming: incoming
        }
    """
    
    logger.info("\nMost connected papers:")
    for result in db.aql.execute(connectivity_query):
        logger.info(f"  {result['title'][:60]}...")
        logger.info(f"    Total connections: {result['connections']} (out: {result['outgoing']}, in: {result['incoming']})")
    
    # Save summary
    summary_query = """
    RETURN {
        total_documents: LENGTH(papers_exp2),
        total_doc_similarities: LENGTH(document_similarities_exp2),
        avg_similarity: AVG(FOR e IN document_similarities_exp2 RETURN e.similarity),
        max_similarity: MAX(FOR e IN document_similarities_exp2 RETURN e.similarity),
        min_similarity: MIN(FOR e IN document_similarities_exp2 RETURN e.similarity),
        avg_chunk_comparisons: AVG(FOR e IN document_similarities_exp2 RETURN e.num_chunk_similarities)
    }
    """
    
    summary_result = next(db.aql.execute(summary_query))
    summary = {
        **summary_result,
        'aggregation_method': aggregation_method,
        'timestamp': datetime.now().isoformat()
    }
    
    summary_path = os.path.join(results_dir, "document_similarity_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nSummary statistics:")
    logger.info(f"  Total documents: {summary['total_documents']}")
    logger.info(f"  Document similarity edges: {summary['total_doc_similarities']}")
    logger.info(f"  Average similarity: {summary['avg_similarity']:.4f}")
    logger.info(f"  Similarity range: [{summary['min_similarity']:.4f}, {summary['max_similarity']:.4f}]")
    logger.info(f"  Average chunk comparisons per document pair: {summary['avg_chunk_comparisons']:.1f}")
    
    logger.info(f"\nSummary saved to: {summary_path}")
    logger.info("\n✓ STEP 4 COMPLETE")
    
    return 0

if __name__ == "__main__":
    exit(main())