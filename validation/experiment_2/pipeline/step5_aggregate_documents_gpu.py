#!/usr/bin/env python3
"""
Step 5: GPU-Accelerated Document Aggregation
Aggregates chunk-level similarities to document level using GPU
"""

import os
import json
import torch
import numpy as np
from datetime import datetime
import logging
from arango import ArangoClient
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
import gc

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GPUDocumentAggregator:
    def __init__(self, device_ids=[0, 1], aggregation_method='weighted_mean'):
        """
        Initialize GPU document aggregator
        
        Args:
            device_ids: List of GPU IDs to use
            aggregation_method: Method for aggregating chunk similarities
        """
        self.device_ids = device_ids
        self.primary_device = torch.device(f'cuda:{device_ids[0]}')
        self.aggregation_method = aggregation_method
        
        logger.info(f"GPU Document Aggregator initialized")
        logger.info(f"Using GPUs: {device_ids}")
        logger.info(f"Aggregation method: {aggregation_method}")
    
    def aggregate_chunk_similarities(self, chunk_sims, method='weighted_mean'):
        """
        Aggregate chunk-level similarities to document level on GPU
        
        Args:
            chunk_sims: List of (similarity, weight) tuples
            method: Aggregation method
        
        Returns:
            Aggregated similarity score
        """
        if not chunk_sims:
            return 0.0
        
        # Convert to tensors
        similarities = torch.tensor([s[0] for s in chunk_sims], device=self.primary_device)
        weights = torch.tensor([s[1] for s in chunk_sims], device=self.primary_device)
        
        if method == 'max':
            return similarities.max().item()
        elif method == 'mean':
            return similarities.mean().item()
        elif method == 'weighted_mean':
            # Normalize weights
            weights = weights / weights.sum()
            return (similarities * weights).sum().item()
        elif method == 'top_k_mean':
            # Average of top-k similarities
            k = min(5, len(similarities))
            top_k = torch.topk(similarities, k)[0]
            return top_k.mean().item()
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def compute_document_embeddings(self, chunks_by_paper):
        """
        Compute document-level embeddings from chunks on GPU
        
        Args:
            chunks_by_paper: Dict mapping paper_id to list of chunk embeddings
        
        Returns:
            Dict mapping paper_id to document embedding
        """
        doc_embeddings = {}
        
        for paper_id, chunk_embeddings in tqdm(chunks_by_paper.items(), desc="Computing doc embeddings"):
            if not chunk_embeddings:
                continue
            
            # Convert to tensor
            embeddings_tensor = torch.tensor(chunk_embeddings, device=self.primary_device)
            
            # Compute document embedding (weighted average)
            # Weight by position (later chunks often less important)
            weights = torch.exp(-torch.arange(len(chunk_embeddings), device=self.primary_device) * 0.1)
            weights = weights / weights.sum()
            
            # Weighted average
            doc_embedding = (embeddings_tensor * weights.unsqueeze(1)).sum(dim=0)
            
            # Normalize
            doc_embedding = F.normalize(doc_embedding.unsqueeze(0), p=2, dim=1).squeeze()
            
            doc_embeddings[paper_id] = doc_embedding.cpu().numpy()
        
        return doc_embeddings

def main():
    # Configuration
    db_name = os.environ.get("EXP2_DB_NAME", "information_reconstructionism_exp2_gpu")
    results_dir = os.environ.get("EXP2_RESULTS_DIR", ".")
    
    # GPU configuration
    gpu_ids = [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")]
    aggregation_method = os.environ.get("EXP2_AGGREGATION_METHOD", "weighted_mean")
    similarity_threshold = float(os.environ.get("EXP2_DOC_SIM_THRESHOLD", "0.3"))
    
    # ArangoDB connection
    arango_host = os.environ.get('ARANGO_HOST', 'http://192.168.1.69:8529')
    username = os.environ.get('ARANGO_USERNAME', 'root')
    password = os.environ.get('ARANGO_PASSWORD', '')
    
    logger.info("=" * 60)
    logger.info("STEP 5: GPU-ACCELERATED DOCUMENT AGGREGATION")
    logger.info("=" * 60)
    logger.info(f"Database: {db_name}")
    logger.info(f"Using GPUs: {gpu_ids}")
    logger.info(f"Aggregation method: {aggregation_method}")
    logger.info(f"Document similarity threshold: {similarity_threshold}")
    
    # Initialize GPU aggregator
    aggregator = GPUDocumentAggregator(device_ids=gpu_ids, aggregation_method=aggregation_method)
    
    # Connect to database
    client = ArangoClient(hosts=arango_host)
    db = client.db(db_name, username=username, password=password)
    
    # Create document similarities collection
    if not db.has_collection('document_similarities_exp2'):
        db.create_collection('document_similarities_exp2', edge=True)
    
    doc_sim_coll = db.collection('document_similarities_exp2')
    doc_sim_coll.truncate()
    
    start_time = datetime.now()
    
    # Step 1: Load all chunk embeddings grouped by paper
    logger.info("\nLoading chunk embeddings...")
    chunks_by_paper = defaultdict(list)
    
    query = """
    FOR chunk IN chunks_exp2
        RETURN {
            paper_id: chunk.paper_id,
            chunk_id: chunk.chunk_id,
            embedding: chunk.embedding
        }
    """
    
    for result in tqdm(db.aql.execute(query, batch_size=10000), desc="Loading chunks"):
        chunks_by_paper[result['paper_id']].append(result['embedding'])
    
    logger.info(f"Loaded embeddings for {len(chunks_by_paper)} papers")
    
    # Step 2: Compute document-level embeddings on GPU
    logger.info("\nComputing document embeddings on GPU...")
    doc_embeddings = aggregator.compute_document_embeddings(chunks_by_paper)
    
    # Clear chunk data to free memory
    del chunks_by_paper
    gc.collect()
    torch.cuda.empty_cache()
    
    # Step 3: Aggregate chunk similarities to document level
    logger.info("\nAggregating chunk similarities to document level...")
    
    # Query to get chunk similarities grouped by document pairs
    query = """
    FOR edge IN chunk_similarities_exp2
        COLLECT 
            from_paper = edge.from_paper,
            to_paper = edge.to_paper
        INTO group
        LET chunk_sims = (
            FOR item IN group
                RETURN [item.edge.similarity, 1.0]
        )
        RETURN {
            from_paper: from_paper,
            to_paper: to_paper,
            chunk_sims: chunk_sims,
            num_chunk_pairs: LENGTH(group)
        }
    """
    
    doc_similarities = []
    total_pairs = 0
    
    for result in tqdm(db.aql.execute(query, batch_size=1000), desc="Aggregating similarities"):
        # Aggregate on GPU
        doc_sim = aggregator.aggregate_chunk_similarities(
            result['chunk_sims'],
            method=aggregation_method
        )
        
        if doc_sim >= similarity_threshold:
            doc_sim_edge = {
                '_from': f"papers_exp2/{result['from_paper']}",
                '_to': f"papers_exp2/{result['to_paper']}",
                'similarity': doc_sim,
                'num_chunk_pairs': result['num_chunk_pairs'],
                'aggregation_method': aggregation_method,
                'computed_at': datetime.now().isoformat()
            }
            doc_similarities.append(doc_sim_edge)
        
        total_pairs += 1
        
        # Batch insert
        if len(doc_similarities) >= 1000:
            doc_sim_coll.insert_many(doc_similarities)
            doc_similarities = []
    
    # Insert remaining
    if doc_similarities:
        doc_sim_coll.insert_many(doc_similarities)
    
    # Step 4: Compute document similarity using document embeddings
    logger.info("\nComputing direct document similarities on GPU...")
    
    # Convert document embeddings to tensor
    paper_ids = list(doc_embeddings.keys())
    embeddings_matrix = np.array([doc_embeddings[pid] for pid in paper_ids])
    
    # Process in batches on GPU
    batch_size = 100
    direct_similarities = []
    
    for i in tqdm(range(0, len(paper_ids), batch_size), desc="Computing doc similarities"):
        batch_ids = paper_ids[i:i+batch_size]
        batch_embeddings = torch.tensor(
            embeddings_matrix[i:i+batch_size], 
            device=aggregator.primary_device
        )
        
        # Compute similarities with all documents
        all_similarities = torch.mm(batch_embeddings, torch.tensor(embeddings_matrix.T, device=aggregator.primary_device))
        
        # Process results
        for j, from_paper in enumerate(batch_ids):
            # Get top-k similar documents (excluding self)
            similarities = all_similarities[j].cpu().numpy()
            
            # Sort and get top-k
            sorted_indices = np.argsort(similarities)[::-1]
            
            for idx in sorted_indices[:50]:  # Top 50 similar docs
                if paper_ids[idx] != from_paper and similarities[idx] >= similarity_threshold:
                    direct_sim_edge = {
                        '_from': f"papers_exp2/{from_paper}",
                        '_to': f"papers_exp2/{paper_ids[idx]}",
                        'similarity': float(similarities[idx]),
                        'method': 'document_embedding',
                        'computed_at': datetime.now().isoformat()
                    }
                    direct_similarities.append(direct_sim_edge)
            
            # Batch insert
            if len(direct_similarities) >= 1000:
                doc_sim_coll.insert_many(direct_similarities)
                direct_similarities = []
        
        # Clear GPU cache
        torch.cuda.empty_cache()
    
    # Insert remaining
    if direct_similarities:
        doc_sim_coll.insert_many(direct_similarities)
    
    # Calculate statistics
    duration = (datetime.now() - start_time).total_seconds()
    doc_sim_count = doc_sim_coll.count()
    
    logger.info("\n" + "=" * 60)
    logger.info("DOCUMENT AGGREGATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total document pairs processed: {total_pairs:,}")
    logger.info(f"Document similarities above threshold: {doc_sim_count:,}")
    logger.info(f"Total time: {duration:.1f} seconds")
    
    # Analyze results
    query = """
    FOR edge IN document_similarities_exp2
        COLLECT method = edge.method || edge.aggregation_method
        WITH COUNT INTO count
        RETURN {
            method: method,
            count: count
        }
    """
    
    logger.info("\nSimilarities by method:")
    for result in db.aql.execute(query):
        logger.info(f"  {result['method']}: {result['count']:,}")
    
    # Top similar document pairs
    query = """
    FOR edge IN document_similarities_exp2
        SORT edge.similarity DESC
        LIMIT 10
        RETURN {
            from: edge._from,
            to: edge._to,
            similarity: edge.similarity,
            method: edge.method || edge.aggregation_method
        }
    """
    
    logger.info("\nTop 10 most similar document pairs:")
    for result in db.aql.execute(query):
        from_id = result['from'].split('/')[-1]
        to_id = result['to'].split('/')[-1]
        logger.info(f"  {from_id} <-> {to_id}: {result['similarity']:.4f} ({result['method']})")
    
    # GPU memory stats
    for gpu_id in gpu_ids:
        allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
        reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
        logger.info(f"\nGPU {gpu_id} memory usage:")
        logger.info(f"  Allocated: {allocated:.2f} GB")
        logger.info(f"  Reserved: {reserved:.2f} GB")
    
    # Save summary
    summary = {
        'total_document_pairs': total_pairs,
        'document_similarities': doc_sim_count,
        'similarity_threshold': similarity_threshold,
        'aggregation_method': aggregation_method,
        'duration_seconds': duration,
        'gpu_ids': gpu_ids,
        'timestamp': datetime.now().isoformat()
    }
    
    summary_path = os.path.join(results_dir, "document_aggregation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nSummary saved to: {summary_path}")
    logger.info("\n✓ GPU-ACCELERATED DOCUMENT AGGREGATION COMPLETE")
    
    return 0

if __name__ == "__main__":
    exit(main())