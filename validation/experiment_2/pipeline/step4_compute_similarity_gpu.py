#!/usr/bin/env python3
"""
Step 3: Compute chunk-level similarities using Jina V4 embeddings (GPU-accelerated version)
Uses both A6000 GPUs with NVLink for efficient similarity computation
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from arango import ArangoClient
from datetime import datetime
import logging
from tqdm import tqdm
import math

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GPUSimilarityComputer:
    def __init__(self, device_ids=[0, 1], dtype=torch.float16):
        """
        Initialize GPU similarity computer with multi-GPU support
        
        Args:
            device_ids: List of GPU IDs to use (default: [0, 1] for both A6000s)
            dtype: Data type for tensors (float16 for memory efficiency)
        """
        self.device_ids = device_ids
        self.dtype = dtype
        self.primary_device = torch.device(f'cuda:{device_ids[0]}')
        
        # Check GPU availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        
        gpu_count = torch.cuda.device_count()
        logger.info(f"Found {gpu_count} GPUs")
        
        for device_id in device_ids:
            if device_id >= gpu_count:
                raise ValueError(f"GPU {device_id} not available. Only {gpu_count} GPUs found.")
            
            gpu_name = torch.cuda.get_device_name(device_id)
            gpu_memory = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
            logger.info(f"GPU {device_id}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    def compute_similarity_matrix(self, embeddings1, embeddings2=None):
        """
        Compute cosine similarity matrix between two sets of embeddings
        
        Args:
            embeddings1: First set of embeddings (n x d)
            embeddings2: Second set of embeddings (m x d), if None uses embeddings1
        
        Returns:
            Similarity matrix (n x m)
        """
        # Convert to tensors and move to GPU
        if isinstance(embeddings1, np.ndarray):
            embeddings1 = torch.from_numpy(embeddings1).to(self.dtype)
        if embeddings2 is not None and isinstance(embeddings2, np.ndarray):
            embeddings2 = torch.from_numpy(embeddings2).to(self.dtype)
        
        # Move to primary GPU
        embeddings1 = embeddings1.to(self.primary_device)
        if embeddings2 is not None:
            embeddings2 = embeddings2.to(self.primary_device)
        else:
            embeddings2 = embeddings1
        
        # Normalize embeddings
        embeddings1_norm = nn.functional.normalize(embeddings1, p=2, dim=1)
        embeddings2_norm = nn.functional.normalize(embeddings2, p=2, dim=1)
        
        # Use DataParallel for multi-GPU if batch is large enough
        if len(self.device_ids) > 1 and embeddings1.shape[0] > 1000:
            # Create a simple module for similarity computation
            class SimilarityModule(nn.Module):
                def forward(self, x1, x2):
                    return torch.mm(x1, x2.t())
            
            model = SimilarityModule()
            model = nn.DataParallel(model, device_ids=self.device_ids)
            similarity_matrix = model(embeddings1_norm, embeddings2_norm)
        else:
            # Single GPU computation
            similarity_matrix = torch.mm(embeddings1_norm, embeddings2_norm.t())
        
        return similarity_matrix
    
    def process_batch_similarities(self, chunks_batch1, chunks_batch2, similarity_threshold):
        """
        Process similarities between two batches of chunks
        
        Returns:
            List of similarity documents above threshold
        """
        # Extract embeddings
        embeddings1 = np.array([chunk['embedding'] for chunk in chunks_batch1])
        embeddings2 = np.array([chunk['embedding'] for chunk in chunks_batch2])
        
        # Compute similarity matrix on GPU
        similarity_matrix = self.compute_similarity_matrix(embeddings1, embeddings2)
        
        # Find similarities above threshold
        mask = similarity_matrix >= similarity_threshold
        indices = torch.nonzero(mask, as_tuple=False)
        
        # Convert back to CPU for processing
        indices_cpu = indices.cpu().numpy()
        similarities_cpu = similarity_matrix[mask].cpu().numpy()
        
        # Create similarity documents
        similarity_docs = []
        for idx, (i, j) in enumerate(indices_cpu):
            chunk1 = chunks_batch1[i]
            chunk2 = chunks_batch2[j]
            
            # Skip same paper comparisons
            if chunk1['paper_id'] == chunk2['paper_id']:
                continue
            
            similarity_doc = {
                '_from': f"chunks_exp2/{chunk1['_key']}",
                '_to': f"chunks_exp2/{chunk2['_key']}",
                'similarity': float(similarities_cpu[idx]),
                'from_paper': chunk1['paper_id'],
                'to_paper': chunk2['paper_id'],
                'computed_at': datetime.now().isoformat()
            }
            similarity_docs.append(similarity_doc)
        
        return similarity_docs

def main():
    # Configuration
    db_name = os.environ.get("EXP2_DB_NAME", "information_reconstructionism_exp2")
    results_dir = os.environ.get("EXP2_RESULTS_DIR", ".")
    batch_size = int(os.environ.get("EXP2_BATCH_SIZE", "10000"))  # Larger batches for GPU
    similarity_threshold = float(os.environ.get("EXP2_SIM_THRESHOLD", "0.5"))
    
    # GPU configuration
    gpu_ids = [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")]
    use_fp16 = os.environ.get("EXP2_USE_FP16", "true").lower() == "true"
    dtype = torch.float16 if use_fp16 else torch.float32
    
    # ArangoDB connection
    arango_host = os.environ.get('ARANGO_HOST')
    username = os.environ.get('ARANGO_USERNAME', 'root')
    password = os.environ.get('ARANGO_PASSWORD')
    
    # Validate required environment variables
    if not arango_host:
        raise ValueError("ARANGO_HOST environment variable must be set")
    if not password:
        raise ValueError("ARANGO_PASSWORD environment variable must be set")
    
    logger.info("=" * 60)
    logger.info("STEP 3: COMPUTE CHUNK SIMILARITIES (GPU-ACCELERATED)")
    logger.info("=" * 60)
    logger.info(f"Database: {db_name}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Similarity threshold: {similarity_threshold}")
    logger.info(f"Using GPUs: {gpu_ids}")
    logger.info(f"Data type: {dtype}")
    
    # Initialize GPU computer
    gpu_computer = GPUSimilarityComputer(device_ids=gpu_ids, dtype=dtype)
    
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
    
    # Estimate memory requirements
    embedding_dim = 1024  # Jina v4 dimension
    memory_per_chunk = embedding_dim * 2 if dtype == torch.float16 else embedding_dim * 4
    total_memory_gb = (total_chunks * memory_per_chunk) / (1024**3)
    logger.info(f"Estimated memory for all embeddings: {total_memory_gb:.2f} GB")
    
    # Determine optimal batch size based on GPU memory
    gpu_memory_gb = 48  # A6000 has 48GB
    safety_factor = 0.7  # Use only 70% of GPU memory
    max_batch_size = int((gpu_memory_gb * safety_factor * 1024**3) / (memory_per_chunk * 2))
    chunk_batch_size = min(batch_size, max_batch_size)
    logger.info(f"Using chunk batch size: {chunk_batch_size}")
    
    # Load all chunks (if fits in memory) or process in batches
    if total_memory_gb < 10:  # If all embeddings fit in 10GB
        logger.info("Loading all chunks into memory for efficient GPU processing...")
        all_chunks = list(db.aql.execute("FOR c IN chunks_exp2 RETURN c"))
        
        # Process all-vs-all on GPU
        logger.info("Computing all-vs-all similarities on GPU...")
        embeddings = np.array([chunk['embedding'] for chunk in all_chunks])
        
        # Split computation into manageable chunks to avoid OOM
        n_chunks = len(all_chunks)
        n_splits = math.ceil(n_chunks / chunk_batch_size)
        
        similarities_computed = 0
        similarities_above_threshold = 0
        
        with tqdm(total=n_splits * n_splits, desc="Computing similarities") as pbar:
            for i in range(n_splits):
                start_i = i * chunk_batch_size
                end_i = min((i + 1) * chunk_batch_size, n_chunks)
                batch_i = all_chunks[start_i:end_i]
                
                for j in range(i, n_splits):  # Only compute upper triangle
                    start_j = j * chunk_batch_size
                    end_j = min((j + 1) * chunk_batch_size, n_chunks)
                    batch_j = all_chunks[start_j:end_j]
                    
                    # Compute similarities for this batch pair
                    similarity_docs = gpu_computer.process_batch_similarities(
                        batch_i, batch_j, similarity_threshold
                    )
                    
                    similarities_computed += len(batch_i) * len(batch_j)
                    similarities_above_threshold += len(similarity_docs)
                    
                    # Insert in batches
                    if similarity_docs:
                        for k in range(0, len(similarity_docs), 10000):
                            similarities_coll.insert_many(similarity_docs[k:k+10000])
                    
                    pbar.update(1)
                    
                    # Clear GPU cache periodically
                    if (i * n_splits + j) % 10 == 0:
                        torch.cuda.empty_cache()
    
    else:
        # Process in memory-efficient batches
        logger.info("Processing chunks in memory-efficient batches...")
        similarities_computed = 0
        similarities_above_threshold = 0
        
        # Process chunks in sliding window batches
        for offset_i in tqdm(range(0, total_chunks, chunk_batch_size), desc="Processing batches"):
            # Fetch batch i
            query_i = f"FOR c IN chunks_exp2 LIMIT {offset_i}, {chunk_batch_size} RETURN c"
            batch_i = list(db.aql.execute(query_i))
            
            if not batch_i:
                break
            
            # Compare with all subsequent chunks
            for offset_j in range(offset_i, total_chunks, chunk_batch_size):
                # Fetch batch j
                query_j = f"FOR c IN chunks_exp2 LIMIT {offset_j}, {chunk_batch_size} RETURN c"
                batch_j = list(db.aql.execute(query_j))
                
                if not batch_j:
                    break
                
                # Skip if comparing identical batches (unless it's the diagonal)
                if offset_i == offset_j:
                    # For diagonal blocks, we need to handle upper triangle only
                    embeddings = np.array([chunk['embedding'] for chunk in batch_i])
                    similarity_matrix = gpu_computer.compute_similarity_matrix(embeddings)
                    
                    # Process upper triangle only
                    similarity_docs = []
                    for i in range(len(batch_i)):
                        for j in range(i + 1, len(batch_i)):
                            if batch_i[i]['paper_id'] != batch_i[j]['paper_id']:
                                sim = similarity_matrix[i, j].item()
                                if sim >= similarity_threshold:
                                    similarity_doc = {
                                        '_from': f"chunks_exp2/{batch_i[i]['_key']}",
                                        '_to': f"chunks_exp2/{batch_i[j]['_key']}",
                                        'similarity': float(sim),
                                        'from_paper': batch_i[i]['paper_id'],
                                        'to_paper': batch_i[j]['paper_id'],
                                        'computed_at': datetime.now().isoformat()
                                    }
                                    similarity_docs.append(similarity_doc)
                else:
                    # Compute similarities for this batch pair
                    similarity_docs = gpu_computer.process_batch_similarities(
                        batch_i, batch_j, similarity_threshold
                    )
                
                similarities_computed += len(batch_i) * len(batch_j)
                similarities_above_threshold += len(similarity_docs)
                
                # Insert in batches
                if similarity_docs:
                    for k in range(0, len(similarity_docs), 10000):
                        similarities_coll.insert_many(similarity_docs[k:k+10000])
                
                # Clear GPU cache periodically
                torch.cuda.empty_cache()
    
    # Verify results
    logger.info("\nVerifying computed similarities:")
    
    # Count similarities
    similarity_count = similarities_coll.count()
    logger.info(f"  Total comparisons made: {similarities_computed}")
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
    
    # GPU memory stats
    for device_id in gpu_ids:
        allocated = torch.cuda.memory_allocated(device_id) / 1024**3
        reserved = torch.cuda.memory_reserved(device_id) / 1024**3
        logger.info(f"\nGPU {device_id} memory usage:")
        logger.info(f"  Allocated: {allocated:.2f} GB")
        logger.info(f"  Reserved: {reserved:.2f} GB")
    
    # Save summary statistics
    summary = {
        'total_chunks': total_chunks,
        'total_comparisons': similarities_computed,
        'similarities_above_threshold': similarities_above_threshold,
        'threshold': similarity_threshold,
        'gpu_ids': gpu_ids,
        'batch_size': chunk_batch_size,
        'dtype': str(dtype),
        'timestamp': datetime.now().isoformat()
    }
    
    summary_path = os.path.join(results_dir, "chunk_similarity_summary_gpu.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nSummary saved to: {summary_path}")
    logger.info("\n✓ STEP 3 (GPU-ACCELERATED) COMPLETE")
    
    return 0

if __name__ == "__main__":
    exit(main())