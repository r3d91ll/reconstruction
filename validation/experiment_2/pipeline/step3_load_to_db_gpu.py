#!/usr/bin/env python3
"""
Step 3: GPU-Accelerated Database Loading
Batch processes embeddings on GPU before loading to ArangoDB
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from arango import ArangoClient
from tqdm import tqdm
import gc
from concurrent.futures import ThreadPoolExecutor
import torch.nn.functional as F

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GPUBatchProcessor:
    def __init__(self, device_id=0, batch_size=1000):
        """
        Initialize GPU batch processor for embeddings
        
        Args:
            device_id: GPU to use
            batch_size: Batch size for processing
        """
        self.device = torch.device(f'cuda:{device_id}')
        self.batch_size = batch_size
        
        logger.info(f"GPU Batch Processor initialized on GPU {device_id}")
    
    def normalize_embeddings_batch(self, embeddings):
        """Normalize a batch of embeddings on GPU"""
        # Convert to tensor
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        
        embeddings_tensor = torch.from_numpy(embeddings).float().to(self.device)
        
        # L2 normalize
        normalized = F.normalize(embeddings_tensor, p=2, dim=1)
        
        # Return as numpy
        return normalized.cpu().numpy()
    
    def validate_embeddings_batch(self, embeddings):
        """Validate embeddings have correct properties"""
        embeddings_tensor = torch.from_numpy(embeddings).float().to(self.device)
        
        # Check for NaN or Inf
        has_nan = torch.isnan(embeddings_tensor).any()
        has_inf = torch.isinf(embeddings_tensor).any()
        
        # Check norms
        norms = torch.norm(embeddings_tensor, p=2, dim=1)
        
        return {
            'valid': not (has_nan or has_inf),
            'has_nan': has_nan.item(),
            'has_inf': has_inf.item(),
            'min_norm': norms.min().item(),
            'max_norm': norms.max().item(),
            'mean_norm': norms.mean().item()
        }

def process_embedding_file(file_path, gpu_processor):
    """Process a single embedding file with GPU acceleration"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    paper_id = data.get('paper_id')
    if not paper_id:
        # Fallback: try to get from filename
        paper_id = Path(file_path).stem
    chunks = data.get('chunks', [])
    
    # Extract embeddings
    embeddings = []
    for chunk in chunks:
        if 'embedding' in chunk:
            embeddings.append(chunk['embedding'])
    
    if not embeddings:
        logger.warning(f"No embeddings found in {file_path}")
        return None
    
    # Process embeddings on GPU
    embeddings_array = np.array(embeddings)
    
    # Normalize on GPU
    normalized_embeddings = gpu_processor.normalize_embeddings_batch(embeddings_array)
    
    # Validate
    validation = gpu_processor.validate_embeddings_batch(normalized_embeddings)
    
    if not validation['valid']:
        logger.error(f"Invalid embeddings in {paper_id}: {validation}")
        return None
    
    # Update chunks with normalized embeddings
    for i, chunk in enumerate(chunks):
        chunk['embedding'] = normalized_embeddings[i].tolist()
        chunk['embedding_norm'] = float(np.linalg.norm(normalized_embeddings[i]))
    
    # Prepare documents for database
    paper_doc = {
        '_key': paper_id,
        'paper_id': paper_id,
        'metadata': data.get('metadata', {}),
        'num_chunks': len(chunks),
        'embedding_stats': validation,
        'processed_timestamp': datetime.now().isoformat()
    }
    
    chunk_docs = []
    for i, chunk in enumerate(chunks):
        chunk_doc = {
            '_key': f"{paper_id}_chunk_{i}",
            'paper_id': paper_id,
            'chunk_id': i,
            'text': chunk['text'],
            'embedding': chunk['embedding'],
            'embedding_norm': chunk['embedding_norm'],
            'start_char': chunk.get('start_char', 0),
            'end_char': chunk.get('end_char', len(chunk['text']))
        }
        chunk_docs.append(chunk_doc)
    
    return {
        'paper_doc': paper_doc,
        'chunk_docs': chunk_docs,
        'paper_id': paper_id
    }

def batch_insert_documents(db, collection_name, documents, batch_size=1000):
    """Batch insert documents into ArangoDB"""
    collection = db.collection(collection_name)
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        try:
            collection.insert_many(batch, overwrite=True)
        except Exception as e:
            logger.error(f"Error inserting batch: {e}")
            # Try individual inserts as fallback
            for doc in batch:
                try:
                    collection.insert(doc, overwrite=True)
                except Exception as e2:
                    logger.error(f"Error inserting document {doc.get('_key', 'unknown')}: {e2}")

def main():
    # Configuration
    results_dir = os.environ.get("EXP2_RESULTS_DIR", ".")
    embeddings_dir = os.path.join(results_dir, "embeddings")
    db_name = os.environ.get("EXP2_DB_NAME", "information_reconstructionism_exp2_gpu")
    
    # GPU configuration
    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    batch_size = int(os.environ.get("EXP2_BATCH_SIZE", "10000"))
    
    # ArangoDB connection
    arango_host = os.environ.get('ARANGO_HOST', 'http://192.168.1.69:8529')
    username = os.environ.get('ARANGO_USERNAME', 'root')
    password = os.environ.get('ARANGO_PASSWORD', '')
    
    logger.info("=" * 60)
    logger.info("STEP 3: GPU-ACCELERATED DATABASE LOADING")
    logger.info("=" * 60)
    logger.info(f"Embeddings directory: {embeddings_dir}")
    logger.info(f"Database: {db_name}")
    logger.info(f"Using GPU: {gpu_id}")
    logger.info(f"Batch size: {batch_size}")
    
    # Initialize GPU processor
    gpu_processor = GPUBatchProcessor(device_id=gpu_id, batch_size=batch_size)
    
    # Connect to database
    client = ArangoClient(hosts=arango_host)
    db = client.db(db_name, username=username, password=password)
    
    # Ensure collections exist
    if not db.has_collection('papers_exp2'):
        db.create_collection('papers_exp2')
    if not db.has_collection('chunks_exp2'):
        db.create_collection('chunks_exp2')
    
    # Get embedding files
    embedding_files = list(Path(embeddings_dir).glob("*.json"))
    logger.info(f"Found {len(embedding_files)} embedding files to load")
    
    if not embedding_files:
        logger.error("No embedding files found!")
        return 1
    
    # Process files and prepare for batch insertion
    all_paper_docs = []
    all_chunk_docs = []
    
    start_time = datetime.now()
    
    # Process files in parallel with GPU acceleration
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Process files
        futures = []
        for file_path in tqdm(embedding_files, desc="Processing embeddings"):
            future = executor.submit(process_embedding_file, file_path, gpu_processor)
            futures.append(future)
            
            # Process in batches to avoid memory issues
            if len(futures) >= 100:
                for future in futures:
                    result = future.result()
                    if result:
                        all_paper_docs.append(result['paper_doc'])
                        all_chunk_docs.extend(result['chunk_docs'])
                
                # Clear futures
                futures = []
                
                # Clear GPU cache
                torch.cuda.empty_cache()
        
        # Process remaining futures
        for future in futures:
            result = future.result()
            if result:
                all_paper_docs.append(result['paper_doc'])
                all_chunk_docs.extend(result['chunk_docs'])
    
    logger.info(f"\nProcessed {len(all_paper_docs)} papers with {len(all_chunk_docs)} chunks")
    
    # Batch insert into database
    logger.info("\nInserting papers into database...")
    batch_insert_documents(db, 'papers_exp2', all_paper_docs, batch_size=1000)
    
    logger.info("Inserting chunks into database...")
    batch_insert_documents(db, 'chunks_exp2', all_chunk_docs, batch_size=batch_size)
    
    # Verify insertion
    papers_count = db.collection('papers_exp2').count()
    chunks_count = db.collection('chunks_exp2').count()
    
    duration = (datetime.now() - start_time).total_seconds()
    
    logger.info("\n" + "=" * 60)
    logger.info("DATABASE LOADING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Papers in database: {papers_count:,}")
    logger.info(f"Chunks in database: {chunks_count:,}")
    logger.info(f"Average chunks per paper: {chunks_count/papers_count:.1f}")
    logger.info(f"Total time: {duration:.1f} seconds")
    logger.info(f"Loading rate: {chunks_count/duration:.1f} chunks/second")
    
    # GPU memory stats
    allocated = torch.cuda.memory_allocated(gpu_processor.device) / 1024**3
    reserved = torch.cuda.memory_reserved(gpu_processor.device) / 1024**3
    logger.info(f"\nGPU {gpu_processor.device} memory usage:")
    logger.info(f"  Allocated: {allocated:.2f} GB")
    logger.info(f"  Reserved: {reserved:.2f} GB")
    
    # Save summary
    summary = {
        'papers_loaded': papers_count,
        'chunks_loaded': chunks_count,
        'avg_chunks_per_paper': chunks_count / papers_count if papers_count > 0 else 0,
        'duration_seconds': duration,
        'chunks_per_second': chunks_count / duration,
        'gpu_id': gpu_id,
        'batch_size': batch_size,
        'timestamp': datetime.now().isoformat()
    }
    
    summary_path = os.path.join(results_dir, "db_loading_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nSummary saved to: {summary_path}")
    logger.info("\n✓ GPU-ACCELERATED DATABASE LOADING COMPLETE")
    
    return 0

if __name__ == "__main__":
    exit(main())