#!/usr/bin/env python3
"""
Abstract Processing Pipeline with Jina Embeddings and Three-Collection Structure
Processes arXiv metadata/abstracts using the same architecture as PDF pipeline v10
"""

import os
import sys
import json
import time
import queue
import logging
import warnings
import threading
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from tqdm import tqdm
from arango import ArangoClient

# Set GPU visibility BEFORE any imports
def set_worker_gpu(gpu_id: int):
    """Set GPU visibility for worker process"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Import Jina after GPU setup
import torch
from transformers import AutoTokenizer, AutoModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('abstracts_jina_v1.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except:
    NVML_AVAILABLE = False

@dataclass
class Config:
    """Pipeline configuration"""
    # Data
    metadata_dir: str = "/mnt/data-cold/arxiv_data/metadata"
    max_abstracts: Optional[int] = None
    
    # Database
    db_name: str = "arxiv_abstracts_jina"
    db_host: str = "localhost"
    db_port: int = 8529
    db_username: str = "root"
    db_password: str = os.environ.get('ARANGO_PASSWORD', '')
    
    # Processing
    metadata_workers: int = 4
    embedding_workers: int = 2
    metadata_batch_size: int = 100
    embedding_batch_size: int = 32
    metadata_queue_size: int = 1000
    embedding_queue_size: int = 500
    db_queue_size: int = 2000
    
    # No chunking needed for abstracts
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints/abstracts_jina_v1"
    clean_start: bool = False
    resume: bool = True
    
    # GPU
    embedding_gpu: int = 1

# No chunking needed for abstracts - they fit in single embeddings

class CheckpointManager:
    """File-based checkpoint management for multi-process safety"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.processed_file = self.checkpoint_dir / 'processed_ids.txt'
        self._lock = threading.Lock()
        
        # Load existing processed IDs
        self.processed_ids = set()
        if self.processed_file.exists():
            with open(self.processed_file, 'r') as f:
                self.processed_ids = set(line.strip() for line in f if line.strip())
        
    def is_processed(self, arxiv_id: str) -> bool:
        """Check if document was processed"""
        with self._lock:
            return arxiv_id in self.processed_ids
            
    def mark_processed(self, arxiv_id: str):
        """Mark document as processed"""
        with self._lock:
            if arxiv_id not in self.processed_ids:
                self.processed_ids.add(arxiv_id)
                with open(self.processed_file, 'a') as f:
                    f.write(f"{arxiv_id}\n")
            
    def get_processed_count(self) -> int:
        """Get total processed documents"""
        with self._lock:
            return len(self.processed_ids)
            
    def clear(self):
        """Clear all checkpoints"""
        with self._lock:
            self.processed_ids.clear()
            if self.processed_file.exists():
                self.processed_file.unlink()

def metadata_worker_process(
    worker_id: int,
    metadata_files: List[Path],
    metadata_queue: mp.Queue,
    checkpoint_dir: str,
    config: Config
):
    """Process metadata files and extract abstracts"""
    logger = logging.getLogger(f"MetadataWorker-{worker_id}")
    
    batch = []
    for file_path in tqdm(metadata_files, desc=f"Worker {worker_id}", position=worker_id):
        try:
            with open(file_path, 'r') as f:
                metadata = json.load(f)
                
            arxiv_id = metadata.get('arxiv_id', metadata.get('id', ''))
            if not arxiv_id:
                continue
                
            # Skip if already processed
            checkpoint_file = Path(checkpoint_dir) / 'processed' / f"{arxiv_id}.done"
            if config.resume and checkpoint_file.exists():
                continue
                
            abstract = metadata.get('abstract', '').strip()
            if len(abstract) < 50:  # Skip very short abstracts
                continue
                
            # Extract relevant metadata
            doc_metadata = {
                'arxiv_id': arxiv_id,
                'title': metadata.get('title', ''),
                'authors': metadata.get('authors', []),
                'categories': metadata.get('categories', []),
                'published': metadata.get('published', metadata.get('versions', [{}])[0].get('created', '')),
                'abstract': abstract
            }
            
            batch.append(doc_metadata)
            
            if len(batch) >= config.metadata_batch_size:
                metadata_queue.put(batch)
                batch = []
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            
    # Send remaining batch
    if batch:
        metadata_queue.put(batch)

class JinaEmbedder:
    """Jina embeddings with late chunking support"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.model_name = "jinaai/jina-embeddings-v4"
        
        logger.info(f"Loading Jina model on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_name, 
            trust_remote_code=True,
            torch_dtype=torch.float16  # Use FP16 for RTX A6000
        )
        
        if device == 'cuda':
            self.model = self.model.cuda()
        self.model.eval()
        
        logger.info("Jina model loaded successfully")
        
    def embed_text(self, text: str, task: str = 'retrieval') -> np.ndarray:
        """Generate single embedding for text"""
        
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=8192)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, task_label=task)
            
        # Get single-vector embedding (Jina v4 uses task-specific outputs)
        if hasattr(outputs, 'single_vec_emb'):
            # For single document embedding
            embedding = outputs.single_vec_emb[0]
        else:
            # Fallback to mean pooling
            embedding = outputs.last_hidden_state.mean(dim=1)[0]
            
        return embedding.cpu().numpy()

def embedding_worker_process(
    gpu_id: int,
    embedding_queue: mp.Queue,
    db_queue: mp.Queue,
    stop_event: mp.Event,
    checkpoint_dir: str,
    config: Config
):
    """GPU worker for generating embeddings"""
    set_worker_gpu(gpu_id)
    
    logger = logging.getLogger(f"EmbeddingWorker-GPU{gpu_id}")
    logger.info(f"Starting on GPU {gpu_id}")
    
    # Initialize embedder
    embedder = JinaEmbedder(device='cuda')
    
    while not stop_event.is_set():
        try:
            batch = embedding_queue.get(timeout=1.0)
            if batch is None:  # Poison pill
                break
                
            for doc in batch:
                try:
                    # Generate single embedding for abstract
                    abstract = doc['abstract']
                    if not abstract or len(abstract.strip()) < 10:
                        logger.warning(f"Skipping document {doc['arxiv_id']} - abstract too short")
                        continue
                    embedding = embedder.embed_text(abstract, task='retrieval')
                    
                    # Prepare records for database
                    timestamp = datetime.utcnow().isoformat()
                    
                    # Metadata record
                    metadata_record = {
                        '_key': doc['arxiv_id'],
                        'arxiv_id': doc['arxiv_id'],
                        'title': doc['title'],
                        'authors': doc['authors'],
                        'categories': doc['categories'],
                        'published': doc['published'],
                        'processed_at': timestamp
                    }
                    
                    # Embedding record
                    embedding_record = {
                        '_key': doc['arxiv_id'],
                        'arxiv_id': doc['arxiv_id'],
                        'embedding': embedding.tolist(),
                        'processed_at': timestamp
                    }
                    
                    # Abstract record (markdown format)
                    abstract_record = {
                        '_key': doc['arxiv_id'],
                        'arxiv_id': doc['arxiv_id'],
                        'abstract_text': abstract,
                        'abstract_markdown': f"## Abstract\n\n{abstract}",
                        'processed_at': timestamp
                    }
                    
                    # Send to database queue
                    db_queue.put({
                        'type': 'metadata',
                        'record': metadata_record
                    })
                    db_queue.put({
                        'type': 'embeddings',
                        'record': embedding_record
                    })
                    db_queue.put({
                        'type': 'abstracts',
                        'record': abstract_record
                    })
                    
                    # Mark as processed in checkpoint
                    checkpoint_file = Path(checkpoint_dir) / 'processed' / f"{doc['arxiv_id']}.done"
                    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                    checkpoint_file.touch()
                        
                except Exception as e:
                    logger.error(f"Error processing document {doc.get('arxiv_id', 'unknown')}: {e}")
                    
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Worker error: {e}")

class DatabaseWriter(threading.Thread):
    """Database writer thread"""
    
    def __init__(self, db_queue: queue.Queue, config: Config):
        super().__init__()
        self.db_queue = db_queue
        self.config = config
        self.stop_event = threading.Event()
        
        # Batch buffers for each collection
        self.batch_buffers = {
            'metadata': [],
            'embeddings': [],
            'abstracts': []
        }
        self.batch_size = 100
        
        self._initialize_db()
        
    def _initialize_db(self):
        """Initialize database connection"""
        client = ArangoClient(hosts=f'http://{self.config.db_host}:{self.config.db_port}')
        self.db = client.db(
            self.config.db_name, 
            username=self.config.db_username, 
            password=self.config.db_password
        )
        
        # Get collections
        self.collections = {
            'metadata': self.db.collection('metadata'),
            'embeddings': self.db.collection('embeddings'),
            'abstracts': self.db.collection('abstracts')
        }
        
    def run(self):
        """Main database writer loop"""
        while not self.stop_event.is_set() or not self.db_queue.empty():
            try:
                item = self.db_queue.get(timeout=1.0)
                
                collection_type = item['type']
                record = item['record']
                
                self.batch_buffers[collection_type].append(record)
                
                # Flush if batch is full
                if len(self.batch_buffers[collection_type]) >= self.batch_size:
                    self._flush_batch(collection_type)
                    
            except queue.Empty:
                # Flush any remaining records
                for collection_type in self.batch_buffers:
                    if self.batch_buffers[collection_type]:
                        self._flush_batch(collection_type)
                        
            except Exception as e:
                logger.error(f"Database writer error: {e}")
                
    def _flush_batch(self, collection_type: str):
        """Flush a batch to database"""
        if not self.batch_buffers[collection_type]:
            return
            
        try:
            collection = self.collections[collection_type]
            collection.insert_many(self.batch_buffers[collection_type], overwrite=True)
            logger.debug(f"Flushed {len(self.batch_buffers[collection_type])} records to {collection_type}")
            self.batch_buffers[collection_type] = []
        except Exception as e:
            logger.error(f"Error flushing {collection_type}: {e}")
            
    def stop(self):
        """Stop the writer"""
        self.stop_event.set()
        
        # Final flush
        for collection_type in self.batch_buffers:
            if self.batch_buffers[collection_type]:
                self._flush_batch(collection_type)

class AbstractPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config: Config):
        self.config = config
        
        if config.clean_start:
            logger.info("Clean start - clearing checkpoints")
            import shutil
            checkpoint_path = Path(config.checkpoint_dir)
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            
    def setup_database(self) -> bool:
        """Setup database and collections"""
        try:
            client = ArangoClient(hosts=f'http://{self.config.db_host}:{self.config.db_port}')
            sys_db = client.db('_system', username=self.config.db_username, password=self.config.db_password)
            
            if not sys_db.has_database(self.config.db_name):
                sys_db.create_database(self.config.db_name)
                logger.info(f"Created database: {self.config.db_name}")
            
            db = client.db(self.config.db_name, username=self.config.db_username, password=self.config.db_password)
            
            # Create all three collections
            collections_to_create = ['metadata', 'embeddings', 'abstracts']
            
            for collection_name in collections_to_create:
                if not db.has_collection(collection_name):
                    collection = db.create_collection(collection_name)
                    logger.info(f"Created collection: {collection_name}")
                    
                    # Add indexes
                    if collection_name == 'metadata':
                        collection.add_hash_index(fields=['arxiv_id'], unique=True)
                        collection.add_persistent_index(fields=['categories[*]'])
                        collection.add_persistent_index(fields=['published'])
                    elif collection_name == 'embeddings':
                        collection.add_hash_index(fields=['arxiv_id'], unique=True)
                    elif collection_name == 'abstracts':
                        collection.add_hash_index(fields=['arxiv_id'], unique=True)
                        
            return True
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            return False
            
    def run(self):
        """Run the pipeline"""
        logger.info("Starting Abstract Processing Pipeline with Jina Embeddings")
        logger.info(f"Configuration: {self.config}")
        
        # Setup database
        if not self.setup_database():
            logger.error("Failed to setup database")
            return
            
        # Get metadata files
        metadata_dir = Path(self.config.metadata_dir)
        all_files = list(metadata_dir.glob("*.json"))
        
        if self.config.max_abstracts:
            all_files = all_files[:self.config.max_abstracts]
            
        logger.info(f"Found {len(all_files)} metadata files")
        
        # Create queues
        metadata_queue = mp.Queue(maxsize=self.config.metadata_queue_size)
        embedding_queue = mp.Queue(maxsize=self.config.embedding_queue_size)
        db_queue = mp.Queue(maxsize=self.config.db_queue_size)
        
        # Start database writer
        db_writer = DatabaseWriter(db_queue, self.config)
        db_writer.start()
        
        # Split files among metadata workers
        files_per_worker = len(all_files) // self.config.metadata_workers
        
        # Start metadata workers
        metadata_processes = []
        for i in range(self.config.metadata_workers):
            start_idx = i * files_per_worker
            end_idx = start_idx + files_per_worker if i < self.config.metadata_workers - 1 else len(all_files)
            worker_files = all_files[start_idx:end_idx]
            
            p = mp.Process(
                target=metadata_worker_process,
                args=(i, worker_files, metadata_queue, self.config.checkpoint_dir, self.config)
            )
            p.start()
            metadata_processes.append(p)
            
        # Start embedding workers
        stop_event = mp.Event()
        embedding_processes = []
        
        for i in range(self.config.embedding_workers):
            p = mp.Process(
                target=embedding_worker_process,
                args=(self.config.embedding_gpu, embedding_queue, db_queue, stop_event, self.config.checkpoint_dir, self.config)
            )
            p.start()
            embedding_processes.append(p)
            
        # Transfer thread between queues
        def transfer_batches():
            while any(p.is_alive() for p in metadata_processes) or not metadata_queue.empty():
                try:
                    batch = metadata_queue.get(timeout=1.0)
                    embedding_queue.put(batch)
                except queue.Empty:
                    continue
                    
            # Send poison pills to embedding workers
            for _ in embedding_processes:
                embedding_queue.put(None)
                
        transfer_thread = threading.Thread(target=transfer_batches)
        transfer_thread.start()
        
        # Monitor progress
        start_time = time.time()
        last_count = 0
        
        while any(p.is_alive() for p in metadata_processes + embedding_processes):
            time.sleep(10)
            
            # Count processed files
            processed_dir = Path(self.config.checkpoint_dir) / 'processed'
            current_count = len(list(processed_dir.glob('*.done'))) if processed_dir.exists() else 0
            rate = (current_count - last_count) / 10.0
            total_rate = current_count / (time.time() - start_time)
            
            logger.info(
                f"Processed: {current_count}/{len(all_files)} | "
                f"Rate: {rate:.1f} docs/s | "
                f"Avg: {total_rate:.1f} docs/s | "
                f"Queues: meta={metadata_queue.qsize()}, "
                f"embed={embedding_queue.qsize()}, "
                f"db={db_queue.qsize()}"
            )
            
            last_count = current_count
            
        # Wait for all processes
        for p in metadata_processes:
            p.join()
            
        transfer_thread.join()
        stop_event.set()
        
        for p in embedding_processes:
            p.join()
            
        # Stop database writer
        db_writer.stop()
        db_writer.join()
        
        elapsed = time.time() - start_time
        # Final count
        processed_dir = Path(self.config.checkpoint_dir) / 'processed'
        final_count = len(list(processed_dir.glob('*.done'))) if processed_dir.exists() else 0
        
        logger.info(f"Pipeline complete!")
        logger.info(f"Processed {final_count} abstracts in {elapsed/60:.1f} minutes")
        logger.info(f"Average rate: {final_count/elapsed:.1f} abstracts/second")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Process arXiv abstracts with Jina embeddings")
    parser.add_argument('--metadata-dir', type=str, default='/mnt/data-cold/arxiv_data/metadata')
    parser.add_argument('--max-abstracts', type=int, help='Maximum abstracts to process')
    parser.add_argument('--db-name', type=str, default='arxiv_abstracts_jina')
    parser.add_argument('--db-host', type=str, default='localhost')
    parser.add_argument('--metadata-workers', type=int, default=4)
    parser.add_argument('--embedding-workers', type=int, default=2)
    parser.add_argument('--embedding-gpu', type=int, default=1)
    parser.add_argument('--clean-start', action='store_true')
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    
    args = parser.parse_args()
    
    config = Config(
        metadata_dir=args.metadata_dir,
        max_abstracts=args.max_abstracts,
        db_name=args.db_name,
        db_host=args.db_host,
        metadata_workers=args.metadata_workers,
        embedding_workers=args.embedding_workers,
        embedding_gpu=args.embedding_gpu,
        clean_start=args.clean_start,
        resume=args.resume
    )
    
    pipeline = AbstractPipeline(config)
    pipeline.run()

if __name__ == '__main__':
    main()