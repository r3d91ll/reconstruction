#!/usr/bin/env python3
"""
Single Collection Abstract Processing Pipeline
Phase 1 MVP: Consolidates 3 collections into 1 for improved performance
Target: 25-30 documents/second (40-65% improvement)
"""

import os
import sys

# Set environment variable before any imports
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json
import time
import queue
import logging
import warnings
import threading
import multiprocessing as mp
import lmdb
import psutil
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from tqdm import tqdm
from arango import ArangoClient
from arango.exceptions import DocumentInsertError

# Import torch and transformers after setting environment
import torch
from transformers import AutoTokenizer, AutoModel

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('abstracts_single_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Pipeline configuration for single collection approach"""
    # Data
    metadata_dir: str = "/mnt/data-cold/arxiv_data/metadata"
    max_abstracts: Optional[int] = None
    
    # Database
    db_name: str = "arxiv_single_collection"
    collection_name: str = "arxiv_documents"
    db_host: str = "192.168.1.69"
    db_port: int = 8529
    db_username: str = "root"
    db_password: str = os.environ.get('ARANGO_PASSWORD', '')
    
    # Processing
    metadata_workers: int = 4
    embedding_workers: int = 1
    batch_size: int = 100  # Small batches for better GPU memory management
    write_buffer_size: int = 128 * 1024 * 1024  # 128MB
    commit_frequency: int = 10000  # Commit every 10K documents
    
    # Queues (7 batches * 100 docs = 700 texts max in pipeline)
    metadata_queue_size: int = 7  # Allow up to 7 batches to be prepared
    embedding_queue_size: int = 7  # Allow up to 7 batches in GPU pipeline
    db_queue_size: int = 7  # Match the pipeline depth
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints/single_collection"
    clean_start: bool = False
    resume: bool = True
    
    # GPU
    embedding_gpu: int = 0
    
    # Performance tracking
    track_metrics: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters"""
        # Validate worker counts
        if self.metadata_workers <= 0:
            raise ValueError("metadata_workers must be positive")
        if self.embedding_workers <= 0:
            raise ValueError("embedding_workers must be positive")
            
        # Validate batch and queue sizes
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.metadata_queue_size <= 0:
            raise ValueError("metadata_queue_size must be positive")
        if self.embedding_queue_size <= 0:
            raise ValueError("embedding_queue_size must be positive")
        if self.db_queue_size <= 0:
            raise ValueError("db_queue_size must be positive")
            
        # Validate database connection parameters
        if not self.db_host:
            raise ValueError("db_host cannot be empty")
        if self.db_port <= 0 or self.db_port > 65535:
            raise ValueError("db_port must be between 1 and 65535")
        if not self.db_username:
            raise ValueError("db_username cannot be empty")
        if not self.db_password:
            raise ValueError("ARANGO_PASSWORD environment variable not set")
            
        # Validate GPU ID
        if self.embedding_gpu < 0:
            raise ValueError("embedding_gpu must be non-negative")
            
        # Validate file sizes
        if self.write_buffer_size <= 0:
            raise ValueError("write_buffer_size must be positive")
        if self.commit_frequency <= 0:
            raise ValueError("commit_frequency must be positive")

class ValidationMetrics:
    """Minimal metrics to validate collection consolidation benefits"""
    
    def __init__(self):
        self.start_time = time.time()
        self.checkpoint_times = []  # Track 10K document intervals
        self.checkpoint_counter = 0   # Efficient modulo replacement
        self.write_operations = 0    # Count actual DB writes
        self.documents_processed = 0
        self.errors = []             # Track failures
        self.batch_times = []        # Track batch processing times
        
    def record_batch(self, batch_size: int, batch_time: float, success: bool = True):
        """Record batch processing results"""
        if success:
            self.documents_processed += batch_size
            self.checkpoint_counter += batch_size
            self.batch_times.append(batch_time)
            
            # Check if we've hit 10K threshold
            if self.checkpoint_counter >= 10000:
                self.checkpoint_counter -= 10000  # Preserve remainder
                self.checkpoint()
        else:
            self.errors.append({
                'timestamp': time.time(),
                'batch_size': batch_size,
                'documents_processed': self.documents_processed
            })
        
    def checkpoint(self):
        """Called every 10,000 documents"""
        current_time = time.time()
        if self.checkpoint_times:
            interval_time = current_time - self.checkpoint_times[-1]
            rate = 10000 / interval_time
            elapsed = current_time - self.start_time
            logger.info(f"[{elapsed:.0f}s] Last 10K docs: {rate:.1f} docs/sec")
        self.checkpoint_times.append(current_time)
        
    def record_write_operation(self):
        """Record a database write operation"""
        self.write_operations += 1
        
    def final_report(self):
        """Generate comparison metrics with error summary"""
        total_time = time.time() - self.start_time
        overall_rate = self.documents_processed / total_time if total_time > 0 else 0
        writes_per_doc = self.write_operations / self.documents_processed if self.documents_processed > 0 else 0
        
        print(f"\nFinal metrics:")
        print(f"  Overall rate: {overall_rate:.1f} docs/sec (target: 25-30)")
        print(f"  Writes per doc: {writes_per_doc:.2f} (target: 1.0)")
        print(f"  Processing time: {total_time/3600:.2f} hours")
        print(f"  Error count: {len(self.errors)}")
        
        if self.errors:
            print(f"  First error at: {self.errors[0]['documents_processed']} documents")
            
        # Calculate percentiles for batch times
        if self.batch_times:
            sorted_times = sorted(self.batch_times)
            p50 = sorted_times[len(sorted_times)//2]
            p95 = sorted_times[int(len(sorted_times)*0.95)]
            p99 = sorted_times[int(len(sorted_times)*0.99)]
            
            print(f"\nBatch processing times:")
            print(f"  P50: {p50:.2f}s")
            print(f"  P95: {p95:.2f}s")
            print(f"  P99: {p99:.2f}s")

class CheckpointManager:
    """LMDB-based checkpoint management with file fallback"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize LMDB
        self.lmdb_path = self.checkpoint_dir / 'checkpoints.lmdb'
        self.lmdb_env = None
        self._lock = threading.Lock()
        
        try:
            self.lmdb_env = lmdb.open(
                str(self.lmdb_path),
                map_size=100 * 1024 * 1024,  # 100MB
                max_dbs=1
            )
            logger.info("LMDB checkpoint system initialized")
        except Exception as e:
            logger.warning(f"LMDB initialization failed: {e}, using file fallback")
            
        # Fallback file
        self.fallback_file = self.checkpoint_dir / 'checkpoint_backup.json'
        self.processed_ids = set()
        self._load_checkpoint()
        
    def _load_checkpoint(self):
        """Load existing checkpoint data"""
        if self.lmdb_env:
            try:
                with self.lmdb_env.begin() as txn:
                    cursor = txn.cursor()
                    for key, _ in cursor:
                        self.processed_ids.add(key.decode())
                logger.info(f"Loaded {len(self.processed_ids)} processed IDs from LMDB")
                return
            except Exception as e:
                logger.error(f"Failed to load from LMDB: {e}")
                
        # Fallback to file
        if self.fallback_file.exists():
            try:
                with open(self.fallback_file, 'r') as f:
                    data = json.load(f)
                    self.processed_ids = set(data.get('processed_ids', []))
                logger.info(f"Loaded {len(self.processed_ids)} processed IDs from fallback file")
            except Exception as e:
                logger.error(f"Failed to load fallback file: {e}")
                
    def save_checkpoint(self, doc_count: int, last_key: str):
        """Save checkpoint with current progress"""
        checkpoint_data = {
            'documents_processed': doc_count,
            'last_document_key': last_key,
            'timestamp': time.time()
        }
        
        with self._lock:
            if self.lmdb_env:
                try:
                    with self.lmdb_env.begin(write=True) as txn:
                        txn.put(b'progress', json.dumps(checkpoint_data).encode())
                    return
                except lmdb.Error as e:
                    logger.error(f"LMDB write error: {e}, using file backup")
                    
            # Fallback to atomic file write
            temp_file = self.fallback_file + '.tmp'
            try:
                with open(temp_file, 'w') as f:
                    json.dump({
                        'checkpoint': checkpoint_data,
                        'processed_ids': list(self.processed_ids)
                    }, f)
                # Atomically rename to final file
                os.replace(temp_file, self.fallback_file)
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")
                # Clean up temp file if it exists
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                
    def is_processed(self, arxiv_id: str) -> bool:
        """Check if document was processed"""
        with self._lock:
            return arxiv_id in self.processed_ids
            
    def mark_processed(self, arxiv_id: str):
        """Mark document as processed"""
        with self._lock:
            if arxiv_id not in self.processed_ids:
                self.processed_ids.add(arxiv_id)
                
                if self.lmdb_env:
                    try:
                        with self.lmdb_env.begin(write=True) as txn:
                            txn.put(arxiv_id.encode(), b'1')
                    except Exception as e:
                        logger.error(f"Failed to mark {arxiv_id} in LMDB: {e}")

class ErrorRecovery:
    """Batch failure recovery with exponential backoff"""
    
    def __init__(self):
        self.max_retries = 3
        self.base_delay = 1.0  # seconds
        
    def retry_batch(self, batch: List[Dict], collection, attempt: int = 0) -> bool:
        """Retry batch with exponential backoff"""
        if attempt >= self.max_retries:
            # Write to dead letter queue
            self.save_failed_batch(batch)
            return False
            
        try:
            # Retry logic with exponential backoff
            delay = self.base_delay * (2 ** attempt)
            time.sleep(delay)
            
            # Attempt write
            collection.insert_many(batch, overwrite=True)
            return True
            
        except DocumentInsertError as e:
            # Handle specific document errors
            if hasattr(e, 'error_list'):
                # Filter out documents that already exist
                failed_indices = {err.get('index') for err in e.error_list if err.get('error')}
                retry_batch = [doc for i, doc in enumerate(batch) if i in failed_indices]
                
                if retry_batch:
                    return self.retry_batch(retry_batch, collection, attempt + 1)
                else:
                    # All documents already exist
                    return True
                    
            return self.retry_batch(batch, collection, attempt + 1)
            
        except Exception as e:
            # Other errors - log and retry
            logger.error(f"Retry {attempt}: {e}")
            return self.retry_batch(batch, collection, attempt + 1)
            
    def save_failed_batch(self, batch: List[Dict]):
        """Save failed batch with context for manual inspection"""
        failed_path = Path('failed_batches')
        failed_path.mkdir(exist_ok=True)
        
        timestamp = datetime.utcnow()
        timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S')
        filename = failed_path / f'failed_batch_{timestamp_str}.json'
        
        # Gather debugging context
        context = {
            'timestamp': timestamp.isoformat(),
            'timestamp_unix': timestamp.timestamp(),
            'batch_size': len(batch),
            'retry_attempts': self.max_retries,
            'documents': batch,
            'document_ids': [doc.get('_key', 'unknown') for doc in batch],
            'first_doc_sample': batch[0] if batch else None,
            'system_info': {
                'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown',
                'python_version': sys.version,
                'process_id': os.getpid()
            }
        }
        
        # Add document statistics
        if batch:
            context['stats'] = {
                'total_documents': len(batch),
                'has_embeddings': sum(1 for doc in batch if 'abstract_embedding' in doc),
                'categories': list(set(cat for doc in batch for cat in doc.get('categories', []))),
                'date_range': {
                    'earliest': min((doc.get('submitted_date', '') for doc in batch), default=''),
                    'latest': max((doc.get('submitted_date', '') for doc in batch), default='')
                }
            }
        
        with open(filename, 'w') as f:
            json.dump(context, f, indent=2)
            
        logger.error(f"Saved failed batch to {filename} (size: {len(batch)} documents)")

class JinaEmbedder:
    """Jina embeddings optimized for single GPU"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.model_name = "jinaai/jina-embeddings-v3"
        
        logger.info(f"Loading Jina model on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_name, 
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # Ensure model is on the correct device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info("Jina model loaded successfully")
        
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts"""
        try:
            inputs = self.tokenizer(
                texts, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=8192
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings_np = embeddings.cpu().numpy()
            
            # Clean up GPU memory immediately
            del outputs, embeddings, inputs
            torch.cuda.empty_cache()
            
            return embeddings_np
            
        except torch.cuda.OutOfMemoryError:
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            # Log the error
            logger.error(f"GPU OOM error while processing batch of {len(texts)} texts")
            
            # Try processing in smaller sub-batches with adaptive sizing
            if len(texts) > 1:
                # Try progressively smaller batch sizes: 75%, 50%, 25%, 10%
                for fraction in [0.75, 0.5, 0.25, 0.1]:
                    new_size = max(1, int(len(texts) * fraction))
                    if new_size < len(texts):
                        logger.info(f"Retrying with batch size {new_size}")
                        try:
                            # Process in chunks of new_size
                            embeddings_list = []
                            for i in range(0, len(texts), new_size):
                                chunk = texts[i:i+new_size]
                                chunk_embeddings = self.embed_batch(chunk)
                                embeddings_list.append(chunk_embeddings)
                            return np.vstack(embeddings_list)
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            continue
                
                # If all else fails, process one by one
                logger.warning("Processing texts individually due to GPU memory constraints")
                embeddings_list = []
                for text in texts:
                    emb = self.embed_batch([text])
                    embeddings_list.append(emb)
                return np.vstack(embeddings_list)
            else:
                # Single text still causes OOM
                logger.error("Cannot process even single text due to GPU memory constraints")
                raise

def metadata_worker_process(
    worker_id: int,
    metadata_files: List[Path],
    metadata_queue: mp.Queue,
    checkpoint_manager_path: str,  # Pass path instead of object
    config: Config
):
    """Process metadata files and prepare documents"""
    logger = logging.getLogger(f"MetadataWorker-{worker_id}")
    
    # Create local checkpoint manager instance
    checkpoint_manager = CheckpointManager(checkpoint_manager_path)
    
    batch = []
    for file_path in tqdm(metadata_files, desc=f"Worker {worker_id}", position=worker_id):
        try:
            with open(file_path, 'r') as f:
                metadata = json.load(f)
                
            arxiv_id = metadata.get('arxiv_id', metadata.get('id', ''))
            if not arxiv_id:
                continue
                
            # Skip if already processed
            if config.resume and checkpoint_manager.is_processed(arxiv_id):
                continue
                
            abstract = metadata.get('abstract', '').strip()
            if len(abstract) < 50:  # Skip very short abstracts
                continue
                
            # Create single document with all fields
            document = {
                '_key': arxiv_id,
                'title': metadata.get('title', ''),
                'authors': metadata.get('authors', []),
                'categories': metadata.get('categories', []),
                'abstract': abstract,
                'submitted_date': metadata.get('published', metadata.get('versions', [{}])[0].get('created', '')),
                'updated_date': metadata.get('updated', ''),
                'pdf_status': {
                    'state': 'unprocessed',
                    'tar_source': None,
                    'last_updated': None,
                    'retry_count': 0,
                    'error_message': None
                }
            }
            
            batch.append(document)
            
            if len(batch) >= config.batch_size:
                metadata_queue.put(batch)
                batch = []
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            
    # Send remaining batch
    if batch:
        metadata_queue.put(batch)

def embedding_worker_process(
    gpu_id: int,
    embedding_queue: mp.Queue,
    db_queue: mp.Queue,
    stop_event: mp.Event,
    config: Config
):
    """GPU worker for generating embeddings"""
    # Set the GPU device for this process
    torch.cuda.set_device(gpu_id)
    
    logger = logging.getLogger(f"EmbeddingWorker-GPU{gpu_id}")
    logger.info(f"Starting on GPU {gpu_id}")
    
    # Initialize embedder with specific device
    embedder = JinaEmbedder(device=f'cuda:{gpu_id}')
    
    while not stop_event.is_set():
        try:
            batch = embedding_queue.get(timeout=1.0)
            if batch is None:  # Poison pill
                break
                
            # Extract abstracts for batch embedding
            abstracts = [doc['abstract'] for doc in batch]
            
            # Generate embeddings for entire batch
            embeddings = embedder.embed_batch(abstracts)
            
            # Add embeddings to documents
            for i, doc in enumerate(batch):
                doc['abstract_embedding'] = embeddings[i].tolist()
                
            # Send to database queue
            db_queue.put(batch)
            
            # Clear GPU memory after each batch to prevent accumulation
            del embeddings
            del abstracts
            torch.cuda.empty_cache()
                
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Worker error: {e}")
            torch.cuda.empty_cache()  # Also clear on errors

class DatabaseWriter(threading.Thread):
    """Optimized database writer for single collection"""
    
    def __init__(self, db_queue: queue.Queue, config: Config, metrics: ValidationMetrics):
        super().__init__()
        self.db_queue = db_queue
        self.config = config
        self.metrics = metrics
        self.stop_event = threading.Event()
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir)
        self.error_recovery = ErrorRecovery()
        
        self._initialize_db()
        
    def _initialize_db(self):
        """Initialize database connection with optimized settings"""
        client = ArangoClient(hosts=f'http://{self.config.db_host}:{self.config.db_port}')
        
        # Get system database
        sys_db = client.db('_system', username=self.config.db_username, password=self.config.db_password)
        
        # Create database if needed
        if not sys_db.has_database(self.config.db_name):
            sys_db.create_database(self.config.db_name)
            logger.info(f"Created database: {self.config.db_name}")
            
        # Connect to database
        self.db = client.db(
            self.config.db_name, 
            username=self.config.db_username, 
            password=self.config.db_password
        )
        
        # Create collection if needed
        if not self.db.has_collection(self.config.collection_name):
            collection = self.db.create_collection(self.config.collection_name)
            logger.info(f"Created collection: {self.config.collection_name}")
        else:
            collection = self.db.collection(self.config.collection_name)
            
        # Add indexes if they don't exist
        existing_indexes = {idx['fields'][0] if idx.get('fields') else None for idx in collection.indexes()}
        
        # Check and add indexes only if needed
        if 'categories[*]' not in existing_indexes:
            collection.add_hash_index(fields=['categories[*]'])
            logger.info("Added hash index on categories")
            
        if 'submitted_date' not in existing_indexes:
            collection.add_persistent_index(fields=['submitted_date'])
            logger.info("Added persistent index on submitted_date")
            
        if 'pdf_status.state' not in existing_indexes:
            collection.add_persistent_index(fields=['pdf_status.state'])
            logger.info("Added persistent index on pdf_status.state")
        self.collection = collection
        
    def run(self):
        """Main database writer loop"""
        buffer = []
        last_commit_time = time.time()
        documents_since_commit = 0
        
        while not self.stop_event.is_set() or not self.db_queue.empty():
            try:
                # Get batch with timeout
                batch = self.db_queue.get(timeout=1.0)
                buffer.extend(batch)
                
                # Write when buffer is full
                if len(buffer) >= self.config.batch_size:
                    self._write_buffer(buffer)
                    buffer = []
                    
                # Periodic commit based on document count
                documents_since_commit += len(batch)
                if documents_since_commit >= self.config.commit_frequency:
                    self._commit()
                    documents_since_commit = 0
                    last_commit_time = time.time()
                    
            except queue.Empty:
                # Write any remaining buffer
                if buffer:
                    self._write_buffer(buffer)
                    buffer = []
                    
                # Commit if it's been too long
                if time.time() - last_commit_time > 60:  # 1 minute
                    self._commit()
                    last_commit_time = time.time()
                    
            except Exception as e:
                logger.error(f"Database writer error: {e}")
                
        # Final flush
        if buffer:
            self._write_buffer(buffer)
        self._commit()
            
    def _write_buffer(self, buffer: List[Dict]):
        """Write buffer to database with error recovery"""
        if not buffer:
            return
            
        start_time = time.time()
        
        try:
            # Single write operation
            self.collection.insert_many(buffer, overwrite=True)
            
            # Update metrics
            self.metrics.record_write_operation()
            batch_time = time.time() - start_time
            self.metrics.record_batch(len(buffer), batch_time, success=True)
            
            # Update checkpoints
            for doc in buffer:
                self.checkpoint_manager.mark_processed(doc['_key'])
                
            logger.debug(f"Wrote {len(buffer)} documents in {batch_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Batch write failed: {e}")
            
            # Try error recovery
            if self.error_recovery.retry_batch(buffer, self.collection):
                self.metrics.record_write_operation()
                batch_time = time.time() - start_time
                self.metrics.record_batch(len(buffer), batch_time, success=True)
            else:
                self.metrics.record_batch(len(buffer), 0, success=False)
                
    def _commit(self):
        """Commit checkpoint"""
        try:
            doc_count = self.metrics.documents_processed
            last_key = list(self.checkpoint_manager.processed_ids)[-1] if self.checkpoint_manager.processed_ids else ""
            self.checkpoint_manager.save_checkpoint(doc_count, last_key)
            logger.debug("Checkpoint committed")
        except Exception as e:
            logger.error(f"Failed to commit checkpoint: {e}")
            
    def stop(self):
        """Stop the writer"""
        self.stop_event.set()

def monitor_resources():
    """Monitor system resources"""
    return {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_mb': psutil.Process().memory_info().rss / 1024 / 1024,
        'io_wait': psutil.cpu_times().iowait if hasattr(psutil.cpu_times(), 'iowait') else 0,
    }

class SingleCollectionPipeline:
    """Main pipeline orchestrator for single collection approach"""
    
    def __init__(self, config: Config):
        self.config = config
        self.metrics = ValidationMetrics() if config.track_metrics else None
        
        if config.clean_start:
            logger.info("Clean start - clearing checkpoints")
            import shutil
            checkpoint_path = Path(config.checkpoint_dir)
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            
    def run(self):
        """Run the pipeline"""
        logger.info("Starting Single Collection Abstract Processing Pipeline")
        logger.info(f"Target: 25-30 documents/second")
        logger.info(f"Configuration: {self.config}")
        
        # Get metadata files
        metadata_dir = Path(self.config.metadata_dir)
        all_files = list(metadata_dir.glob("*.json"))
        
        if self.config.max_abstracts:
            all_files = all_files[:self.config.max_abstracts]
            
        logger.info(f"Found {len(all_files)} metadata files to process")
        
        # Create queues with smaller sizes for better flow control
        metadata_queue = mp.Queue(maxsize=self.config.metadata_queue_size)
        embedding_queue = mp.Queue(maxsize=self.config.embedding_queue_size)
        db_queue = mp.Queue(maxsize=self.config.db_queue_size)
        
        # Start database writer first
        db_writer = DatabaseWriter(db_queue, self.config, self.metrics)
        db_writer.start()
        
        # Initialize checkpoint manager for main process
        checkpoint_manager = CheckpointManager(self.config.checkpoint_dir)
        
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
                args=(self.config.embedding_gpu, embedding_queue, db_queue, stop_event, self.config)
            )
            p.start()
            embedding_processes.append(p)
            
        # Transfer thread between queues with flow control
        def transfer_batches():
            while any(p.is_alive() for p in metadata_processes) or not metadata_queue.empty():
                try:
                    batch = metadata_queue.get(timeout=1.0)
                    
                    # Flow control: wait if embedding queue is getting full
                    # This creates backpressure to slow down metadata workers
                    embedding_queue.put(batch, block=True)
                    
                    # Log coal shoveling progress
                    if embedding_queue.qsize() == embedding_queue._maxsize:
                        logger.debug(f"GPU furnace full! {embedding_queue.qsize()} batches staged")
                    
                except queue.Empty:
                    continue
                    
            # Send poison pills to embedding workers
            for _ in embedding_processes:
                embedding_queue.put(None)
                
        transfer_thread = threading.Thread(target=transfer_batches)
        transfer_thread.start()
        
        # Monitor progress
        start_time = time.time()
        last_resource_check = time.time()
        
        try:
            while any(p.is_alive() for p in metadata_processes + embedding_processes):
                time.sleep(10)
                
                # Resource monitoring
                if time.time() - last_resource_check > 30:
                    resources = monitor_resources()
                    logger.info(
                        f"Resources: CPU={resources['cpu_percent']:.1f}%, "
                        f"Memory={resources['memory_mb']:.0f}MB, "
                        f"IO_wait={resources['io_wait']:.1f}"
                    )
                    last_resource_check = time.time()
                
                # Progress update
                if self.metrics:
                    elapsed = time.time() - start_time
                    rate = self.metrics.documents_processed / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"Progress: {self.metrics.documents_processed}/{len(all_files)} | "
                        f"Rate: {rate:.1f} docs/s | "
                        f"Writes: {self.metrics.write_operations} | "
                        f"Queues: meta={metadata_queue.qsize()}, "
                        f"embed={embedding_queue.qsize()}, "
                        f"db={db_queue.qsize()}"
                    )
                    
        except KeyboardInterrupt:
            logger.warning("Caught keyboard interrupt - attempting graceful shutdown")
            
            # Stop all processes
            stop_event.set()
            
            # Send poison pills to queues
            for _ in range(self.config.metadata_workers):
                try:
                    metadata_queue.put(None, timeout=1.0)
                except:
                    pass
                    
            for _ in range(self.config.embedding_workers):
                try:
                    embedding_queue.put(None, timeout=1.0)
                except:
                    pass
                    
            logger.info("Waiting for workers to finish current batches...")
            
            # Give processes time to finish current work
            timeout = 30  # 30 seconds timeout
            start_wait = time.time()
            while any(p.is_alive() for p in metadata_processes + embedding_processes) and time.time() - start_wait < timeout:
                time.sleep(1)
                
            # Force terminate if still running
            for p in metadata_processes + embedding_processes:
                if p.is_alive():
                    logger.warning(f"Force terminating process {p.pid}")
                    p.terminate()
                    
            raise  # Re-raise to ensure proper cleanup
            
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
        
        # Final report
        if self.metrics:
            self.metrics.final_report()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Single Collection Abstract Processing Pipeline")
    parser.add_argument('--metadata-dir', type=str, default='/mnt/data-cold/arxiv_data/metadata')
    parser.add_argument('--max-abstracts', type=int, help='Maximum abstracts to process (for testing)')
    parser.add_argument('--db-name', type=str, default='arxiv_single_collection')
    parser.add_argument('--collection-name', type=str, default='arxiv_documents')
    parser.add_argument('--db-host', type=str, default='192.168.1.69')
    parser.add_argument('--metadata-workers', type=int, default=4)
    parser.add_argument('--embedding-workers', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--embedding-gpu', type=int, default=0)
    parser.add_argument('--clean-start', action='store_true')
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    parser.add_argument('--no-metrics', dest='track_metrics', action='store_false')
    
    args = parser.parse_args()
    
    config = Config(
        metadata_dir=args.metadata_dir,
        max_abstracts=args.max_abstracts,
        db_name=args.db_name,
        collection_name=args.collection_name,
        db_host=args.db_host,
        metadata_workers=args.metadata_workers,
        embedding_workers=args.embedding_workers,
        batch_size=args.batch_size,
        embedding_gpu=args.embedding_gpu,
        clean_start=args.clean_start,
        resume=args.resume,
        track_metrics=args.track_metrics
    )
    
    pipeline = SingleCollectionPipeline(config)
    pipeline.run()

if __name__ == '__main__':
    main()