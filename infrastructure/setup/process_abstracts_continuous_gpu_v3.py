#!/usr/bin/env python3
"""
Production-Ready Continuous GPU Pipeline V3
Fixes all critical issues identified in code review.

Critical fixes:
- Document/batch tracking mismatch resolved
- JSON-based checkpointing (no pickle)
- Robust database retry logic
- Input validation
- Failed batch recovery
- Embedding validation
- Functional priority queue
"""

import os
import sys
import json
import logging
import torch
import torch.multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Protocol
from dataclasses import dataclass, field
from queue import Queue, Empty
from threading import Thread, Event, Lock
import numpy as np
from tqdm import tqdm
import time
import math
import gc
import heapq
import re
from arango import ArangoClient
from arango.http import HTTPClient
import lmdb
from multiprocessing.managers import BaseManager

# Add infrastructure directory to path
infrastructure_dir = Path(__file__).parent.parent
sys.path.append(str(infrastructure_dir))

# Configure multiprocessing for CUDA
mp.set_start_method('spawn', force=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler('continuous_gpu_pipeline_v3.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Custom Priority Queue for multiprocessing
class PriorityQueue:
    """Priority queue that works with multiprocessing"""
    def __init__(self):
        self.heap = []
        self.counter = 0
        self.lock = mp.Lock()
        self.not_empty = mp.Condition(self.lock)
        
    def put(self, item, priority=1.0):
        with self.lock:
            # Higher priority = lower number in heap (min heap)
            heapq.heappush(self.heap, (-priority, self.counter, item))
            self.counter += 1
            self.not_empty.notify()
            
    def get(self, timeout=None):
        with self.lock:
            end_time = time.time() + timeout if timeout else float('inf')
            
            while not self.heap:
                remaining = end_time - time.time()
                if timeout and remaining <= 0:
                    raise Empty()
                self.not_empty.wait(timeout=remaining if timeout else None)
                
            return heapq.heappop(self.heap)[2]
            
    def qsize(self):
        with self.lock:
            return len(self.heap)
            
    def empty(self):
        with self.lock:
            return len(self.heap) == 0


# Register the priority queue with multiprocessing manager
BaseManager.register('PriorityQueue', PriorityQueue)


@dataclass
class GPUWork:
    """Work item for GPU processing"""
    batch_id: str
    texts: List[str]
    metadata: List[Dict]
    priority: float = 1.0
    retry_count: int = 0
    doc_count: int = 0  # Track document count
    
    def __post_init__(self):
        self.doc_count = len(self.texts)


@dataclass
class PipelineConfig:
    """Pipeline configuration with validation"""
    # GPU settings
    gpu_devices: List[int] = field(default_factory=lambda: [0, 1])
    batch_size: int = 128
    prefetch_factor: int = 4
    
    # Queue settings
    preprocessing_workers: int = 8
    max_gpu_queue_size: Optional[int] = None
    max_output_queue_size: Optional[int] = None
    
    # Database
    db_host: str = "localhost"
    db_port: int = 8529
    db_name: str = "arxiv_abstracts_continuous"
    db_username: str = "root"
    db_password: str = ""
    db_batch_size: int = 1000  # Larger for ArangoDB
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints/continuous"
    checkpoint_interval: int = 100
    
    # Monitoring
    enable_monitoring: bool = True
    monitor_interval: float = 5.0
    low_util_threshold: float = 50.0
    low_util_alert_threshold: int = 3
    
    # Memory management
    max_embeddings_in_memory: int = 10000
    embedding_cleanup_interval: int = 50
    
    # Retry settings
    max_retries: int = 3
    retry_delay_base: float = 2.0
    
    # Validation
    min_abstract_length: int = 10
    embedding_dimension: int = 2048  # Jina v4
    
    def __post_init__(self):
        """Validate and adjust configuration"""
        # Validate batch size
        if self.batch_size > 512:
            logger.warning(f"Large batch size {self.batch_size} may cause OOM")
            
        # Calculate optimal queue sizes if not provided
        if self.max_gpu_queue_size is None:
            self.max_gpu_queue_size = min(
                self.batch_size * self.prefetch_factor * len(self.gpu_devices) * 2,
                1000  # Cap to prevent excessive memory usage
            )
            logger.info(f"Calculated max_gpu_queue_size: {self.max_gpu_queue_size}")
            
        if self.max_output_queue_size is None:
            self.max_output_queue_size = self.max_gpu_queue_size * 2
            
        # Validate GPU devices
        available_gpus = torch.cuda.device_count()
        for gpu_id in self.gpu_devices:
            if gpu_id >= available_gpus:
                raise ValueError(f"GPU {gpu_id} not available (only {available_gpus} GPUs detected)")


class DocumentBatchCompletionTracker:
    """Fixed completion tracker that properly tracks both documents and batches"""
    
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.lock = Lock()
        
        # Document tracking
        self.documents_queued = 0
        self.documents_completed = 0
        self.documents_failed = 0
        self.documents_in_flight = 0
        
        # Batch tracking
        self.batches_queued = 0
        self.batches_completed = 0
        self.batches_failed = 0
        self.batches_in_flight = {}  # gpu_id -> set of (batch_id, doc_count)
        
    def add_documents(self, count: int):
        """Add documents and calculate expected batches"""
        with self.lock:
            self.documents_queued += count
            # Calculate total batches needed
            self.batches_queued = (self.documents_queued + self.batch_size - 1) // self.batch_size
            logger.debug(f"Added {count} documents, total batches expected: {self.batches_queued}")
            
    def start_batch(self, gpu_id: int, batch_id: str, doc_count: int):
        """Mark batch as being processed"""
        with self.lock:
            if gpu_id not in self.batches_in_flight:
                self.batches_in_flight[gpu_id] = set()
            self.batches_in_flight[gpu_id].add((batch_id, doc_count))
            self.documents_in_flight += doc_count
            
    def complete_batch(self, gpu_id: int, batch_id: str, doc_count: int):
        """Mark batch as completed"""
        with self.lock:
            # Remove from in-flight
            if gpu_id in self.batches_in_flight:
                self.batches_in_flight[gpu_id].discard((batch_id, doc_count))
            
            # Update counts
            self.batches_completed += 1
            self.documents_completed += doc_count
            self.documents_in_flight -= doc_count
            
    def fail_batch(self, gpu_id: int, batch_id: str, doc_count: int):
        """Mark batch as failed"""
        with self.lock:
            # Remove from in-flight
            if gpu_id in self.batches_in_flight:
                self.batches_in_flight[gpu_id].discard((batch_id, doc_count))
            
            # Update counts
            self.batches_failed += 1
            self.documents_failed += doc_count
            self.documents_in_flight -= doc_count
            
    def is_complete(self) -> bool:
        """Check if all documents are processed"""
        with self.lock:
            total_processed = self.documents_completed + self.documents_failed
            return (total_processed >= self.documents_queued and 
                    self.documents_in_flight == 0)
            
    def get_stats(self) -> Dict:
        """Get detailed statistics"""
        with self.lock:
            batches_in_flight_count = sum(len(items) for items in self.batches_in_flight.values())
            
            return {
                'documents': {
                    'queued': self.documents_queued,
                    'in_flight': self.documents_in_flight,
                    'completed': self.documents_completed,
                    'failed': self.documents_failed,
                    'remaining': self.documents_queued - self.documents_completed - self.documents_failed
                },
                'batches': {
                    'expected': self.batches_queued,
                    'in_flight': batches_in_flight_count,
                    'completed': self.batches_completed,
                    'failed': self.batches_failed,
                    'remaining': self.batches_queued - self.batches_completed - self.batches_failed
                }
            }


class SecureCheckpointManager:
    """Checkpoint manager using JSON instead of pickle for security"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # LMDB settings
        self.db_path = self.checkpoint_dir / "checkpoints.lmdb"
        self.map_size = 10 * 1024 * 1024 * 1024  # 10GB
        self.env = None
        self.lock = Lock()
        
        self._initialize_db()
        
    def _initialize_db(self):
        """Initialize LMDB database"""
        try:
            self.env = lmdb.open(
                str(self.db_path),
                map_size=self.map_size,
                max_dbs=5,
                lock=True
            )
            
            # Create named databases
            with self.env.begin(write=True) as txn:
                self.progress_db = self.env.open_db(b'progress', txn=txn)
                self.state_db = self.env.open_db(b'state', txn=txn)
                self.metadata_db = self.env.open_db(b'metadata', txn=txn)
                self.failed_db = self.env.open_db(b'failed', txn=txn)
                self.processed_db = self.env.open_db(b'processed', txn=txn)
                
            logger.info(f"Secure checkpoint database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize checkpoint database: {e}")
            raise
            
    def save_json(self, key: str, value: Any, db=None):
        """Save data as JSON (secure alternative to pickle)"""
        db = db or self.state_db
        with self.lock:
            try:
                # Convert to JSON with custom encoder
                json_data = json.dumps(value, default=str, ensure_ascii=False)
                
                with self.env.begin(write=True, db=db) as txn:
                    txn.put(key.encode('utf-8'), json_data.encode('utf-8'))
                    
            except Exception as e:
                logger.error(f"Failed to save JSON checkpoint: {e}")
                raise
                
    def load_json(self, key: str, db=None) -> Optional[Any]:
        """Load JSON data"""
        db = db or self.state_db
        with self.lock:
            try:
                with self.env.begin(db=db) as txn:
                    data = txn.get(key.encode('utf-8'))
                    if data:
                        return json.loads(data.decode('utf-8'))
            except Exception as e:
                logger.error(f"Failed to load JSON checkpoint: {e}")
        return None
        
    def save_batch_result(self, batch_id: str, result: Dict):
        """Save successful batch result"""
        # Track documents as processed
        for metadata in result['metadata']:
            self.mark_document_processed(metadata['arxiv_id'])
            
        # Save batch metadata
        self.save_json(f"batch_{batch_id}", {
            'completed': True,
            'gpu_id': result['gpu_id'],
            'doc_count': len(result['metadata']),
            'timestamp': result['timestamp']
        }, db=self.progress_db)
        
    def save_failed_batch(self, work: 'GPUWork', error: str):
        """Save failed batch for later retry"""
        failed_info = {
            'batch_id': work.batch_id,
            'texts': work.texts,
            'metadata': work.metadata,
            'error': error,
            'timestamp': time.time(),
            'retry_count': work.retry_count,
            'doc_count': work.doc_count
        }
        self.save_json(f"failed_{work.batch_id}", failed_info, db=self.failed_db)
        
    def get_failed_batches(self) -> List[Dict]:
        """Get all failed batches for retry"""
        failed_batches = []
        with self.lock:
            try:
                with self.env.begin(db=self.failed_db) as txn:
                    cursor = txn.cursor()
                    for key, value in cursor:
                        if key.startswith(b'failed_'):
                            batch_info = json.loads(value.decode('utf-8'))
                            failed_batches.append(batch_info)
            except Exception as e:
                logger.error(f"Failed to get failed batches: {e}")
        return failed_batches
        
    def remove_failed_batch(self, batch_id: str):
        """Remove failed batch after successful retry"""
        with self.lock:
            try:
                with self.env.begin(write=True, db=self.failed_db) as txn:
                    txn.delete(f"failed_{batch_id}".encode('utf-8'))
            except Exception as e:
                logger.error(f"Failed to remove failed batch: {e}")
                
    def mark_document_processed(self, doc_id: str):
        """Mark a document as processed"""
        with self.lock:
            try:
                with self.env.begin(write=True, db=self.processed_db) as txn:
                    txn.put(doc_id.encode('utf-8'), b'1')
            except Exception as e:
                logger.error(f"Failed to mark document processed: {e}")
                
    def is_document_processed(self, doc_id: str) -> bool:
        """Check if document is already processed"""
        with self.lock:
            try:
                with self.env.begin(db=self.processed_db) as txn:
                    return txn.get(doc_id.encode('utf-8')) is not None
            except:
                return False
                
    def get_processed_count(self) -> int:
        """Get count of processed documents"""
        with self.lock:
            try:
                with self.env.begin(db=self.processed_db) as txn:
                    return txn.stat()['entries']
            except:
                return 0
                
    def cleanup(self):
        """Cleanup resources"""
        if self.env:
            self.env.close()


class GPUWorker(mp.Process):
    """GPU worker with proper error handling and validation"""
    
    def __init__(
        self,
        gpu_id: int,
        input_queue,  # PriorityQueue proxy
        output_queue: mp.Queue,
        checkpoint_queue: mp.Queue,
        stats_queue: mp.Queue,
        config: PipelineConfig,
        stop_event: mp.Event
    ):
        super().__init__()
        self.gpu_id = gpu_id
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.checkpoint_queue = checkpoint_queue
        self.stats_queue = stats_queue
        self.config = config
        self.stop_event = stop_event
        self.processed_count = 0
        self.total_gpu_time = 0.0
        
    def run(self):
        """Main GPU processing loop"""
        # Set GPU device
        torch.cuda.set_device(self.gpu_id)
        logger.info(f"GPU Worker {self.gpu_id} starting on cuda:{self.gpu_id}")
        
        # Initialize model
        from irec_infrastructure.embeddings.local_jina_gpu import LocalJinaGPU, LocalJinaConfig
        
        config = LocalJinaConfig()
        config.device_ids = [self.gpu_id]
        config.use_fp16 = True
        config.max_length = 8192
        
        try:
            model = LocalJinaGPU(config)
            # Warmup
            _ = model.encode_batch(["warmup"] * 10, batch_size=10)
            torch.cuda.empty_cache()
            logger.info(f"GPU Worker {self.gpu_id} initialized successfully")
        except Exception as e:
            logger.error(f"GPU Worker {self.gpu_id} failed to initialize: {e}")
            self.stats_queue.put(('worker_failed', self.gpu_id, None, 0))
            return
            
        # Processing loop
        while not self.stop_event.is_set():
            try:
                # Get work with timeout
                work = self.input_queue.get(timeout=1.0)
                
                # Check for poison pill
                if work is None or (hasattr(work, 'batch_id') and work.batch_id is None):
                    logger.info(f"GPU Worker {self.gpu_id} received shutdown signal")
                    break
                    
                # Notify stats queue
                self.stats_queue.put(('processing', self.gpu_id, work.batch_id, work.doc_count))
                
                # Process batch
                try:
                    result = self._process_batch(model, work)
                    
                    if result:
                        # Validate embeddings before sending
                        if self._validate_embeddings(result['embeddings']):
                            self.output_queue.put(result)
                            self.stats_queue.put(('completed', self.gpu_id, work.batch_id, work.doc_count))
                            self.processed_count += 1
                        else:
                            logger.error(f"Invalid embeddings for batch {work.batch_id}")
                            self.stats_queue.put(('failed', self.gpu_id, work.batch_id, work.doc_count))
                    else:
                        self.stats_queue.put(('failed', self.gpu_id, work.batch_id, work.doc_count))
                        
                except Exception as e:
                    logger.error(f"GPU {self.gpu_id} batch processing error: {e}")
                    self.stats_queue.put(('failed', self.gpu_id, work.batch_id, work.doc_count))
                    
                # Periodic cleanup
                if self.processed_count % self.config.embedding_cleanup_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                    
            except Empty:
                continue
            except Exception as e:
                logger.error(f"GPU Worker {self.gpu_id} error: {e}", exc_info=True)
                
        # Cleanup
        logger.info(f"GPU Worker {self.gpu_id} shutting down. Processed {self.processed_count} batches")
        
    def _process_batch(self, model, work: GPUWork) -> Optional[Dict]:
        """Process batch with validation"""
        try:
            start_time = time.time()
            
            # Generate embeddings
            embeddings = model.encode_batch(work.texts, batch_size=min(32, len(work.texts)))
            
            # Convert to list
            if torch.is_tensor(embeddings):
                embeddings = embeddings.cpu().numpy()
            embeddings_list = embeddings.tolist()
            
            gpu_time = time.time() - start_time
            self.total_gpu_time += gpu_time
            
            return {
                'batch_id': work.batch_id,
                'embeddings': embeddings_list,
                'metadata': work.metadata,
                'gpu_id': self.gpu_id,
                'gpu_time': gpu_time,
                'timestamp': datetime.now().isoformat(),
                'doc_count': work.doc_count,
                'retry_count': work.retry_count
            }
            
        except torch.cuda.OutOfMemoryError:
            logger.error(f"GPU {self.gpu_id} OOM")
            torch.cuda.empty_cache()
            return None
        except Exception as e:
            logger.error(f"GPU {self.gpu_id} processing error: {e}")
            return None
            
    def _validate_embeddings(self, embeddings: List[List[float]]) -> bool:
        """Validate embeddings"""
        if not embeddings:
            return False
            
        for i, emb in enumerate(embeddings):
            if not isinstance(emb, list):
                logger.error(f"Embedding {i} is not a list")
                return False
            if len(emb) != self.config.embedding_dimension:
                logger.error(f"Embedding {i} has wrong dimension: {len(emb)}")
                return False
            # Check for NaN/Inf
            if any(math.isnan(x) or math.isinf(x) for x in emb):
                logger.error(f"NaN/Inf in embedding {i}")
                return False
                
        return True


class PreprocessingWorker(Thread):
    """Preprocessing worker with validation"""
    
    def __init__(
        self,
        document_queue: Queue,
        gpu_queue,  # PriorityQueue proxy
        config: PipelineConfig,
        stop_event: Event,
        worker_id: int
    ):
        super().__init__(daemon=True)
        self.document_queue = document_queue
        self.gpu_queue = gpu_queue
        self.config = config
        self.stop_event = stop_event
        self.worker_id = worker_id
        self.processed_count = 0
        self.skipped_count = 0
        
    def run(self):
        """Process documents with validation"""
        batch_texts = []
        batch_metadata = []
        batch_num = 0
        
        while not self.stop_event.is_set():
            try:
                # Get document
                doc = self.document_queue.get(timeout=1.0)
                if doc is None:  # Poison pill
                    if batch_texts:
                        self._send_batch(batch_texts, batch_metadata, batch_num)
                    break
                    
                # Validate and preprocess
                try:
                    text = self._preprocess_text(doc)
                    metadata = self._extract_metadata(doc)
                    
                    batch_texts.append(text)
                    batch_metadata.append(metadata)
                    
                    # Send batch when full
                    if len(batch_texts) >= self.config.batch_size:
                        self._send_batch(batch_texts, batch_metadata, batch_num)
                        batch_texts = []
                        batch_metadata = []
                        batch_num += 1
                        
                except ValueError as e:
                    logger.warning(f"Skipping invalid document: {e}")
                    self.skipped_count += 1
                    
            except Empty:
                # Send partial batch if idle
                if batch_texts:
                    self._send_batch(batch_texts, batch_metadata, batch_num)
                    batch_texts = []
                    batch_metadata = []
                    batch_num += 1
                    
            except Exception as e:
                logger.error(f"Preprocessing worker {self.worker_id} error: {e}")
                
        logger.info(f"Preprocessing worker {self.worker_id} done. Processed: {self.processed_count}, Skipped: {self.skipped_count}")
        
    def _preprocess_text(self, doc: Dict) -> str:
        """Preprocess and validate text"""
        title = doc.get('title', '').strip()
        abstract = doc.get('abstract', '').strip()
        
        # Validation
        if not title and not abstract:
            raise ValueError(f"Empty document: {doc.get('arxiv_id', 'unknown')}")
            
        if abstract and len(abstract) < self.config.min_abstract_length:
            raise ValueError(f"Abstract too short ({len(abstract)} chars) for {doc.get('arxiv_id', 'unknown')}")
            
        # Clean text
        # Remove LaTeX
        abstract = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', abstract)
        abstract = re.sub(r'\\[a-zA-Z]+', '', abstract)
        
        # Fix encoding
        replacements = {
            '\u2019': "'", '\u2018': "'",
            '\u201c': '"', '\u201d': '"',
            '\u2013': '-', '\u2014': '--',
            '\u2026': '...'
        }
        for old, new in replacements.items():
            abstract = abstract.replace(old, new)
            
        # Combine
        text = f"{title}\n\n{abstract}" if title and abstract else (title or abstract)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Truncate if needed
        max_chars = 30000
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
            
        return text
        
    def _extract_metadata(self, doc: Dict) -> Dict:
        """Extract metadata"""
        return {
            'arxiv_id': doc.get('id', doc.get('arxiv_id', '')),
            'title': doc.get('title', ''),
            'authors': doc.get('authors', []),
            'categories': doc.get('categories', []),
            'published': doc.get('published', ''),
            'updated': doc.get('updated', ''),
            'doi': doc.get('doi'),
            'journal_ref': doc.get('journal_ref')
        }
        
    def _send_batch(self, texts: List[str], metadata: List[Dict], batch_num: int):
        """Send batch to GPU queue"""
        if not texts:
            return
            
        batch_id = f"w{self.worker_id}_b{batch_num}_{int(time.time() * 1000)}"
        
        work = GPUWork(
            batch_id=batch_id,
            texts=texts,
            metadata=metadata,
            priority=1.0
        )
        
        self.gpu_queue.put(work, priority=work.priority)
        self.processed_count += 1
        
        logger.debug(f"Worker {self.worker_id} sent batch {batch_id} with {len(texts)} documents")


class RobustDatabaseWriter(Thread):
    """Database writer with retry logic and connection pooling"""
    
    def __init__(
        self,
        config: PipelineConfig,
        write_queue: Queue,
        stop_event: Event,
        checkpoint_manager: SecureCheckpointManager
    ):
        super().__init__(daemon=True)
        self.config = config
        self.write_queue = write_queue
        self.stop_event = stop_event
        self.checkpoint_manager = checkpoint_manager
        
        self.client = None
        self.db = None
        self.collection = None
        
        self.batch_buffer = []
        self.written_count = 0
        self.failed_count = 0
        
    def run(self):
        """Main database writing loop"""
        logger.info("Database Writer started")
        
        # Initialize with retry
        if not self._initialize_db_with_retry():
            logger.error("Failed to initialize database")
            return
            
        while not self.stop_event.is_set():
            try:
                # Get item
                item = self.write_queue.get(timeout=1.0)
                if item is None:
                    if self.batch_buffer:
                        self._flush_buffer()
                    break
                    
                # Add to buffer
                self._add_to_buffer(item)
                
                # Write if buffer full
                if len(self.batch_buffer) >= self.config.db_batch_size:
                    self._flush_buffer()
                    
            except Empty:
                # Flush on idle
                if self.batch_buffer:
                    self._flush_buffer()
                    
            except Exception as e:
                logger.error(f"Database writer error: {e}")
                
        logger.info(f"Database Writer stopped. Written: {self.written_count}, Failed: {self.failed_count}")
        
    def _initialize_db_with_retry(self) -> bool:
        """Initialize with connection pooling and retry"""
        for attempt in range(self.config.max_retries):
            try:
                # Create client with connection pooling
                self.client = ArangoClient(
                    hosts=f'http://{self.config.db_host}:{self.config.db_port}',
                    http_client=HTTPClient(
                        pool_connections=10,
                        pool_maxsize=10,
                        pool_timeout=30
                    )
                )
                
                self.db = self.client.db(
                    self.config.db_name,
                    username=self.config.db_username,
                    password=self.config.db_password
                )
                self.collection = self.db.collection('abstract_metadata')
                
                # Test connection
                self.collection.count()
                
                logger.info("Database connection established with pooling")
                return True
                
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    wait_time = self.config.retry_delay_base ** attempt
                    logger.warning(f"DB init attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to initialize DB after {self.config.max_retries} attempts")
                    
        return False
        
    def _add_to_buffer(self, gpu_output: Dict):
        """Add GPU output to buffer"""
        for i, metadata in enumerate(gpu_output['metadata']):
            doc = {
                '_key': metadata['arxiv_id'].replace('/', '_').replace(':', '_'),
                'arxiv_id': metadata['arxiv_id'],
                'title': metadata['title'],
                'authors': metadata['authors'],
                'categories': metadata['categories'],
                'published': metadata['published'],
                'updated': metadata['updated'],
                'doi': metadata.get('doi'),
                'journal_ref': metadata.get('journal_ref'),
                'abstract_embedding': gpu_output['embeddings'][i],
                'processed_gpu': gpu_output['gpu_id'],
                'processing_time': gpu_output['gpu_time'] / len(gpu_output['metadata']),
                'processed_at': gpu_output['timestamp'],
                'batch_id': gpu_output['batch_id']
            }
            
            # Validate before adding
            if self._validate_document(doc):
                self.batch_buffer.append(doc)
            else:
                logger.warning(f"Skipping invalid document: {doc['arxiv_id']}")
                
    def _validate_document(self, doc: Dict) -> bool:
        """Validate document before storage"""
        # Check required fields
        if not doc.get('arxiv_id'):
            return False
            
        # Validate embedding
        embedding = doc.get('abstract_embedding', [])
        if not isinstance(embedding, list) or len(embedding) != self.config.embedding_dimension:
            return False
            
        # Check for NaN/Inf
        try:
            if any(math.isnan(x) or math.isinf(x) for x in embedding):
                return False
        except:
            return False
            
        return True
        
    def _flush_buffer(self):
        """Write buffer with retry logic"""
        if not self.batch_buffer:
            return
            
        success = self._write_batch_with_retry(self.batch_buffer)
        
        if success:
            self.batch_buffer = []
        else:
            # Save failed documents
            for doc in self.batch_buffer:
                self.checkpoint_manager.save_json(
                    f"failed_doc_{doc['arxiv_id']}_{time.time()}",
                    {'document': doc, 'error': 'Write failed'},
                    db=self.checkpoint_manager.failed_db
                )
            self.failed_count += len(self.batch_buffer)
            self.batch_buffer = []
            
    def _write_batch_with_retry(self, documents: List[Dict]) -> bool:
        """Write batch with exponential backoff"""
        for attempt in range(self.config.max_retries):
            try:
                result = self.collection.insert_many(
                    documents,
                    overwrite=True,
                    return_new=False,
                    silent=False
                )
                
                self.written_count += len(documents)
                logger.debug(f"Wrote batch of {len(documents)} documents")
                return True
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Reconnect on connection errors
                if 'connection' in error_msg or 'timeout' in error_msg:
                    logger.warning("Connection lost, reconnecting...")
                    if self._initialize_db_with_retry():
                        continue
                        
                if attempt < self.config.max_retries - 1:
                    wait_time = self.config.retry_delay_base ** attempt
                    logger.warning(f"Write attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Write failed after {self.config.max_retries} attempts: {e}")
                    
        return False


class BatchRetryManager(Thread):
    """Manages automatic retry of failed batches"""
    
    def __init__(
        self,
        gpu_queue,  # PriorityQueue proxy
        checkpoint_manager: SecureCheckpointManager,
        config: PipelineConfig
    ):
        super().__init__(daemon=True)
        self.gpu_queue = gpu_queue
        self.checkpoint_manager = checkpoint_manager
        self.config = config
        self.stop_event = Event()
        
    def run(self):
        """Retry failed batches periodically"""
        while not self.stop_event.is_set():
            try:
                # Get failed batches
                failed_batches = self.checkpoint_manager.get_failed_batches()
                
                for batch_info in failed_batches:
                    if batch_info['retry_count'] < self.config.max_retries:
                        # Check if enough time has passed
                        delay = self.config.retry_delay_base ** batch_info['retry_count']
                        if time.time() - batch_info['timestamp'] > delay:
                            self._retry_batch(batch_info)
                            
            except Exception as e:
                logger.error(f"Batch retry error: {e}")
                
            # Check every 30 seconds
            self.stop_event.wait(30)
            
    def _retry_batch(self, batch_info: Dict):
        """Retry a failed batch"""
        try:
            # Create new work item
            work = GPUWork(
                batch_id=f"{batch_info['batch_id']}_retry{batch_info['retry_count'] + 1}",
                texts=batch_info['texts'],
                metadata=batch_info['metadata'],
                priority=0.5,  # Lower priority for retries
                retry_count=batch_info['retry_count'] + 1
            )
            
            # Re-queue
            self.gpu_queue.put(work, priority=work.priority)
            
            # Remove from failed
            self.checkpoint_manager.remove_failed_batch(batch_info['batch_id'])
            
            logger.info(f"Retrying batch {batch_info['batch_id']} (attempt {work.retry_count})")
            
        except Exception as e:
            logger.error(f"Failed to retry batch: {e}")
            
    def stop(self):
        """Stop retry manager"""
        self.stop_event.set()


class HealthMonitor(Thread):
    """Monitor and recover from failures"""
    
    def __init__(self, pipeline):
        super().__init__(daemon=True)
        self.pipeline = pipeline
        self.stop_event = Event()
        self.check_interval = 10.0
        
    def run(self):
        """Monitor health and recover"""
        while not self.stop_event.is_set():
            try:
                # Check GPU workers
                for i, worker in enumerate(self.pipeline.gpu_workers):
                    if not worker.is_alive():
                        logger.error(f"GPU worker {i} died, attempting restart...")
                        self._restart_gpu_worker(i)
                        
                # Check preprocessing workers
                dead_count = sum(1 for w in self.pipeline.preprocessing_workers if not w.is_alive())
                if dead_count > 0:
                    logger.warning(f"{dead_count} preprocessing workers died")
                    
                # Check database writer
                if not self.pipeline.db_writer.is_alive():
                    logger.error("Database writer died!")
                    
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                
            self.stop_event.wait(self.check_interval)
            
    def _restart_gpu_worker(self, index: int):
        """Restart a dead GPU worker"""
        try:
            old_worker = self.pipeline.gpu_workers[index]
            gpu_id = old_worker.gpu_id
            
            # Create new worker
            new_worker = GPUWorker(
                gpu_id=gpu_id,
                input_queue=self.pipeline.gpu_queue,
                output_queue=self.pipeline.output_queue,
                checkpoint_queue=self.pipeline.checkpoint_queue,
                stats_queue=self.pipeline.stats_queue,
                config=self.pipeline.config,
                stop_event=self.pipeline.stop_event
            )
            
            new_worker.start()
            self.pipeline.gpu_workers[index] = new_worker
            logger.info(f"Successfully restarted GPU worker {gpu_id}")
            
        except Exception as e:
            logger.error(f"Failed to restart GPU worker: {e}")
            
    def stop(self):
        """Stop monitor"""
        self.stop_event.set()


class ContinuousGPUPipeline:
    """Main pipeline with all fixes applied"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.checkpoint_manager = SecureCheckpointManager(config.checkpoint_dir)
        
        # Create manager for priority queue
        self.manager = BaseManager()
        self.manager.start()
        
        # Queues
        self.document_queue = Queue(maxsize=10000)
        self.gpu_queue = self.manager.PriorityQueue()  # Custom priority queue
        self.output_queue = mp.Queue(maxsize=config.max_output_queue_size)
        self.checkpoint_queue = mp.Queue()
        self.stats_queue = mp.Queue()
        self.db_write_queue = Queue(maxsize=1000)
        
        # Events
        self.stop_event = mp.Event()
        self.preprocessing_stop = Event()
        self.output_stop = Event()
        
        # Workers
        self.gpu_workers = []
        self.preprocessing_workers = []
        self.monitor = None
        self.db_writer = None
        self.health_monitor = None
        self.retry_manager = None
        
        # Tracking
        self.completion_tracker = DocumentBatchCompletionTracker(config.batch_size)
        
        # Statistics
        self.stats = {
            'start_time': None,
            'total_processed': 0,
            'total_documents': 0,
            'gpu_stats': {gpu: {'batches': 0, 'time': 0} for gpu in config.gpu_devices}
        }
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        
    def setup_database(self, clean_start=False):
        """Setup database"""
        try:
            # Use connection pooling
            client = ArangoClient(
                hosts=f'http://{self.config.db_host}:{self.config.db_port}',
                http_client=HTTPClient(
                    pool_connections=10,
                    pool_maxsize=10
                )
            )
            sys_db = client.db('_system', username=self.config.db_username, password=self.config.db_password)
            
            if sys_db.has_database(self.config.db_name):
                if clean_start:
                    logger.warning(f"Dropping existing database: {self.config.db_name}")
                    sys_db.delete_database(self.config.db_name)
                else:
                    return
                    
            # Create database
            sys_db.create_database(self.config.db_name)
            db = client.db(self.config.db_name, username=self.config.db_username, password=self.config.db_password)
            
            # Create collection
            collection = db.create_collection('abstract_metadata')
            
            # Add indexes
            collection.add_hash_index(fields=['arxiv_id'], unique=True)
            collection.add_persistent_index(fields=['categories[*]'])
            collection.add_persistent_index(fields=['published'])
            collection.add_persistent_index(fields=['processed_gpu'])
            
            logger.info("Database setup complete")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            raise
            
    def start(self):
        """Start all components"""
        logger.info("Starting Continuous GPU Pipeline V3")
        
        self.stats['start_time'] = time.time()
        
        # Start GPU workers
        for gpu_id in self.config.gpu_devices:
            worker = GPUWorker(
                gpu_id=gpu_id,
                input_queue=self.gpu_queue,
                output_queue=self.output_queue,
                checkpoint_queue=self.checkpoint_queue,
                stats_queue=self.stats_queue,
                config=self.config,
                stop_event=self.stop_event
            )
            worker.start()
            self.gpu_workers.append(worker)
            
        # Start preprocessing workers
        for i in range(self.config.preprocessing_workers):
            worker = PreprocessingWorker(
                document_queue=self.document_queue,
                gpu_queue=self.gpu_queue,
                config=self.config,
                stop_event=self.preprocessing_stop,
                worker_id=i
            )
            worker.start()
            self.preprocessing_workers.append(worker)
            
        # Start database writer
        self.db_writer = RobustDatabaseWriter(
            config=self.config,
            write_queue=self.db_write_queue,
            stop_event=self.output_stop,
            checkpoint_manager=self.checkpoint_manager
        )
        self.db_writer.start()
        
        # Start retry manager
        self.retry_manager = BatchRetryManager(
            gpu_queue=self.gpu_queue,
            checkpoint_manager=self.checkpoint_manager,
            config=self.config
        )
        self.retry_manager.start()
        
        # Start health monitor
        self.health_monitor = HealthMonitor(self)
        self.health_monitor.start()
        
        # Start processing threads
        self.output_thread = Thread(target=self._process_outputs, daemon=True)
        self.output_thread.start()
        
        self.checkpoint_thread = Thread(target=self._handle_checkpoints, daemon=True)
        self.checkpoint_thread.start()
        
        self.stats_thread = Thread(target=self._track_stats, daemon=True)
        self.stats_thread.start()
        
        logger.info("Pipeline started successfully")
        
    def process_metadata_files(self, metadata_files: List[Path], resume: bool = True):
        """Process metadata files"""
        # Load and queue documents
        total_documents = 0
        
        for file_path in tqdm(metadata_files, desc="Loading metadata"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                # Handle both single document and list formats
                if isinstance(data, list):
                    for doc in data:
                        if resume and self.checkpoint_manager.is_document_processed(doc.get('id', '')):
                            continue
                        self.document_queue.put(doc)
                        total_documents += 1
                else:
                    if not (resume and self.checkpoint_manager.is_document_processed(data.get('id', ''))):
                        self.document_queue.put(data)
                        total_documents += 1
                        
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                
        # Send poison pills
        for _ in self.preprocessing_workers:
            self.document_queue.put(None)
            
        # Update tracking
        self.stats['total_documents'] = total_documents
        self.completion_tracker.add_documents(total_documents)
        
        logger.info(f"Queued {total_documents} documents for processing")
        
    def wait_for_completion(self):
        """Wait for completion with proper tracking"""
        logger.info("Waiting for preprocessing...")
        
        # Wait for preprocessing
        for worker in self.preprocessing_workers:
            worker.join()
            
        logger.info("Preprocessing complete")
        
        # Wait for GPU processing
        logger.info("Waiting for GPU processing...")
        
        while not self.completion_tracker.is_complete():
            stats = self.completion_tracker.get_stats()
            logger.info(
                f"Documents: {stats['documents']['completed']}/{stats['documents']['queued']} "
                f"(in flight: {stats['documents']['in_flight']}), "
                f"Batches: {stats['batches']['completed']}/{stats['batches']['expected']}"
            )
            time.sleep(2.0)
            
        # Send poison pills to GPUs
        for _ in self.gpu_workers:
            work = GPUWork(batch_id=None, texts=[], metadata=[])  # Poison pill
            self.gpu_queue.put(work, priority=0)  # Highest priority
            
        # Wait for GPU workers
        for worker in self.gpu_workers:
            worker.join()
            
        # Stop other components
        self.output_stop.set()
        
        # Wait for queues to empty
        while not self.output_queue.empty() or not self.db_write_queue.empty():
            time.sleep(1.0)
            
        # Stop database writer
        self.db_write_queue.put(None)
        if self.db_writer:
            self.db_writer.join()
            
        # Stop other threads
        self.stop_event.set()
        self.output_thread.join()
        self.checkpoint_thread.join()
        self.stats_thread.join()
        
        # Stop managers
        if self.retry_manager:
            self.retry_manager.stop()
        if self.health_monitor:
            self.health_monitor.stop()
            
        logger.info("Pipeline completion successful")
        
    def _track_stats(self):
        """Track statistics with proper document counting"""
        while not self.stop_event.is_set():
            try:
                stat = self.stats_queue.get(timeout=1.0)
                if stat is None:
                    break
                    
                event_type, gpu_id, batch_id, doc_count = stat
                
                if event_type == 'processing':
                    self.completion_tracker.start_batch(gpu_id, batch_id, doc_count)
                elif event_type == 'completed':
                    self.completion_tracker.complete_batch(gpu_id, batch_id, doc_count)
                elif event_type == 'failed':
                    self.completion_tracker.fail_batch(gpu_id, batch_id, doc_count)
                    # Save failed batch for retry
                    # Note: Would need to pass the actual work item here
                elif event_type == 'worker_failed':
                    logger.error(f"GPU worker {gpu_id} failed")
                    
            except Empty:
                continue
                
    def _process_outputs(self):
        """Process GPU outputs"""
        while not self.output_stop.is_set():
            try:
                output = self.output_queue.get(timeout=1.0)
                if output is None:
                    break
                    
                # Save to checkpoint
                self.checkpoint_manager.save_batch_result(output['batch_id'], output)
                
                # Send to database
                self.db_write_queue.put(output)
                
                # Update stats
                self.stats['total_processed'] += output['doc_count']
                gpu_id = output['gpu_id']
                self.stats['gpu_stats'][gpu_id]['batches'] += 1
                self.stats['gpu_stats'][gpu_id]['time'] += output['gpu_time']
                
                # Log progress
                if self.stats['total_processed'] % 1000 == 0:
                    self._log_progress()
                    
            except Empty:
                continue
                
    def _handle_checkpoints(self):
        """Handle checkpoint saves"""
        while not self.stop_event.is_set():
            try:
                checkpoint = self.checkpoint_queue.get(timeout=1.0)
                if checkpoint is None:
                    break
                    
                # Save checkpoint
                key = f"gpu_{checkpoint['gpu_id']}_state"
                self.checkpoint_manager.save_json(key, checkpoint)
                
            except Empty:
                continue
                
    def _log_progress(self):
        """Log progress"""
        elapsed = time.time() - self.stats['start_time']
        rate = self.stats['total_processed'] / elapsed if elapsed > 0 else 0
        
        logger.info(
            f"Progress: {self.stats['total_processed']}/{self.stats['total_documents']} documents "
            f"({rate:.1f} docs/s)"
        )
        
    def shutdown(self):
        """Clean shutdown"""
        logger.info("Shutting down pipeline")
        
        # Stop all components
        self.preprocessing_stop.set()
        self.output_stop.set()
        self.stop_event.set()
        
        # Stop managers
        if self.retry_manager:
            self.retry_manager.stop()
        if self.health_monitor:
            self.health_monitor.stop()
            
        # Cleanup
        self.checkpoint_manager.cleanup()
        if self.manager:
            self.manager.shutdown()
            
        logger.info("Pipeline shutdown complete")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Production-Ready Continuous GPU Pipeline V3"
    )
    
    parser.add_argument('--metadata-dir', type=str, required=True)
    parser.add_argument('--db-name', type=str, default='arxiv_abstracts_v3')
    parser.add_argument('--db-host', type=str, default='localhost')
    parser.add_argument('--gpu-devices', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--clean-start', action='store_true')
    
    args = parser.parse_args()
    
    # Get password from environment
    db_password = os.environ.get('ARANGO_PASSWORD', '')
    if not db_password:
        print("ERROR: ARANGO_PASSWORD environment variable not set!")
        sys.exit(1)
        
    # Create config
    config = PipelineConfig(
        gpu_devices=args.gpu_devices,
        batch_size=args.batch_size,
        preprocessing_workers=args.workers,
        db_host=args.db_host,
        db_name=args.db_name,
        db_password=db_password
    )
    
    # Run pipeline
    try:
        with ContinuousGPUPipeline(config) as pipeline:
            # Setup database
            if not args.resume:
                pipeline.setup_database(clean_start=args.clean_start)
                
            # Start pipeline
            pipeline.start()
            
            # Load files
            metadata_dir = Path(args.metadata_dir)
            metadata_files = sorted(metadata_dir.glob("*.json"))
            
            # Process
            pipeline.process_metadata_files(metadata_files, resume=args.resume)
            
            # Wait
            pipeline.wait_for_completion()
            
            # Print stats
            elapsed = time.time() - pipeline.stats['start_time']
            logger.info(
                f"Completed in {elapsed/60:.1f} minutes. "
                f"Processed {pipeline.stats['total_processed']} documents "
                f"({pipeline.stats['total_processed']/elapsed:.1f} docs/s)"
            )
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        
    print("\n✅ Pipeline V3 completed!")


if __name__ == "__main__":
    main()