#!/usr/bin/env python3
"""
Critical improvements and fixes for the continuous GPU pipeline V2.
These can be integrated into the existing implementation.
"""

import json
import pickle
import time
import math
import logging
from typing import Dict, List, Optional, Any
from threading import Thread, Lock
from queue import Queue, Empty
from dataclasses import dataclass
from pathlib import Path
import lmdb

logger = logging.getLogger(__name__)


# 1. IMPROVED COMPLETION TRACKER - Tracks both documents and batches
class ImprovedCompletionTracker:
    """Enhanced completion tracker that handles both documents and batches"""
    
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.lock = Lock()
        
        # Document tracking
        self.documents_queued = 0
        self.documents_completed = 0
        self.documents_failed = 0
        
        # Batch tracking
        self.batches_expected = 0
        self.batches_processing = {}  # gpu_id -> set of batch_ids
        self.batches_completed = 0
        self.batches_failed = 0
        
    def add_documents(self, count: int):
        """Add documents and calculate expected batches"""
        with self.lock:
            self.documents_queued += count
            # Calculate batches needed
            new_batches = (count + self.batch_size - 1) // self.batch_size
            self.batches_expected += new_batches
            
    def start_batch(self, gpu_id: int, batch_id: str, doc_count: int):
        """Mark batch as being processed"""
        with self.lock:
            if gpu_id not in self.batches_processing:
                self.batches_processing[gpu_id] = set()
            self.batches_processing[gpu_id].add((batch_id, doc_count))
            
    def complete_batch(self, gpu_id: int, batch_id: str, doc_count: int):
        """Mark batch as completed"""
        with self.lock:
            # Remove from processing
            if gpu_id in self.batches_processing:
                self.batches_processing[gpu_id].discard((batch_id, doc_count))
            
            # Update counts
            self.batches_completed += 1
            self.documents_completed += doc_count
            
    def fail_batch(self, gpu_id: int, batch_id: str, doc_count: int):
        """Mark batch as failed"""
        with self.lock:
            # Remove from processing
            if gpu_id in self.batches_processing:
                self.batches_processing[gpu_id].discard((batch_id, doc_count))
            
            # Update counts
            self.batches_failed += 1
            self.documents_failed += doc_count
            
    def is_complete(self) -> bool:
        """Check if all documents are processed"""
        with self.lock:
            # Count documents in flight
            docs_in_flight = sum(
                doc_count for gpu_items in self.batches_processing.values()
                for _, doc_count in gpu_items
            )
            
            # All documents should be either completed, failed, or in flight
            total_processed = self.documents_completed + self.documents_failed + docs_in_flight
            return total_processed >= self.documents_queued and docs_in_flight == 0
            
    def get_stats(self) -> Dict:
        """Get detailed statistics"""
        with self.lock:
            docs_in_flight = sum(
                doc_count for gpu_items in self.batches_processing.values()
                for _, doc_count in gpu_items
            )
            batches_in_flight = sum(len(items) for items in self.batches_processing.values())
            
            return {
                'documents': {
                    'queued': self.documents_queued,
                    'in_flight': docs_in_flight,
                    'completed': self.documents_completed,
                    'failed': self.documents_failed,
                    'pending': self.documents_queued - self.documents_completed - self.documents_failed - docs_in_flight
                },
                'batches': {
                    'expected': self.batches_expected,
                    'in_flight': batches_in_flight,
                    'completed': self.batches_completed,
                    'failed': self.batches_failed,
                    'pending': self.batches_expected - self.batches_completed - self.batches_failed - batches_in_flight
                }
            }


# 2. SECURE CHECKPOINT MANAGER - Uses JSON instead of pickle
class SecureCheckpointManager:
    """Checkpoint manager using JSON for security"""
    
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
                max_dbs=4,  # Added one for failed batches
                lock=True
            )
            
            # Create named databases
            with self.env.begin(write=True) as txn:
                self.progress_db = self.env.open_db(b'progress', txn=txn)
                self.state_db = self.env.open_db(b'state', txn=txn)
                self.metadata_db = self.env.open_db(b'metadata', txn=txn)
                self.failed_db = self.env.open_db(b'failed', txn=txn)
                
            logger.info(f"Secure checkpoint database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize checkpoint database: {e}")
            raise
            
    def save_json(self, key: str, value: Dict, db=None):
        """Save data as JSON (secure alternative to pickle)"""
        db = db or self.state_db
        with self.lock:
            try:
                # Convert to JSON with custom encoder for datetime, etc.
                json_data = json.dumps(value, default=str, ensure_ascii=False)
                
                with self.env.begin(write=True, db=db) as txn:
                    txn.put(key.encode('utf-8'), json_data.encode('utf-8'))
                    
            except Exception as e:
                logger.error(f"Failed to save JSON checkpoint: {e}")
                raise
                
    def load_json(self, key: str, db=None) -> Optional[Dict]:
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
        
    def save_failed_batch(self, batch_id: str, batch_data: Dict, error: str):
        """Save failed batch for later retry"""
        failed_info = {
            'batch_id': batch_id,
            'batch_data': batch_data,
            'error': error,
            'timestamp': time.time(),
            'retry_count': batch_data.get('retry_count', 0)
        }
        self.save_json(f"failed_{batch_id}", failed_info, db=self.failed_db)
        
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


# 3. DATABASE WRITER WITH RETRY LOGIC
class RobustDatabaseWriter(Thread):
    """Database writer with retry logic and failure recovery"""
    
    def __init__(self, config, write_queue: Queue, stop_event, checkpoint_manager):
        super().__init__(daemon=True)
        self.config = config
        self.write_queue = write_queue
        self.stop_event = stop_event
        self.checkpoint_manager = checkpoint_manager
        
        # Connection settings
        self.client = None
        self.db = None
        self.collection = None
        
        # Batch settings
        self.batch_buffer = []
        self.batch_size = 1000  # Larger batches for ArangoDB
        self.max_retries = 3
        
        # Statistics
        self.written_count = 0
        self.failed_count = 0
        
    def _initialize_db_with_retry(self) -> bool:
        """Initialize database with connection pooling and retry"""
        from arango import ArangoClient
        from arango.http import HTTPClient
        
        for attempt in range(self.max_retries):
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
                
                logger.info("Database connection established with connection pooling")
                return True
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"DB connection attempt {attempt + 1}/{self.max_retries} failed: {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to connect to database after {self.max_retries} attempts")
                    
        return False
        
    def _write_batch_with_retry(self, documents: List[Dict]) -> bool:
        """Write batch with exponential backoff retry"""
        for attempt in range(self.max_retries):
            try:
                # Validate documents before writing
                valid_docs = []
                for doc in documents:
                    if self._validate_document(doc):
                        valid_docs.append(doc)
                    else:
                        logger.warning(f"Skipping invalid document: {doc.get('arxiv_id', 'unknown')}")
                
                if not valid_docs:
                    return True  # Nothing to write
                
                # Bulk insert with overwrite
                result = self.collection.insert_many(
                    valid_docs,
                    overwrite=True,
                    return_new=False,
                    silent=False
                )
                
                self.written_count += len(valid_docs)
                logger.debug(f"Successfully wrote batch of {len(valid_docs)} documents")
                return True
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check if it's a connection error
                if 'connection' in error_msg or 'timeout' in error_msg:
                    logger.warning("Database connection lost, attempting to reconnect...")
                    if self._initialize_db_with_retry():
                        continue  # Retry with new connection
                
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"DB write attempt {attempt + 1}/{self.max_retries} failed: {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"DB write failed after {self.max_retries} attempts: {e}")
                    # Save failed batch for recovery
                    self._save_failed_documents(documents, str(e))
                    return False
                    
        return False
        
    def _validate_document(self, doc: Dict) -> bool:
        """Validate document before storage"""
        # Check required fields
        if not doc.get('arxiv_id'):
            return False
            
        # Validate embedding
        embedding = doc.get('embedding', [])
        if not self._validate_embedding(embedding):
            return False
            
        return True
        
    def _validate_embedding(self, embedding: List[float], expected_dim: int = 2048) -> bool:
        """Validate embedding dimensions and values"""
        if not isinstance(embedding, list) or len(embedding) != expected_dim:
            return False
            
        # Check for valid numerical values
        try:
            for val in embedding:
                if not isinstance(val, (int, float)):
                    return False
                if math.isnan(val) or math.isinf(val):
                    return False
        except:
            return False
            
        return True
        
    def _save_failed_documents(self, documents: List[Dict], error: str):
        """Save failed documents to checkpoint for later recovery"""
        timestamp = time.time()
        for doc in documents:
            key = f"failed_doc_{doc.get('arxiv_id', 'unknown')}_{timestamp}"
            self.checkpoint_manager.save_json(
                key,
                {
                    'document': doc,
                    'error': error,
                    'timestamp': timestamp
                },
                db=self.checkpoint_manager.failed_db
            )
        self.failed_count += len(documents)
        logger.error(f"Saved {len(documents)} failed documents for recovery")
        
    def run(self):
        """Main database writing loop with recovery"""
        logger.info("Robust Database Writer started")
        
        # Initialize database connection
        if not self._initialize_db_with_retry():
            logger.error("Failed to initialize database connection")
            return
            
        # Check for previously failed documents
        self._recover_failed_documents()
        
        while not self.stop_event.is_set():
            try:
                # Get write request with timeout
                item = self.write_queue.get(timeout=1.0)
                if item is None:  # Poison pill
                    # Write any remaining buffered items
                    if self.batch_buffer:
                        self._flush_buffer()
                    break
                    
                # Process GPU output batch
                self._process_gpu_output(item)
                
                # Write batch if full
                if len(self.batch_buffer) >= self.batch_size:
                    self._flush_buffer()
                    
            except Empty:
                # Write partial batch if idle
                if self.batch_buffer:
                    self._flush_buffer()
                continue
                
            except Exception as e:
                logger.error(f"Database writer error: {e}")
                
        # Final cleanup
        self._cleanup()
        
    def _process_gpu_output(self, gpu_output: Dict):
        """Process GPU output and add to buffer"""
        for i, metadata in enumerate(gpu_output['metadata']):
            doc = {
                '_key': metadata['arxiv_id'].replace('/', '_'),
                'arxiv_id': metadata['arxiv_id'],
                'title': metadata['title'],
                'authors': metadata['authors'],
                'categories': metadata['categories'],
                'published': metadata['published'],
                'updated': metadata['updated'],
                'doi': metadata.get('doi'),
                'journal_ref': metadata.get('journal_ref'),
                'embedding': gpu_output['embeddings'][i],
                'processed_gpu': gpu_output['gpu_id'],
                'processing_time': gpu_output['gpu_time'] / len(gpu_output['metadata']),
                'processed_at': gpu_output['timestamp'],
                'batch_id': gpu_output['batch_id']
            }
            self.batch_buffer.append(doc)
            
    def _flush_buffer(self):
        """Write buffered documents to database"""
        if not self.batch_buffer:
            return
            
        success = self._write_batch_with_retry(self.batch_buffer)
        if success:
            self.batch_buffer = []
        else:
            # Keep buffer for next attempt
            logger.warning(f"Failed to write batch, keeping {len(self.batch_buffer)} documents in buffer")
            
    def _recover_failed_documents(self):
        """Attempt to recover previously failed documents"""
        # This would load failed documents from checkpoint and retry them
        pass
        
    def _cleanup(self):
        """Cleanup database connection and log statistics"""
        if self.client:
            self.client.close()
            
        logger.info(
            f"Database Writer stopped. "
            f"Successfully wrote: {self.written_count}, "
            f"Failed: {self.failed_count}"
        )


# 4. BATCH RETRY MANAGER
class BatchRetryManager:
    """Manages automatic retry of failed batches"""
    
    def __init__(self, gpu_queue, checkpoint_manager, max_retries: int = 3):
        self.gpu_queue = gpu_queue
        self.checkpoint_manager = checkpoint_manager
        self.max_retries = max_retries
        self.retry_thread = None
        self.stop_event = None
        
    def start(self):
        """Start retry thread"""
        self.stop_event = Thread.Event()
        self.retry_thread = Thread(target=self._retry_loop, daemon=True)
        self.retry_thread.start()
        
    def stop(self):
        """Stop retry thread"""
        if self.stop_event:
            self.stop_event.set()
        if self.retry_thread:
            self.retry_thread.join(timeout=5)
            
    def _retry_loop(self):
        """Main retry loop"""
        while not self.stop_event.is_set():
            try:
                # Get failed batches
                failed_batches = self.checkpoint_manager.get_failed_batches()
                
                for batch_info in failed_batches:
                    if batch_info['retry_count'] < self.max_retries:
                        # Calculate delay with exponential backoff
                        delay = 2 ** batch_info['retry_count']
                        
                        # Check if enough time has passed
                        if time.time() - batch_info['timestamp'] > delay:
                            self._retry_batch(batch_info)
                            
            except Exception as e:
                logger.error(f"Batch retry error: {e}")
                
            # Check every 30 seconds
            self.stop_event.wait(30)
            
    def _retry_batch(self, batch_info: Dict):
        """Retry a failed batch"""
        try:
            batch_data = batch_info['batch_data']
            
            # Create new work item with increased retry count
            from continuous_gpu_pipeline_v2 import GPUWork
            work = GPUWork(
                batch_id=f"{batch_data['batch_id']}_retry{batch_info['retry_count'] + 1}",
                texts=batch_data['texts'],
                metadata=batch_data['metadata'],
                priority=0.5,  # Lower priority for retries
                retry_count=batch_info['retry_count'] + 1
            )
            
            # Re-queue the work
            self.gpu_queue.put(work)
            
            # Remove from failed batches
            self.checkpoint_manager.remove_failed_batch(batch_data['batch_id'])
            
            logger.info(f"Retrying batch {batch_data['batch_id']} (attempt {work.retry_count})")
            
        except Exception as e:
            logger.error(f"Failed to retry batch: {e}")


# 5. IMPROVED PREPROCESSING WITH VALIDATION
def preprocess_text_with_validation(doc: Dict) -> str:
    """Enhanced text preprocessing with validation"""
    title = doc.get('title', '').strip()
    abstract = doc.get('abstract', '').strip()
    
    # Validation
    if not abstract or len(abstract) < 10:
        raise ValueError(f"Invalid abstract for {doc.get('arxiv_id', 'unknown')}: too short")
    
    # Clean text
    # Remove LaTeX commands
    import re
    abstract = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', abstract)
    abstract = re.sub(r'\\[a-zA-Z]+', '', abstract)
    
    # Fix common encoding issues
    replacements = {
        '\u2019': "'",
        '\u2018': "'",
        '\u201c': '"',
        '\u201d': '"',
        '\u2013': '-',
        '\u2014': '--',
        '\u2026': '...',
    }
    for old, new in replacements.items():
        abstract = abstract.replace(old, new)
    
    # Combine title and abstract
    text = f"{title}\n\n{abstract}" if title else abstract
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Truncate if too long (approximate token limit)
    max_chars = 30000  # ~7500 tokens for most tokenizers
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
        
    return text


# 6. ADAPTIVE BATCH SIZE MANAGER
class AdaptiveBatchSizeManager:
    """Dynamically adjust batch size based on GPU memory usage"""
    
    def __init__(self, initial_batch_size: int, min_batch_size: int = 16, max_batch_size: int = 512):
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.memory_history = []
        self.history_size = 10
        
    def adjust_batch_size(self, gpu_memory_percent: float, had_oom: bool = False) -> int:
        """Adjust batch size based on GPU memory usage"""
        # Track memory history
        self.memory_history.append(gpu_memory_percent)
        if len(self.memory_history) > self.history_size:
            self.memory_history.pop(0)
            
        # If OOM occurred, reduce significantly
        if had_oom:
            self.current_batch_size = max(
                self.min_batch_size,
                self.current_batch_size // 2
            )
            logger.warning(f"OOM detected, reducing batch size to {self.current_batch_size}")
            return self.current_batch_size
            
        # Calculate average memory usage
        avg_memory = sum(self.memory_history) / len(self.memory_history)
        
        # Adjust based on memory usage
        if avg_memory > 90:
            # Reduce batch size
            self.current_batch_size = max(
                self.min_batch_size,
                int(self.current_batch_size * 0.8)
            )
            logger.info(f"High memory usage ({avg_memory:.1f}%), reducing batch size to {self.current_batch_size}")
            
        elif avg_memory < 50 and len(self.memory_history) >= self.history_size:
            # Increase batch size if consistently low
            self.current_batch_size = min(
                self.max_batch_size,
                int(self.current_batch_size * 1.2)
            )
            logger.info(f"Low memory usage ({avg_memory:.1f}%), increasing batch size to {self.current_batch_size}")
            
        return self.current_batch_size


# Example of how to integrate these improvements into the main pipeline:
"""
# In your main pipeline class:

def __init__(self, config: PipelineConfig):
    # ... existing initialization ...
    
    # Use improved completion tracker
    self.completion_tracker = ImprovedCompletionTracker(config.batch_size)
    
    # Use secure checkpoint manager
    self.checkpoint_manager = SecureCheckpointManager(config.checkpoint_dir)
    
    # Use robust database writer
    self.db_writer = RobustDatabaseWriter(
        config=self.config,
        write_queue=self.db_write_queue,
        stop_event=self.output_stop,
        checkpoint_manager=self.checkpoint_manager
    )
    
    # Add batch retry manager
    self.retry_manager = BatchRetryManager(
        gpu_queue=self.gpu_queue,
        checkpoint_manager=self.checkpoint_manager
    )
    
    # Add adaptive batch size manager for each GPU
    self.batch_size_managers = {
        gpu_id: AdaptiveBatchSizeManager(config.batch_size)
        for gpu_id in config.gpu_devices
    }

# In your stats tracking:
def _track_stats(self):
    # ... existing code ...
    
    # Track with document counts
    if event_type == 'processing':
        self.completion_tracker.start_batch(gpu_id, batch_id, doc_count)
    elif event_type == 'completed':
        self.completion_tracker.complete_batch(gpu_id, batch_id, doc_count)
    elif event_type == 'failed':
        self.completion_tracker.fail_batch(gpu_id, batch_id, doc_count)
"""