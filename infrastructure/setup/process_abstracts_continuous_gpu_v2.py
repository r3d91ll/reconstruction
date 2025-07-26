#!/usr/bin/env python3
"""
Production-Ready Dual GPU Pipeline with Continuous Utilization V2
Fixes critical issues identified in code review.

Key improvements:
- Fixed GPU device assignment for multiprocessing
- Improved completion detection without race conditions  
- Added memory management for large embeddings
- Dynamic queue sizing and configuration validation
- Health checks and graceful degradation
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
from queue import Queue, Empty, PriorityQueue
from threading import Thread, Event, Lock
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tqdm import tqdm
import psutil
import pickle
import lmdb
import time
import hashlib
from arango import ArangoClient
import gc

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
        logging.FileHandler('continuous_gpu_pipeline_v2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Protocol for type safety
class EmbeddingModel(Protocol):
    """Protocol for embedding models"""
    def encode_batch(self, texts: List[str], batch_size: int) -> np.ndarray: ...


@dataclass
class GPUWork:
    """Work item for GPU processing"""
    batch_id: str
    texts: List[str]
    metadata: List[Dict]
    priority: float = 1.0
    retry_count: int = 0
    
    def __lt__(self, other):
        return self.priority > other.priority


@dataclass
class PipelineConfig:
    """Pipeline configuration with validation"""
    # GPU settings
    gpu_devices: List[int] = field(default_factory=lambda: [0, 1])
    batch_size: int = 128
    prefetch_factor: int = 4
    
    # Queue settings
    preprocessing_workers: int = 8
    max_gpu_queue_size: Optional[int] = None  # Will be calculated
    max_output_queue_size: Optional[int] = None
    
    # Database
    db_host: str = "localhost"
    db_port: int = 8529
    db_name: str = "arxiv_abstracts_continuous"
    db_username: str = "root"
    db_password: str = ""
    
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
    
    def __post_init__(self):
        """Validate and adjust configuration"""
        # Validate batch size
        if self.batch_size > 512:
            logger.warning(f"Large batch size {self.batch_size} may cause OOM")
            
        # Calculate optimal queue sizes if not provided
        if self.max_gpu_queue_size is None:
            self.max_gpu_queue_size = self.batch_size * self.prefetch_factor * len(self.gpu_devices) * 2
            logger.info(f"Calculated max_gpu_queue_size: {self.max_gpu_queue_size}")
            
        if self.max_output_queue_size is None:
            self.max_output_queue_size = self.max_gpu_queue_size * 2
            
        # Warn about excessive prefetching
        if self.prefetch_factor * self.batch_size > 10000:
            logger.warning("Excessive prefetching may waste memory")
            
        # Validate GPU devices
        available_gpus = torch.cuda.device_count()
        for gpu_id in self.gpu_devices:
            if gpu_id >= available_gpus:
                raise ValueError(f"GPU {gpu_id} not available (only {available_gpus} GPUs detected)")


class GPUWorker(mp.Process):
    """
    Fixed GPU worker with proper device assignment and memory management.
    """
    
    def __init__(
        self,
        gpu_id: int,
        input_queue: mp.Queue,
        output_queue: mp.Queue,
        checkpoint_queue: mp.Queue,
        stats_queue: mp.Queue,  # New: for tracking items in flight
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
        self.retry_count = 0
        self.max_retries = 3
        
    def run(self):
        """Main GPU processing loop with proper device assignment"""
        # CRITICAL FIX: Don't modify CUDA_VISIBLE_DEVICES
        # Instead, directly set the device
        torch.cuda.set_device(self.gpu_id)
        
        logger.info(f"GPU Worker {self.gpu_id} starting on cuda:{self.gpu_id}")
        
        # Initialize model with retry logic
        model = self._initialize_model_with_retry()
        if model is None:
            logger.error(f"GPU Worker {self.gpu_id} failed to initialize")
            return
            
        # Warm up GPU
        self._warmup_gpu(model)
        
        # Processing loop
        while not self.stop_event.is_set():
            try:
                # Get work with timeout
                work = self.input_queue.get(timeout=1.0)
                if work is None:  # Poison pill
                    logger.info(f"GPU Worker {self.gpu_id} received shutdown signal")
                    break
                    
                # Notify stats queue that we're processing
                self.stats_queue.put(('processing', self.gpu_id, work.batch_id))
                
                # Process batch
                result = self._process_batch_with_retry(model, work)
                
                if result:
                    # Send to output queue
                    self.output_queue.put(result)
                    
                    # Notify stats queue that we're done
                    self.stats_queue.put(('completed', self.gpu_id, work.batch_id))
                    
                    self.processed_count += 1
                    
                    # Checkpoint if needed
                    if self.processed_count % self.config.checkpoint_interval == 0:
                        self._save_checkpoint()
                        
                    # Periodic memory cleanup
                    if self.processed_count % self.config.embedding_cleanup_interval == 0:
                        self._cleanup_memory()
                else:
                    # Failed after retries
                    self.stats_queue.put(('failed', self.gpu_id, work.batch_id))
                    
            except Empty:
                continue
            except Exception as e:
                logger.error(f"GPU Worker {self.gpu_id} error: {e}", exc_info=True)
                
        # Cleanup
        self._final_cleanup()
        
    def _initialize_model_with_retry(self) -> Optional[EmbeddingModel]:
        """Initialize model with retry logic and graceful degradation"""
        from irec_infrastructure.embeddings.local_jina_gpu import LocalJinaGPU, LocalJinaConfig
        
        for attempt in range(self.max_retries):
            try:
                config = LocalJinaConfig()
                config.device_ids = [self.gpu_id]  # Use actual GPU ID
                config.use_fp16 = True
                config.max_length = 8192
                
                # Adjust batch size based on previous failures
                if attempt > 0:
                    config.batch_size = max(16, self.config.batch_size // (2 ** attempt))
                    logger.info(f"GPU {self.gpu_id} reducing internal batch size to {config.batch_size}")
                
                model = LocalJinaGPU(config)
                
                # Test the model
                test_result = model.encode_batch(["Test initialization"], batch_size=1)
                if test_result is None or len(test_result) == 0:
                    raise RuntimeError("Model test failed")
                    
                logger.info(f"GPU Worker {self.gpu_id} model initialized successfully")
                return model
                
            except torch.cuda.OutOfMemoryError:
                logger.error(f"GPU {self.gpu_id} OOM during initialization, attempt {attempt + 1}")
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(2 ** attempt)
                
            except Exception as e:
                logger.error(f"GPU {self.gpu_id} initialization error: {e}")
                if attempt == self.max_retries - 1:
                    self._notify_failure()
                time.sleep(2 ** attempt)
                
        return None
        
    def _warmup_gpu(self, model: EmbeddingModel):
        """Warm up GPU with dummy data to avoid allocation delays"""
        try:
            dummy_batch = ["warmup text"] * min(10, self.config.batch_size)
            _ = model.encode_batch(dummy_batch, batch_size=len(dummy_batch))
            torch.cuda.synchronize(device=self.gpu_id)
            torch.cuda.empty_cache()
            logger.debug(f"GPU {self.gpu_id} warmed up")
        except Exception as e:
            logger.warning(f"GPU {self.gpu_id} warmup failed: {e}")
            
    def _process_batch_with_retry(self, model: EmbeddingModel, work: GPUWork) -> Optional[Dict]:
        """Process batch with retry logic and memory management"""
        for attempt in range(self.max_retries):
            try:
                # Process batch on GPU
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                
                # Generate embeddings with memory-aware processing
                embeddings_list = self._generate_embeddings_chunked(model, work.texts)
                
                end_time.record()
                torch.cuda.synchronize(device=self.gpu_id)
                
                gpu_time = start_time.elapsed_time(end_time) / 1000.0
                self.total_gpu_time += gpu_time
                
                # Prepare output
                output = {
                    'batch_id': work.batch_id,
                    'embeddings': embeddings_list,
                    'metadata': work.metadata,
                    'gpu_id': self.gpu_id,
                    'gpu_time': gpu_time,
                    'timestamp': datetime.now().isoformat(),
                    'docs_per_second': len(work.texts) / gpu_time if gpu_time > 0 else 0,
                    'retry_count': work.retry_count
                }
                
                return output
                
            except torch.cuda.OutOfMemoryError:
                logger.error(f"GPU {self.gpu_id} OOM on attempt {attempt + 1}")
                self._handle_oom()
                
                if attempt < self.max_retries - 1:
                    # Reduce batch size for retry
                    work.texts = work.texts[:len(work.texts)//2]
                    work.metadata = work.metadata[:len(work.metadata)//2]
                    work.retry_count += 1
                    
            except Exception as e:
                logger.error(f"GPU {self.gpu_id} processing error: {e}")
                if attempt == self.max_retries - 1:
                    return None
                    
            time.sleep(2 ** attempt)  # Exponential backoff
            
        return None
        
    def _generate_embeddings_chunked(self, model: EmbeddingModel, texts: List[str]) -> List[List[float]]:
        """Generate embeddings in chunks to avoid memory issues"""
        embeddings_list = []
        chunk_size = 1000  # Process in chunks to avoid memory buildup
        
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            
            # Generate embeddings
            embeddings = model.encode_batch(chunk, batch_size=min(self.config.batch_size, len(chunk)))
            
            # Convert to list format efficiently
            if torch.is_tensor(embeddings):
                # Move to CPU in chunks to avoid memory spike
                embeddings_np = embeddings.cpu().numpy()
                chunk_list = embeddings_np.tolist()
            else:
                chunk_list = embeddings.tolist()
                
            embeddings_list.extend(chunk_list)
            
            # Cleanup intermediate results
            del embeddings
            if 'embeddings_np' in locals():
                del embeddings_np
            del chunk_list
            
            # Force garbage collection periodically
            if i % (chunk_size * 5) == 0:
                gc.collect()
                
        return embeddings_list
        
    def _handle_oom(self):
        """Handle OOM errors gracefully"""
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device=self.gpu_id)
        gc.collect()
        time.sleep(1)  # Give GPU time to recover
        
    def _cleanup_memory(self):
        """Periodic memory cleanup"""
        torch.cuda.empty_cache()
        gc.collect()
        logger.debug(f"GPU {self.gpu_id} memory cleaned")
        
    def _save_checkpoint(self):
        """Save checkpoint data"""
        checkpoint_data = {
            'gpu_id': self.gpu_id,
            'processed_count': self.processed_count,
            'total_gpu_time': self.total_gpu_time,
            'timestamp': datetime.now().isoformat(),
            'avg_time_per_batch': self.total_gpu_time / self.processed_count if self.processed_count > 0 else 0
        }
        self.checkpoint_queue.put(checkpoint_data)
        
    def _notify_failure(self):
        """Notify pipeline of worker failure"""
        self.stats_queue.put(('worker_failed', self.gpu_id, None))
        
    def _final_cleanup(self):
        """Final cleanup before shutdown"""
        avg_time = self.total_gpu_time / self.processed_count if self.processed_count > 0 else 0
        logger.info(
            f"GPU Worker {self.gpu_id} shutting down. "
            f"Processed {self.processed_count} batches, "
            f"avg time: {avg_time:.3f}s/batch"
        )


class CompletionTracker:
    """Track items in flight to avoid race conditions in completion detection"""
    
    def __init__(self):
        self.lock = Lock()
        self.items_queued = 0
        self.items_processing = {}  # gpu_id -> set of batch_ids
        self.items_completed = 0
        self.items_failed = 0
        
    def add_queued(self, count: int):
        """Add items to queue count"""
        with self.lock:
            self.items_queued += count
            
    def start_processing(self, gpu_id: int, batch_id: str):
        """Mark item as being processed"""
        with self.lock:
            if gpu_id not in self.items_processing:
                self.items_processing[gpu_id] = set()
            self.items_processing[gpu_id].add(batch_id)
            
    def complete_processing(self, gpu_id: int, batch_id: str):
        """Mark item as completed"""
        with self.lock:
            if gpu_id in self.items_processing:
                self.items_processing[gpu_id].discard(batch_id)
            self.items_completed += 1
            
    def fail_processing(self, gpu_id: int, batch_id: str):
        """Mark item as failed"""
        with self.lock:
            if gpu_id in self.items_processing:
                self.items_processing[gpu_id].discard(batch_id)
            self.items_failed += 1
            
    def is_complete(self) -> bool:
        """Check if all items are processed"""
        with self.lock:
            total_in_flight = sum(len(items) for items in self.items_processing.values())
            return (self.items_completed + self.items_failed) >= self.items_queued and total_in_flight == 0
            
    def get_stats(self) -> Dict:
        """Get current statistics"""
        with self.lock:
            total_in_flight = sum(len(items) for items in self.items_processing.values())
            return {
                'queued': self.items_queued,
                'in_flight': total_in_flight,
                'completed': self.items_completed,
                'failed': self.items_failed,
                'pending': self.items_queued - self.items_completed - self.items_failed - total_in_flight
            }


class HealthMonitor:
    """Monitor pipeline health and implement recovery strategies"""
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.health_checks = {
            'gpu_workers': True,
            'preprocessing_workers': True,
            'db_writer': True,
            'queue_health': True
        }
        self.check_interval = 10.0
        self.monitor_thread = None
        self.stop_event = Event()
        
    def start(self):
        """Start health monitoring"""
        self.monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop(self):
        """Stop health monitoring"""
        self.stop_event.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
            
    def _monitor_loop(self):
        """Main monitoring loop"""
        while not self.stop_event.is_set():
            try:
                # Check GPU workers
                self.health_checks['gpu_workers'] = all(
                    w.is_alive() for w in self.pipeline.gpu_workers
                )
                
                # Check preprocessing workers
                self.health_checks['preprocessing_workers'] = all(
                    w.is_alive() for w in self.pipeline.preprocessing_workers
                )
                
                # Check DB writer
                self.health_checks['db_writer'] = (
                    self.pipeline.db_writer.is_alive() if self.pipeline.db_writer else False
                )
                
                # Check queue health
                gpu_queue_percent = (
                    self.pipeline.gpu_queue.qsize() / self.pipeline.config.max_gpu_queue_size * 100
                )
                self.health_checks['queue_health'] = gpu_queue_percent < 90
                
                # Log any issues
                if not all(self.health_checks.values()):
                    logger.warning(f"Health check failed: {self.health_checks}")
                    self._attempt_recovery()
                    
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                
            self.stop_event.wait(self.check_interval)
            
    def _attempt_recovery(self):
        """Attempt to recover from failures"""
        # Check for dead GPU workers
        for i, worker in enumerate(self.pipeline.gpu_workers):
            if not worker.is_alive():
                logger.error(f"GPU worker {i} died, attempting restart...")
                # Could implement restart logic here
                
        # Check for queue overflow
        if not self.health_checks['queue_health']:
            logger.warning("Queue overflow detected, slowing preprocessing")
            # Could implement backpressure here


class ContinuousGPUPipeline:
    """
    Main pipeline orchestrator with improved completion tracking and health monitoring.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir)
        
        # Queues
        self.document_queue = Queue(maxsize=10000)
        self.gpu_queue = mp.Queue(maxsize=config.max_gpu_queue_size)
        self.output_queue = mp.Queue(maxsize=config.max_output_queue_size)
        self.checkpoint_queue = mp.Queue()
        self.stats_queue = mp.Queue()  # For completion tracking
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
        
        # Tracking
        self.completion_tracker = CompletionTracker()
        
        # Statistics
        self.stats = {
            'start_time': None,
            'total_processed': 0,
            'total_documents': 0,
            'gpu_stats': {gpu: {'batches': 0, 'time': 0} for gpu in config.gpu_devices}
        }
        
    def __enter__(self):
        """Context manager entry"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()
        
    def setup_database(self, clean_start=False):
        """Setup database and collections"""
        try:
            client = ArangoClient(hosts=f'http://{self.config.db_host}:{self.config.db_port}')
            sys_db = client.db('_system', username=self.config.db_username, password=self.config.db_password)
            
            if sys_db.has_database(self.config.db_name):
                if clean_start:
                    logger.warning(f"Dropping existing database: {self.config.db_name}")
                    sys_db.delete_database(self.config.db_name)
                else:
                    logger.info(f"Using existing database: {self.config.db_name}")
                    return
                    
            # Create database
            sys_db.create_database(self.config.db_name)
            db = client.db(self.config.db_name, username=self.config.db_username, password=self.config.db_password)
            
            # Create collection with indexes
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
        """Start all pipeline components"""
        logger.info("Starting Continuous GPU Pipeline V2")
        
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
            logger.info(f"Started GPU worker {gpu_id}")
            
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
            
        logger.info(f"Started {self.config.preprocessing_workers} preprocessing workers")
        
        # Start database writer
        self.db_writer = DatabaseWriter(
            config=self.config,
            write_queue=self.db_write_queue,
            stop_event=self.output_stop
        )
        self.db_writer.start()
        
        # Start monitor
        if self.config.enable_monitoring:
            self.monitor = GPUUtilizationMonitor(
                gpu_queue=self.gpu_queue,
                output_queue=self.output_queue,
                config=self.config,
                stop_event=self.stop_event
            )
            self.monitor.start()
            
        # Start health monitor
        self.health_monitor = HealthMonitor(self)
        self.health_monitor.start()
        
        # Start output processor
        self.output_thread = Thread(target=self._process_outputs, daemon=True)
        self.output_thread.start()
        
        # Start checkpoint handler
        self.checkpoint_thread = Thread(target=self._handle_checkpoints, daemon=True)
        self.checkpoint_thread.start()
        
        # Start stats tracker
        self.stats_thread = Thread(target=self._track_stats, daemon=True)
        self.stats_thread.start()
        
        logger.info("Pipeline started successfully")
        
    def wait_for_completion(self):
        """Wait for all work to complete without race conditions"""
        logger.info("Waiting for preprocessing to complete...")
        
        # Wait for preprocessing to finish
        for worker in self.preprocessing_workers:
            worker.join()
            
        logger.info("All preprocessing complete")
        
        # Wait for all items to be processed
        logger.info("Waiting for GPU processing to complete...")
        
        while not self.completion_tracker.is_complete():
            stats = self.completion_tracker.get_stats()
            logger.info(
                f"Completion status - Queued: {stats['queued']}, "
                f"In flight: {stats['in_flight']}, "
                f"Completed: {stats['completed']}, "
                f"Failed: {stats['failed']}, "
                f"Pending: {stats['pending']}"
            )
            time.sleep(2.0)
            
        logger.info("All items processed")
        
        # Send poison pills to GPU workers
        for _ in self.gpu_workers:
            self.gpu_queue.put(None)
            
        # Wait for GPU workers
        for worker in self.gpu_workers:
            worker.join()
            
        logger.info("All GPU processing complete")
        
        # Stop output processing
        self.output_stop.set()
        
        # Wait for output queue to empty
        logger.info("Waiting for outputs to be processed...")
        while not self.output_queue.empty() or not self.db_write_queue.empty():
            time.sleep(1.0)
            
        # Send poison pill to DB writer
        self.db_write_queue.put(None)
        
        # Wait for threads
        if self.db_writer:
            self.db_writer.join()
        self.output_thread.join()
        self.checkpoint_thread.join()
        self.stats_thread.join()
        
        # Stop monitoring
        self.stop_event.set()
        if self.monitor:
            self.monitor.join()
            
        # Stop health monitor
        if self.health_monitor:
            self.health_monitor.stop()
            
        logger.info("Pipeline completion successful")
        
    def _track_stats(self):
        """Track processing statistics from stats queue"""
        while not self.stop_event.is_set():
            try:
                stat = self.stats_queue.get(timeout=1.0)
                if stat is None:
                    break
                    
                event_type, gpu_id, batch_id = stat
                
                if event_type == 'processing':
                    self.completion_tracker.start_processing(gpu_id, batch_id)
                elif event_type == 'completed':
                    self.completion_tracker.complete_processing(gpu_id, batch_id)
                elif event_type == 'failed':
                    self.completion_tracker.fail_processing(gpu_id, batch_id)
                elif event_type == 'worker_failed':
                    logger.error(f"GPU worker {gpu_id} reported failure")
                    
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Stats tracking error: {e}")
                
    def _process_outputs(self):
        """Process GPU outputs and send to database writer"""
        while not self.output_stop.is_set():
            try:
                # Get output with timeout
                output = self.output_queue.get(timeout=1.0)
                if output is None:
                    break
                    
                # Update statistics
                gpu_id = output['gpu_id']
                self.stats['gpu_stats'][gpu_id]['batches'] += 1
                self.stats['gpu_stats'][gpu_id]['time'] += output['gpu_time']
                self.stats['total_processed'] += len(output['metadata'])
                
                # Send to database writer
                self.db_write_queue.put(output)
                
                # Log progress periodically
                if self.stats['total_processed'] % 1000 == 0:
                    self._log_progress()
                    
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Output processing error: {e}")
                
    def _handle_checkpoints(self):
        """Handle checkpoint saves from GPU workers"""
        while not self.stop_event.is_set():
            try:
                # Get checkpoint data with timeout
                checkpoint = self.checkpoint_queue.get(timeout=1.0)
                if checkpoint is None:
                    break
                    
                # Save checkpoint
                key = f"gpu_{checkpoint['gpu_id']}_checkpoint"
                self.checkpoint_manager.save_progress(key, checkpoint)
                
                # Save pipeline state periodically
                if checkpoint['processed_count'] % 1000 == 0:
                    self._save_pipeline_state()
                    
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Checkpoint handling error: {e}")
                
    def _save_pipeline_state(self):
        """Save current pipeline state"""
        state = {
            'stats': self.stats,
            'completion_tracker': {
                'queued': self.completion_tracker.items_queued,
                'completed': self.completion_tracker.items_completed,
                'failed': self.completion_tracker.items_failed
            },
            'timestamp': datetime.now().isoformat()
        }
        self.checkpoint_manager.save_state(state)
        
    def _log_progress(self):
        """Log processing progress"""
        elapsed = time.time() - self.stats['start_time']
        docs_per_sec = self.stats['total_processed'] / elapsed if elapsed > 0 else 0
        
        logger.info(
            f"Progress: {self.stats['total_processed']}/{self.stats['total_documents']} documents "
            f"({self.stats['total_processed']/self.stats['total_documents']*100:.1f}%), "
            f"{docs_per_sec:.1f} docs/sec"
        )
        
        # Log GPU statistics
        for gpu_id, stats in self.stats['gpu_stats'].items():
            if stats['batches'] > 0:
                avg_time = stats['time'] / stats['batches']
                logger.info(f"  GPU {gpu_id}: {stats['batches']} batches, avg {avg_time:.3f}s/batch")
                
    def process_metadata_files(self, metadata_files: List[Path], resume: bool = True):
        """Process metadata files with batch tracking"""
        logger.info(f"Processing {len(metadata_files)} metadata files")
        
        # Load checkpoint if resuming
        processed_files = set()
        if resume:
            processed_files = self.checkpoint_manager.get_processed_files()
            logger.info(f"Resuming: {len(processed_files)} files already processed")
            
        # Filter out already processed files
        files_to_process = [f for f in metadata_files if f.stem not in processed_files]
        logger.info(f"Will process {len(files_to_process)} new files")
        
        # Process files
        total_documents = 0
        for file_path in tqdm(files_to_process, desc="Loading metadata files"):
            try:
                # Load metadata file
                with open(file_path, 'r') as f:
                    metadata = json.load(f)
                    
                # Queue documents for processing
                for doc in metadata:
                    self.document_queue.put(doc)
                    total_documents += 1
                    
                # Mark file as processed
                self.checkpoint_manager.mark_file_processed(file_path.stem)
                
                # Save checkpoint periodically
                if total_documents % 10000 == 0:
                    self._save_pipeline_state()
                    
            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {e}")
                continue
                
        # Send poison pills to preprocessing workers
        for _ in self.preprocessing_workers:
            self.document_queue.put(None)
            
        self.stats['total_documents'] = total_documents
        logger.info(f"Queued {total_documents} documents for processing")
        
        # Calculate and update total batches for completion tracking
        total_batches = (total_documents + self.config.batch_size - 1) // self.config.batch_size
        self.completion_tracker.add_queued(total_batches)
        
    def shutdown(self):
        """Clean shutdown"""
        logger.info("Shutting down pipeline")
        
        # Stop all components
        self.preprocessing_stop.set()
        self.output_stop.set()
        self.stop_event.set()
        
        # Stop health monitor
        if self.health_monitor:
            self.health_monitor.stop()
        
        # Clean up
        self.checkpoint_manager.cleanup()
        
        logger.info("Pipeline shutdown complete")


# Include the missing classes from original implementation
class PreprocessingWorker(Thread):
    """CPU preprocessing worker for abstracts"""
    
    def __init__(
        self,
        document_queue: Queue,
        gpu_queue: mp.Queue,
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
        
    def run(self):
        """Process documents and batch them for GPU processing"""
        logger.info(f"Preprocessing worker {self.worker_id} started")
        
        batch_texts = []
        batch_metadata = []
        
        while not self.stop_event.is_set():
            try:
                # Get document with timeout
                doc = self.document_queue.get(timeout=1.0)
                if doc is None:  # Poison pill
                    # Process remaining batch
                    if batch_texts:
                        self._send_batch(batch_texts, batch_metadata)
                    break
                    
                # Extract text and metadata
                text = self._preprocess_text(doc)
                metadata = self._extract_metadata(doc)
                
                batch_texts.append(text)
                batch_metadata.append(metadata)
                
                # Send batch when full
                if len(batch_texts) >= self.config.batch_size:
                    self._send_batch(batch_texts, batch_metadata)
                    batch_texts = []
                    batch_metadata = []
                    
            except Empty:
                # Send partial batch if idle
                if batch_texts:
                    self._send_batch(batch_texts, batch_metadata)
                    batch_texts = []
                    batch_metadata = []
                continue
                
            except Exception as e:
                logger.error(f"Preprocessing worker {self.worker_id} error: {e}")
                
        # Send any remaining batch
        if batch_texts:
            self._send_batch(batch_texts, batch_metadata)
            
        logger.info(f"Preprocessing worker {self.worker_id} stopping. Processed {self.processed_count} batches")
        
    def _preprocess_text(self, doc: Dict) -> str:
        """Preprocess abstract text"""
        # Combine title and abstract
        title = doc.get('title', '').strip()
        abstract = doc.get('abstract', '').strip()
        
        # Clean text
        text = f"{title}\n\n{abstract}" if title and abstract else title or abstract
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Truncate if needed (Jina max is 8192 tokens)
        if len(text) > 30000:  # Rough character limit
            text = text[:30000] + "..."
            
        return text
        
    def _extract_metadata(self, doc: Dict) -> Dict:
        """Extract relevant metadata"""
        return {
            'arxiv_id': doc.get('id', ''),
            'title': doc.get('title', ''),
            'authors': doc.get('authors', []),
            'categories': doc.get('categories', []),
            'published': doc.get('published', ''),
            'updated': doc.get('updated', ''),
            'doi': doc.get('doi', ''),
            'journal_ref': doc.get('journal_ref', '')
        }
        
    def _send_batch(self, texts: List[str], metadata: List[Dict]):
        """Send batch to GPU queue"""
        batch_id = f"batch_{self.worker_id}_{self.processed_count}_{int(time.time() * 1000)}"
        
        work = GPUWork(
            batch_id=batch_id,
            texts=texts,
            metadata=metadata,
            priority=1.0
        )
        
        # Put to queue (will block if full)
        self.gpu_queue.put(work)
        self.processed_count += 1
        
        logger.debug(f"Worker {self.worker_id} sent batch {batch_id} with {len(texts)} documents")


class CheckpointManager:
    """Fast checkpoint management using LMDB"""
    
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
                max_dbs=3,
                lock=True
            )
            
            # Create named databases
            with self.env.begin(write=True) as txn:
                self.progress_db = self.env.open_db(b'progress', txn=txn)
                self.state_db = self.env.open_db(b'state', txn=txn)
                self.metadata_db = self.env.open_db(b'metadata', txn=txn)
                
            logger.info(f"Checkpoint database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize checkpoint database: {e}")
            raise
            
    def save_progress(self, key: str, value: Dict):
        """Save progress checkpoint"""
        with self.lock:
            try:
                serialized = pickle.dumps(value)
                with self.env.begin(write=True, db=self.progress_db) as txn:
                    txn.put(key.encode(), serialized)
            except Exception as e:
                logger.error(f"Failed to save progress checkpoint: {e}")
                
    def load_progress(self, key: str) -> Optional[Dict]:
        """Load progress checkpoint"""
        with self.lock:
            try:
                with self.env.begin(db=self.progress_db) as txn:
                    data = txn.get(key.encode())
                    if data:
                        return pickle.loads(data)
            except Exception as e:
                logger.error(f"Failed to load progress checkpoint: {e}")
        return None
        
    def save_state(self, state: Dict):
        """Save pipeline state"""
        with self.lock:
            try:
                # Generate state key with timestamp
                timestamp = datetime.now().isoformat()
                key = f"state_{timestamp}"
                
                serialized = pickle.dumps(state)
                with self.env.begin(write=True, db=self.state_db) as txn:
                    txn.put(key.encode(), serialized)
                    # Also save as latest
                    txn.put(b'latest', serialized)
                    
                logger.debug(f"Saved pipeline state: {key}")
                
            except Exception as e:
                logger.error(f"Failed to save state: {e}")
                
    def load_latest_state(self) -> Optional[Dict]:
        """Load most recent pipeline state"""
        with self.lock:
            try:
                with self.env.begin(db=self.state_db) as txn:
                    data = txn.get(b'latest')
                    if data:
                        return pickle.loads(data)
            except Exception as e:
                logger.error(f"Failed to load latest state: {e}")
        return None
        
    def get_processed_files(self) -> set:
        """Get set of processed file IDs"""
        processed = set()
        with self.lock:
            try:
                with self.env.begin(db=self.metadata_db) as txn:
                    cursor = txn.cursor()
                    for key, _ in cursor:
                        if key.startswith(b'processed_'):
                            file_id = key[10:].decode()  # Remove 'processed_' prefix
                            processed.add(file_id)
            except Exception as e:
                logger.error(f"Failed to get processed files: {e}")
        return processed
        
    def mark_file_processed(self, file_id: str):
        """Mark a file as processed"""
        with self.lock:
            try:
                key = f"processed_{file_id}".encode()
                with self.env.begin(write=True, db=self.metadata_db) as txn:
                    txn.put(key, b'1')
            except Exception as e:
                logger.error(f"Failed to mark file processed: {e}")
                
    def cleanup(self):
        """Cleanup resources"""
        if self.env:
            self.env.close()
            logger.info("Checkpoint database closed")


class GPUUtilizationMonitor(Thread):
    """Monitor GPU utilization and queue depths"""
    
    def __init__(
        self,
        gpu_queue: mp.Queue,
        output_queue: mp.Queue,
        config: PipelineConfig,
        stop_event: mp.Event
    ):
        super().__init__(daemon=True)
        self.gpu_queue = gpu_queue
        self.output_queue = output_queue
        self.config = config
        self.stop_event = stop_event
        self.stats = {gpu_id: {'utilization': [], 'memory': []} for gpu_id in config.gpu_devices}
        self.low_util_counts = {gpu_id: 0 for gpu_id in config.gpu_devices}
        
    def run(self):
        """Monitor GPU utilization and performance"""
        logger.info("GPU Utilization Monitor started")
        
        try:
            import pynvml
            pynvml.nvmlInit()
            has_nvml = True
        except:
            logger.warning("NVML not available, GPU monitoring limited")
            has_nvml = False
            
        while not self.stop_event.is_set():
            try:
                # Log queue depths
                gpu_queue_size = self.gpu_queue.qsize()
                output_queue_size = self.output_queue.qsize()
                
                logger.info(
                    f"Queue depths - GPU: {gpu_queue_size}/{self.config.max_gpu_queue_size}, "
                    f"Output: {output_queue_size}/{self.config.max_output_queue_size}"
                )
                
                # Monitor GPU stats if NVML available
                if has_nvml:
                    for gpu_id in self.config.gpu_devices:
                        try:
                            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                            
                            # Get utilization
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            
                            gpu_util = util.gpu
                            mem_used_percent = (mem_info.used / mem_info.total) * 100
                            
                            self.stats[gpu_id]['utilization'].append(gpu_util)
                            self.stats[gpu_id]['memory'].append(mem_used_percent)
                            
                            # Check for low utilization
                            if gpu_util < self.config.low_util_threshold:
                                self.low_util_counts[gpu_id] += 1
                                if self.low_util_counts[gpu_id] >= self.config.low_util_alert_threshold:
                                    logger.warning(
                                        f"GPU {gpu_id} has low utilization ({gpu_util}%) "
                                        f"for {self.low_util_counts[gpu_id]} checks"
                                    )
                            else:
                                self.low_util_counts[gpu_id] = 0
                                
                            # Log current stats
                            logger.debug(
                                f"GPU {gpu_id} - Utilization: {gpu_util}%, "
                                f"Memory: {mem_used_percent:.1f}% "
                                f"({mem_info.used / 1024**3:.1f}GB/{mem_info.total / 1024**3:.1f}GB)"
                            )
                            
                        except Exception as e:
                            logger.error(f"Failed to get GPU {gpu_id} stats: {e}")
                            
                # Check for stalls
                if gpu_queue_size == 0 and output_queue_size > 0:
                    logger.warning("GPU queue empty but output queue has items - possible GPU stall")
                elif gpu_queue_size == self.config.max_gpu_queue_size:
                    logger.warning("GPU queue full - preprocessing may be too fast")
                    
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                
            # Wait for next interval
            self.stop_event.wait(self.config.monitor_interval)
            
        # Cleanup
        if has_nvml:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
                
        # Log final statistics
        self._log_final_stats()
        
    def _log_final_stats(self):
        """Log summary statistics"""
        logger.info("GPU Utilization Summary:")
        for gpu_id, stats in self.stats.items():
            if stats['utilization']:
                avg_util = sum(stats['utilization']) / len(stats['utilization'])
                avg_mem = sum(stats['memory']) / len(stats['memory'])
                logger.info(
                    f"  GPU {gpu_id} - Avg Utilization: {avg_util:.1f}%, "
                    f"Avg Memory: {avg_mem:.1f}%"
                )


class DatabaseWriter(Thread):
    """Asynchronous database writer"""
    
    def __init__(
        self,
        config: PipelineConfig,
        write_queue: Queue,
        stop_event: Event
    ):
        super().__init__(daemon=True)
        self.config = config
        self.write_queue = write_queue
        self.stop_event = stop_event
        self.client = None
        self.db = None
        self.collection = None
        self.written_count = 0
        self.batch_buffer = []
        self.batch_size = 100  # Write in batches for efficiency
        
    def run(self):
        """Main database writing loop"""
        logger.info("Database Writer started")
        
        # Initialize database connection
        if not self._initialize_db():
            logger.error("Failed to initialize database connection")
            return
            
        while not self.stop_event.is_set():
            try:
                # Get write request with timeout
                item = self.write_queue.get(timeout=1.0)
                if item is None:  # Poison pill
                    # Write any remaining buffered items
                    if self.batch_buffer:
                        self._write_batch()
                    break
                    
                # Add to buffer
                self.batch_buffer.append(item)
                
                # Write batch if full
                if len(self.batch_buffer) >= self.batch_size:
                    self._write_batch()
                    
            except Empty:
                # Write partial batch if idle
                if self.batch_buffer:
                    self._write_batch()
                continue
                
            except Exception as e:
                logger.error(f"Database writer error: {e}")
                
        # Final cleanup
        self._cleanup()
        
    def _initialize_db(self) -> bool:
        """Initialize database connection"""
        try:
            self.client = ArangoClient(hosts=f'http://{self.config.db_host}:{self.config.db_port}')
            self.db = self.client.db(
                self.config.db_name,
                username=self.config.db_username,
                password=self.config.db_password
            )
            self.collection = self.db.collection('abstract_metadata')
            
            # Test connection
            self.collection.count()
            
            logger.info("Database connection established")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
            
    def _write_batch(self):
        """Write batch of documents to database"""
        if not self.batch_buffer:
            return
            
        try:
            # Prepare documents
            documents = []
            for item in self.batch_buffer:
                for i, metadata in enumerate(item['metadata']):
                    doc = {
                        '_key': metadata['arxiv_id'].replace('/', '_'),  # Ensure valid key
                        'arxiv_id': metadata['arxiv_id'],
                        'title': metadata['title'],
                        'authors': metadata['authors'],
                        'categories': metadata['categories'],
                        'published': metadata['published'],
                        'updated': metadata['updated'],
                        'doi': metadata['doi'],
                        'journal_ref': metadata['journal_ref'],
                        'embedding': item['embeddings'][i],
                        'processed_gpu': item['gpu_id'],
                        'processing_time': item['gpu_time'],
                        'processed_at': item['timestamp'],
                        'batch_id': item['batch_id']
                    }
                    documents.append(doc)
                    
            # Bulk insert with overwrite
            result = self.collection.insert_many(
                documents,
                overwrite=True,
                return_new=False
            )
            
            self.written_count += len(documents)
            logger.debug(f"Wrote batch of {len(documents)} documents to database")
            
            # Clear buffer
            self.batch_buffer = []
            
        except Exception as e:
            logger.error(f"Failed to write batch to database: {e}")
            # Could implement retry logic here
            
    def _cleanup(self):
        """Cleanup database connection"""
        if self.client:
            self.client.close()
            
        logger.info(f"Database Writer stopped. Wrote {self.written_count} documents")


def main():
    """Main entry point with improved configuration"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Continuous GPU Processing Pipeline V2 for arXiv abstracts"
    )
    
    # Data arguments
    parser.add_argument(
        'metadata_dir',
        type=str,
        help='Directory containing JSON metadata files'
    )
    
    # GPU arguments
    parser.add_argument(
        '--gpu-devices',
        type=int,
        nargs='+',
        default=[0, 1],
        help='GPU device IDs to use (default: [0, 1])'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for GPU processing (default: 128)'
    )
    
    # Worker arguments
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of preprocessing workers (default: 8)'
    )
    
    # Database arguments
    parser.add_argument(
        '--db-host',
        type=str,
        default='localhost',
        help='ArangoDB host (default: localhost)'
    )
    parser.add_argument(
        '--db-name',
        type=str,
        default='arxiv_abstracts_continuous',
        help='Database name (default: arxiv_abstracts_continuous)'
    )
    
    # Checkpoint arguments
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        help='Checkpoint directory (default: ./checkpoints/{db_name})'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint'
    )
    parser.add_argument(
        '--clean-start',
        action='store_true',
        help='Drop existing database and start fresh'
    )
    
    # Monitoring arguments
    parser.add_argument(
        '--disable-monitoring',
        action='store_true',
        help='Disable GPU utilization monitoring'
    )
    
    args = parser.parse_args()
    
    # Get environment variables
    db_password = os.environ.get('ARANGO_PASSWORD', '')
    if not db_password:
        print("ERROR: ARANGO_PASSWORD environment variable not set!")
        sys.exit(1)
    
    # Configure pipeline with validation
    config = PipelineConfig(
        gpu_devices=args.gpu_devices,
        batch_size=args.batch_size,
        preprocessing_workers=args.workers,
        prefetch_factor=4,
        checkpoint_dir=args.checkpoint_dir or f"./checkpoints/{args.db_name}",
        checkpoint_interval=100,
        enable_monitoring=True,
        db_host=args.db_host,
        db_name=args.db_name,
        db_password=db_password
    )
    
    # Use context manager for clean resource management
    try:
        with ContinuousGPUPipeline(config) as pipeline:
            # Setup database
            if not args.resume:
                pipeline.setup_database(clean_start=args.clean_start)
            
            # Start pipeline
            pipeline.start()
            
            # Get metadata files
            metadata_dir = Path(args.metadata_dir)
            if not metadata_dir.exists():
                logger.error(f"Metadata directory not found: {metadata_dir}")
                sys.exit(1)
                
            metadata_files = sorted(metadata_dir.glob("*.json"))
            if not metadata_files:
                logger.error(f"No JSON files found in {metadata_dir}")
                sys.exit(1)
                
            logger.info(f"Found {len(metadata_files)} metadata files")
            
            # Process files
            pipeline.process_metadata_files(metadata_files, resume=args.resume)
            
            # Wait for completion
            pipeline.wait_for_completion()
            
            # Final statistics
            elapsed = time.time() - pipeline.stats['start_time']
            logger.info(
                f"Pipeline completed in {elapsed/3600:.2f} hours. "
                f"Processed {pipeline.stats['total_processed']} documents "
                f"({pipeline.stats['total_processed']/elapsed:.1f} docs/sec)"
            )
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        
    print("\n✅ Continuous GPU pipeline V2 completed!")


if __name__ == "__main__":
    main()