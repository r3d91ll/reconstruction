#!/usr/bin/env python3
"""
Production-Ready Dual-GPU Processing Pipeline
With advanced memory management, smart batching, and predictive load balancing.

Key enhancements:
- AdvancedMemoryManager with late chunking memory estimation
- SmartBatcher for optimal document grouping
- EnhancedCheckpoint with validation and recovery
- PredictiveLoadBalancer with performance history
- Comprehensive health monitoring
"""

import os
import sys
import json
import logging
import time
import torch
import numpy as np
import psutil
import pickle
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple, Union, Set
from tqdm import tqdm
from arango import ArangoClient
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue, Empty, Full, PriorityQueue
import threading
import multiprocessing as mp
from dataclasses import dataclass, field
import gc
import signal
import atexit
from collections import deque, defaultdict
import statistics

# Add infrastructure directory to path
infrastructure_dir = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(infrastructure_dir))

from irec_infrastructure.models.metadata import ArxivMetadata

# Setup logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler('production_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class DocumentInfo:
    """Information about a document for smart batching."""
    file_path: str
    arxiv_id: str
    abstract: str
    char_count: int
    estimated_tokens: int
    priority: float = 1.0
    
    def __lt__(self, other):
        # For priority queue - higher priority first
        return self.priority > other.priority


@dataclass
class BatchInfo:
    """Information about a batch for processing."""
    documents: List[DocumentInfo]
    total_tokens: int
    estimated_memory_mb: float
    priority: float
    
    def __lt__(self, other):
        return self.priority > other.priority


class AdvancedMemoryManager:
    """Advanced GPU memory management with late chunking awareness."""
    
    def __init__(self):
        self.memory_history = defaultdict(lambda: deque(maxlen=100))
        self.allocation_predictor = {}
        
        # Late chunking memory model parameters
        self.base_model_memory_gb = 4.5  # Jina v4 base memory
        self.attention_memory_factor = 0.00001  # Quadratic scaling factor
        self.embedding_memory_per_token = 0.0001  # GB per token for embeddings
        
        # Safety margins
        self.safety_margin = 0.15  # 15% safety margin
        self.critical_threshold = 0.90  # 90% is critical
        
    def estimate_batch_memory(self, token_counts: List[int]) -> float:
        """Estimate memory needed for a batch with late chunking.
        
        Late chunking processes full documents, so attention memory scales
        quadratically with the longest document in the batch.
        """
        if not token_counts:
            return 0.0
            
        batch_size = len(token_counts)
        max_tokens = max(token_counts)
        total_tokens = sum(token_counts)
        
        # Base model memory (constant)
        base_memory = self.base_model_memory_gb
        
        # Attention memory (quadratic in max sequence length)
        attention_memory = self.attention_memory_factor * (max_tokens ** 2) * batch_size
        
        # Embedding memory (linear in total tokens)
        embedding_memory = self.embedding_memory_per_token * total_tokens
        
        # Gradient memory (if training/fine-tuning)
        gradient_memory = 0  # Not used for inference
        
        # Total with safety margin
        total_memory = (base_memory + attention_memory + embedding_memory + gradient_memory)
        total_with_margin = total_memory * (1 + self.safety_margin)
        
        logger.debug(
            f"Memory estimate for batch (size={batch_size}, max_tokens={max_tokens}): "
            f"base={base_memory:.2f}GB, attention={attention_memory:.2f}GB, "
            f"embedding={embedding_memory:.2f}GB, total={total_with_margin:.2f}GB"
        )
        
        return total_with_margin
    
    def can_fit_batch(self, gpu_id: int, token_counts: List[int]) -> bool:
        """Check if a batch can fit in GPU memory."""
        current_usage = self.get_current_usage(gpu_id)
        total_memory = self.get_total_memory(gpu_id)
        available_memory = total_memory - current_usage
        
        estimated_memory = self.estimate_batch_memory(token_counts)
        
        # Check if it fits with safety margin
        return (current_usage + estimated_memory) / total_memory < self.critical_threshold
    
    def get_current_usage(self, gpu_id: int) -> float:
        """Get current GPU memory usage in GB."""
        torch.cuda.set_device(gpu_id)
        return torch.cuda.memory_allocated(gpu_id) / (1024**3)
    
    def get_total_memory(self, gpu_id: int) -> float:
        """Get total GPU memory in GB."""
        return torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
    
    def record_actual_usage(self, gpu_id: int, token_counts: List[int], actual_memory: float):
        """Record actual memory usage for better predictions."""
        key = (len(token_counts), max(token_counts) if token_counts else 0)
        self.memory_history[gpu_id].append({
            'key': key,
            'estimated': self.estimate_batch_memory(token_counts),
            'actual': actual_memory,
            'timestamp': time.time()
        })
        
        # Update predictor with exponential smoothing
        estimated_memory = self.estimate_batch_memory(token_counts)
        if estimated_memory > 0:
            if key in self.allocation_predictor:
                old_ratio = self.allocation_predictor[key]
                new_ratio = actual_memory / estimated_memory
                self.allocation_predictor[key] = 0.7 * old_ratio + 0.3 * new_ratio
            else:
                self.allocation_predictor[key] = actual_memory / estimated_memory
        else:
            # Skip update if estimated memory is zero
            logger.warning(f"Skipping memory predictor update for GPU {gpu_id}: estimated memory is zero")


class SmartBatcher:
    """Intelligent batching based on document characteristics and memory constraints."""
    
    def __init__(self, memory_manager: AdvancedMemoryManager):
        self.memory_manager = memory_manager
        self.token_estimation_factor = 1.3  # Approximate tokens per character
        
        # Batching parameters
        self.min_batch_size = 1
        self.max_batch_size = 256
        self.target_memory_usage = 0.75  # Target 75% GPU memory usage
        
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count from text."""
        # Simple estimation - can be refined with actual tokenizer
        return int(len(text) * self.token_estimation_factor / 4)  # Approximate
    
    def create_optimal_batches(self, documents: List[DocumentInfo], 
                             max_gpu_memory: float) -> List[BatchInfo]:
        """Create optimal batches considering memory constraints and document sizes."""
        
        # Sort documents by size (largest first for better packing)
        sorted_docs = sorted(documents, key=lambda d: d.estimated_tokens, reverse=True)
        
        batches = []
        current_batch = []
        current_tokens = []
        
        for doc in sorted_docs:
            # Check if adding this document would exceed memory
            test_tokens = current_tokens + [doc.estimated_tokens]
            estimated_memory = self.memory_manager.estimate_batch_memory(test_tokens)
            
            if estimated_memory > max_gpu_memory * self.target_memory_usage:
                # Create batch with current documents
                if current_batch:
                    # Use memory estimate for current batch only (without the new document)
                    current_batch_memory = self.memory_manager.estimate_batch_memory(current_tokens)
                    batch_info = BatchInfo(
                        documents=current_batch.copy(),
                        total_tokens=sum(current_tokens),
                        estimated_memory_mb=current_batch_memory * 1024,
                        priority=statistics.mean(d.priority for d in current_batch)
                    )
                    batches.append(batch_info)
                
                # Start new batch
                current_batch = [doc]
                current_tokens = [doc.estimated_tokens]
            else:
                current_batch.append(doc)
                current_tokens.append(doc.estimated_tokens)
        
        # Don't forget the last batch
        if current_batch:
            batch_info = BatchInfo(
                documents=current_batch,
                total_tokens=sum(current_tokens),
                estimated_memory_mb=self.memory_manager.estimate_batch_memory(current_tokens) * 1024,
                priority=statistics.mean(d.priority for d in current_batch)
            )
            batches.append(batch_info)
        
        logger.info(f"Created {len(batches)} optimal batches from {len(documents)} documents")
        return batches
    
    def rebalance_batch(self, batch: BatchInfo, max_memory_mb: float) -> List[BatchInfo]:
        """Split a batch if it's too large for available memory."""
        if batch.estimated_memory_mb <= max_memory_mb:
            return [batch]
        
        # Binary split until all sub-batches fit
        result_batches = []
        pending = [batch.documents]
        
        while pending:
            docs = pending.pop()
            if len(docs) == 1:
                # Single document - can't split further
                result_batches.append(BatchInfo(
                    documents=docs,
                    total_tokens=docs[0].estimated_tokens,
                    estimated_memory_mb=self.memory_manager.estimate_batch_memory(
                        [docs[0].estimated_tokens]) * 1024,
                    priority=docs[0].priority
                ))
            else:
                # Split in half
                mid = len(docs) // 2
                left_docs = docs[:mid]
                right_docs = docs[mid:]
                
                for subdocs in [left_docs, right_docs]:
                    tokens = [d.estimated_tokens for d in subdocs]
                    mem_estimate = self.memory_manager.estimate_batch_memory(tokens) * 1024
                    
                    if mem_estimate <= max_memory_mb:
                        result_batches.append(BatchInfo(
                            documents=subdocs,
                            total_tokens=sum(tokens),
                            estimated_memory_mb=mem_estimate,
                            priority=statistics.mean(d.priority for d in subdocs)
                        ))
                    else:
                        pending.append(subdocs)
        
        return result_batches


class EnhancedCheckpoint:
    """Enhanced checkpoint system with validation and recovery."""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.main_checkpoint = self.checkpoint_dir / "checkpoint.pkl"
        self.backup_checkpoint = self.checkpoint_dir / "checkpoint.bak"
        self.validation_file = self.checkpoint_dir / "validation.json"
        
        self.data = self._load_with_validation()
        
    def _compute_checksum(self, data: Dict) -> str:
        """Compute checksum for checkpoint data."""
        # Create a deterministic string representation
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _validate_checkpoint(self, data: Dict) -> bool:
        """Validate checkpoint data structure."""
        required_fields = ['processed_files', 'failed_files', 'gpu_stats', 'metadata']
        
        for field in required_fields:
            if field not in data:
                logger.warning(f"Checkpoint missing required field: {field}")
                return False
        
        # Validate data types
        if not isinstance(data['processed_files'], (set, list)):
            return False
        
        if not isinstance(data['failed_files'], dict):
            return False
        
        return True
    
    def _load_with_validation(self) -> Dict:
        """Load checkpoint with validation and fallback."""
        # Try main checkpoint first
        if self.main_checkpoint.exists():
            try:
                with open(self.main_checkpoint, 'rb') as f:
                    data = pickle.load(f)
                
                if self._validate_checkpoint(data):
                    # Convert lists back to sets for efficiency
                    if isinstance(data['processed_files'], list):
                        data['processed_files'] = set(data['processed_files'])
                    
                    logger.info(f"Loaded checkpoint with {len(data['processed_files'])} processed files")
                    return data
                else:
                    logger.warning("Main checkpoint validation failed")
                    
            except Exception as e:
                logger.error(f"Failed to load main checkpoint: {e}")
        
        # Try backup checkpoint
        if self.backup_checkpoint.exists():
            try:
                with open(self.backup_checkpoint, 'rb') as f:
                    data = pickle.load(f)
                
                if self._validate_checkpoint(data):
                    if isinstance(data['processed_files'], list):
                        data['processed_files'] = set(data['processed_files'])
                    
                    logger.warning("Loaded backup checkpoint")
                    return data
                    
            except Exception as e:
                logger.error(f"Failed to load backup checkpoint: {e}")
        
        # Return fresh checkpoint
        return self._create_fresh_checkpoint()
    
    def _create_fresh_checkpoint(self) -> Dict:
        """Create a fresh checkpoint structure."""
        return {
            'processed_files': set(),
            'failed_files': {},
            'gpu_stats': {0: {'processed': 0, 'errors': 0, 'time': 0}, 
                         1: {'processed': 0, 'errors': 0, 'time': 0}},
            'metadata': {
                'created_at': datetime.now(),
                'version': '2.0',
                'last_update': None
            },
            'performance_history': defaultdict(list),
            'error_patterns': defaultdict(int)
        }
    
    def save(self, create_backup: bool = True):
        """Save checkpoint with atomic write and backup."""
        # Update metadata
        self.data['metadata']['last_update'] = datetime.now()
        self.data['metadata']['checksum'] = self._compute_checksum(self.data)
        
        # Convert sets to lists for pickling
        save_data = self.data.copy()
        save_data['processed_files'] = list(self.data['processed_files'])
        
        try:
            # Create backup of existing checkpoint
            if create_backup and self.main_checkpoint.exists():
                import shutil
                shutil.copy2(self.main_checkpoint, self.backup_checkpoint)
            
            # Write to temporary file first (atomic write)
            temp_file = self.main_checkpoint.with_suffix('.tmp')
            with open(temp_file, 'wb') as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Atomic rename
            temp_file.replace(self.main_checkpoint)
            
            # Save validation info
            validation_info = {
                'checksum': self.data['metadata']['checksum'],
                'processed_count': len(self.data['processed_files']),
                'failed_count': len(self.data['failed_files']),
                'last_update': self.data['metadata']['last_update'].isoformat()
            }
            
            with open(self.validation_file, 'w') as f:
                json.dump(validation_info, f, indent=2)
            
            logger.debug(f"Checkpoint saved: {len(self.data['processed_files'])} processed")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def record_performance(self, gpu_id: int, batch_size: int, process_time: float):
        """Record performance metrics for analysis."""
        self.data['performance_history'][gpu_id].append({
            'batch_size': batch_size,
            'process_time': process_time,
            'throughput': batch_size / process_time if process_time > 0 else 0,
            'timestamp': time.time()
        })
        
        # Keep only recent history
        if len(self.data['performance_history'][gpu_id]) > 1000:
            self.data['performance_history'][gpu_id] = self.data['performance_history'][gpu_id][-1000:]
    
    def record_error_pattern(self, error_type: str):
        """Track error patterns for debugging."""
        self.data['error_patterns'][error_type] += 1
    
    def mark_failed(self, file_path: str, error: str):
        """Mark a file as failed with error details."""
        if file_path not in self.data['failed_files']:
            self.data['failed_files'][file_path] = {
                'attempts': 0,
                'errors': []
            }
        
        self.data['failed_files'][file_path]['attempts'] += 1
        self.data['failed_files'][file_path]['errors'].append({
            'error': error,
            'timestamp': datetime.now()
        })
        self.data['failed_files'][file_path]['last_attempt'] = datetime.now()
    
    def should_process(self, file_path: str) -> bool:
        """Check if a file should be processed."""
        return (file_path not in self.data['processed_files'] and 
                file_path not in self.data['failed_files'])


class PredictiveLoadBalancer:
    """Advanced load balancer with performance prediction."""
    
    def __init__(self, checkpoint: EnhancedCheckpoint):
        self.checkpoint = checkpoint
        self.gpu_performance = {0: deque(maxlen=100), 1: deque(maxlen=100)}
        self.gpu_queue_sizes = {0: 0, 1: 0}
        
        # Load historical performance
        self._load_performance_history()
        
    def _load_performance_history(self):
        """Load performance history from checkpoint."""
        for gpu_id in [0, 1]:
            history = self.checkpoint.data.get('performance_history', {}).get(gpu_id, [])
            for entry in history[-100:]:  # Last 100 entries
                self.gpu_performance[gpu_id].append(entry['throughput'])
    
    def predict_completion_time(self, gpu_id: int, batch_size: int) -> float:
        """Predict completion time for a batch on a specific GPU."""
        if not self.gpu_performance[gpu_id]:
            # No history - use default estimate
            return batch_size / 50.0  # Assume 50 docs/sec
        
        # Calculate average throughput
        recent_throughput = list(self.gpu_performance[gpu_id])[-20:]
        avg_throughput = statistics.mean(recent_throughput)
        
        # Adjust for queue size
        queue_penalty = self.gpu_queue_sizes[gpu_id] * 0.1
        
        return (batch_size / avg_throughput) + queue_penalty
    
    def select_gpu(self, batch: BatchInfo) -> int:
        """Select optimal GPU for a batch."""
        # Predict completion time on each GPU
        pred_time_0 = self.predict_completion_time(0, len(batch.documents))
        pred_time_1 = self.predict_completion_time(1, len(batch.documents))
        
        # Consider current memory usage
        mem_0 = torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory
        mem_1 = torch.cuda.memory_allocated(1) / torch.cuda.get_device_properties(1).total_memory
        
        # Penalize high memory usage
        if mem_0 > 0.85:
            pred_time_0 *= 2
        if mem_1 > 0.85:
            pred_time_1 *= 2
        
        # Select GPU with lower predicted completion time
        selected = 0 if pred_time_0 <= pred_time_1 else 1
        
        logger.debug(
            f"GPU selection: GPU0={pred_time_0:.2f}s (mem={mem_0:.1%}), "
            f"GPU1={pred_time_1:.2f}s (mem={mem_1:.1%}) -> GPU{selected}"
        )
        
        return selected
    
    def update_performance(self, gpu_id: int, batch_size: int, actual_time: float):
        """Update performance history with actual results."""
        if actual_time > 0:
            throughput = batch_size / actual_time
            self.gpu_performance[gpu_id].append(throughput)
            self.checkpoint.record_performance(gpu_id, batch_size, actual_time)


class ProductionGPUWorker:
    """Production-ready GPU worker with all enhancements."""
    
    def __init__(self, gpu_id: int, config: Dict):
        self.gpu_id = gpu_id
        self.config = config
        
        # Components
        self.memory_manager = AdvancedMemoryManager()
        self.processor = None
        
        # Health monitoring
        self.health_status = {
            'healthy': True,
            'memory_pressure': False,
            'error_rate': 0.0,
            'last_success': time.time(),
            'consecutive_errors': 0
        }
        
        # Performance tracking
        self.processing_times = deque(maxlen=100)
        self.memory_usage_history = deque(maxlen=100)
        
    def initialize(self):
        """Initialize with comprehensive error handling."""
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        
        try:
            from irec_infrastructure.embeddings.local_jina_gpu import LocalJinaConfig, LocalJinaGPU
            
            config = LocalJinaConfig()
            config.device_ids = [0]
            config.use_fp16 = True
            config.max_length = 8192
            
            self.processor = LocalJinaGPU(config)
            
            # Warm up
            test_batch = ["Initialization test"] * 5
            embeddings = self.processor.encode_batch(test_batch)
            
            if embeddings is None or len(embeddings) != len(test_batch):
                raise RuntimeError("Processor warmup failed")
            
            logger.info(f"GPU {self.gpu_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"GPU {self.gpu_id} initialization failed: {e}")
            self.health_status['healthy'] = False
            return False
    
    def process_batch_with_monitoring(self, batch: BatchInfo) -> Dict[str, Any]:
        """Process batch with comprehensive monitoring."""
        start_time = time.time()
        start_memory = self.memory_manager.get_current_usage(0)
        
        try:
            # Extract texts
            texts = [doc.abstract for doc in batch.documents]
            
            # Check memory before processing
            token_counts = [doc.estimated_tokens for doc in batch.documents]
            if not self.memory_manager.can_fit_batch(0, token_counts):
                # Try to free memory
                torch.cuda.empty_cache()
                gc.collect()
                
                if not self.memory_manager.can_fit_batch(0, token_counts):
                    raise MemoryError("Insufficient GPU memory for batch")
            
            # Process
            embeddings = self.processor.encode_batch(texts, batch_size=32)
            
            # Record actual memory usage
            peak_memory = self.memory_manager.get_current_usage(0)
            actual_memory = peak_memory - start_memory
            self.memory_manager.record_actual_usage(0, token_counts, actual_memory)
            
            # Create results
            results = []
            for i, (doc, embedding) in enumerate(zip(batch.documents, embeddings)):
                results.append({
                    'file_path': doc.file_path,
                    'arxiv_id': doc.arxiv_id,
                    'embedding': embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                    'gpu_id': self.gpu_id,
                    'process_time': (time.time() - start_time) / len(batch.documents)
                })
            
            # Update health status
            self.health_status['last_success'] = time.time()
            self.health_status['consecutive_errors'] = 0
            
            # Record performance
            process_time = time.time() - start_time
            self.processing_times.append(process_time)
            self.memory_usage_history.append(peak_memory)
            
            return {
                'success': True,
                'results': results,
                'metrics': {
                    'process_time': process_time,
                    'peak_memory_gb': peak_memory,
                    'memory_delta_gb': actual_memory,
                    'throughput': len(batch.documents) / process_time
                }
            }
            
        except Exception as e:
            # Update health status
            self.health_status['consecutive_errors'] += 1
            error_rate = self.health_status['consecutive_errors'] / 10.0
            self.health_status['error_rate'] = min(1.0, error_rate)
            
            if self.health_status['consecutive_errors'] > 5:
                self.health_status['healthy'] = False
            
            logger.error(f"GPU {self.gpu_id} batch processing error: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'failed_documents': [doc.arxiv_id for doc in batch.documents]
            }
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        current_memory = self.memory_manager.get_current_usage(0)
        total_memory = self.memory_manager.get_total_memory(0)
        
        avg_process_time = statistics.mean(self.processing_times) if self.processing_times else 0
        avg_memory = statistics.mean(self.memory_usage_history) if self.memory_usage_history else 0
        
        return {
            'gpu_id': self.gpu_id,
            'healthy': self.health_status['healthy'],
            'memory_usage_percent': (current_memory / total_memory) * 100,
            'error_rate': self.health_status['error_rate'],
            'avg_process_time': avg_process_time,
            'avg_memory_gb': avg_memory,
            'time_since_last_success': time.time() - self.health_status['last_success']
        }


def production_worker_process(gpu_id: int, input_queue: mp.Queue, output_queue: mp.Queue,
                            health_queue: mp.Queue, config: Dict):
    """Production worker process with enhanced monitoring."""
    
    worker = ProductionGPUWorker(gpu_id, config)
    
    # Initialize
    if not worker.initialize():
        health_queue.put({
            'gpu_id': gpu_id,
            'status': 'initialization_failed',
            'report': worker.get_health_report()
        })
        return
    
    # Main loop
    last_health_report = time.time()
    health_interval = 30  # seconds
    
    while True:
        try:
            # Send health report periodically
            if time.time() - last_health_report > health_interval:
                health_queue.put({
                    'gpu_id': gpu_id,
                    'status': 'health_report',
                    'report': worker.get_health_report()
                })
                last_health_report = time.time()
            
            # Get batch
            try:
                batch = input_queue.get(timeout=5)
            except Empty:
                continue
            
            if batch is None:  # Stop signal
                break
            
            # Process batch
            result = worker.process_batch_with_monitoring(batch)
            
            # Send result
            output_queue.put({
                'gpu_id': gpu_id,
                'batch_id': id(batch),
                'result': result
            })
            
        except Exception as e:
            logger.error(f"Worker {gpu_id} error: {e}")
            health_queue.put({
                'gpu_id': gpu_id,
                'status': 'error',
                'error': str(e)
            })


class ProductionPipeline:
    """Production-ready pipeline with all enhancements."""
    
    def __init__(self, db_host: str, db_name: str, checkpoint_dir: Path):
        self.db_host = db_host
        self.db_name = db_name
        
        # Initialize components
        self.checkpoint = EnhancedCheckpoint(checkpoint_dir)
        self.memory_manager = AdvancedMemoryManager()
        self.smart_batcher = SmartBatcher(self.memory_manager)
        self.load_balancer = PredictiveLoadBalancer(self.checkpoint)
        
        # Setup multiprocessing
        self.input_queues = {0: mp.Queue(maxsize=10), 1: mp.Queue(maxsize=10)}
        self.output_queue = mp.Queue(maxsize=50)
        self.health_queue = mp.Queue()
        
        # Workers
        self.workers = {}
        self.worker_health = {0: None, 1: None}
        
        # Database connection
        self._connect_database()
        
        logger.info("Production pipeline initialized")
    
    def _connect_database(self):
        """Connect to database."""
        self.client = ArangoClient(hosts=f'http://{self.db_host}:8529')
        self.db = self.client.db(
            self.db_name,
            username=os.environ.get('ARANGO_USERNAME', 'root'),
            password=os.environ['ARANGO_PASSWORD']
        )
    
    def start_workers(self):
        """Start production workers."""
        config = {
            'max_batch_size': 256,
            'timeout': 300
        }
        
        for gpu_id in [0, 1]:
            process = mp.Process(
                target=production_worker_process,
                args=(gpu_id, self.input_queues[gpu_id], self.output_queue,
                      self.health_queue, config)
            )
            process.start()
            self.workers[gpu_id] = process
            logger.info(f"Started production worker for GPU {gpu_id}")
    
    def monitor_health(self):
        """Monitor worker health in background."""
        def _monitor():
            while any(w.is_alive() for w in self.workers.values()):
                try:
                    health_update = self.health_queue.get(timeout=1)
                    gpu_id = health_update['gpu_id']
                    
                    if health_update['status'] == 'health_report':
                        self.worker_health[gpu_id] = health_update['report']
                        
                        # Check for issues
                        report = health_update['report']
                        if not report['healthy']:
                            logger.error(f"GPU {gpu_id} unhealthy: {report}")
                        elif report['memory_usage_percent'] > 90:
                            logger.warning(f"GPU {gpu_id} memory critical: {report['memory_usage_percent']:.1f}%")
                        
                    elif health_update['status'] == 'error':
                        logger.error(f"GPU {gpu_id} error: {health_update['error']}")
                        
                except Empty:
                    pass
                except Exception as e:
                    logger.error(f"Health monitor error: {e}")
        
        monitor_thread = threading.Thread(target=_monitor, daemon=True)
        monitor_thread.start()
    
    def process_documents(self, metadata_files: List[Path]) -> Dict[str, Any]:
        """Process documents with production pipeline."""
        
        # Start workers and monitoring
        self.start_workers()
        self.monitor_health()
        
        # Wait for initialization
        time.sleep(5)
        
        # Process files
        total_processed = 0
        total_failed = 0
        
        with tqdm(total=len(metadata_files), desc="Processing") as pbar:
            # Process in chunks for better memory management
            chunk_size = 10000
            
            for chunk_start in range(0, len(metadata_files), chunk_size):
                chunk_files = metadata_files[chunk_start:chunk_start + chunk_size]
                
                # Load and prepare documents
                documents = []
                for file_path in chunk_files:
                    if not self.checkpoint.should_process(str(file_path)):
                        pbar.update(1)
                        continue
                    
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        
                        if data.get('abstract') and len(data['abstract']) > 10:
                            doc = DocumentInfo(
                                file_path=str(file_path),
                                arxiv_id=data['arxiv_id'],
                                abstract=data['abstract'],
                                char_count=len(data['abstract']),
                                estimated_tokens=self.smart_batcher.estimate_tokens(data['abstract'])
                            )
                            documents.append(doc)
                    except Exception as e:
                        logger.error(f"Error loading {file_path}: {e}")
                        self.checkpoint.mark_failed(str(file_path), str(e))
                
                if not documents:
                    continue
                
                # Create optimal batches
                max_gpu_memory = min(
                    self.memory_manager.get_total_memory(0),
                    self.memory_manager.get_total_memory(1)
                )
                batches = self.smart_batcher.create_optimal_batches(documents, max_gpu_memory)
                
                # Process batches
                for batch in batches:
                    # Select GPU
                    gpu_id = self.load_balancer.select_gpu(batch)
                    
                    # Send to worker
                    self.input_queues[gpu_id].put(batch)
                    self.load_balancer.gpu_queue_sizes[gpu_id] += 1
                
                # Collect results
                pending_batches = len(batches)
                while pending_batches > 0:
                    try:
                        result = self.output_queue.get(timeout=60)
                        gpu_id = result['gpu_id']
                        
                        # Update queue size
                        self.load_balancer.gpu_queue_sizes[gpu_id] -= 1
                        
                        if result['result']['success']:
                            # Process successful results
                            self._store_results(result['result']['results'])
                            
                            # Update performance
                            metrics = result['result']['metrics']
                            self.load_balancer.update_performance(
                                gpu_id,
                                len(result['result']['results']),
                                metrics['process_time']
                            )
                            
                            total_processed += len(result['result']['results'])
                        else:
                            # Handle failures
                            for doc_id in result['result']['failed_documents']:
                                self.checkpoint.mark_failed(doc_id, result['result']['error'])
                            
                            total_failed += len(result['result']['failed_documents'])
                            
                            # Record error pattern
                            self.checkpoint.record_error_pattern(result['result']['error_type'])
                        
                        pending_batches -= 1
                        
                    except Empty:
                        logger.warning("Timeout waiting for results")
                        pending_batches -= 1
                
                # Update progress
                pbar.update(len(chunk_files))
                
                # Save checkpoint
                self.checkpoint.save()
        
        # Stop workers
        for queue in self.input_queues.values():
            queue.put(None)
        
        for worker in self.workers.values():
            worker.join(timeout=30)
        
        return {
            'total_processed': total_processed,
            'total_failed': total_failed,
            'checkpoint_path': str(self.checkpoint.main_checkpoint)
        }
    
    def _store_results(self, results: List[Dict]):
        """Store results in database."""
        if not results:
            return
            
        try:
            # Ensure collection exists
            if not self.db.has_collection('abstract_metadata'):
                collection = self.db.create_collection('abstract_metadata')
                
                # Add indexes for efficient querying
                indexes = [
                    {'type': 'persistent', 'fields': ['arxiv_id'], 'unique': True},
                    {'type': 'persistent', 'fields': ['published']},
                    {'type': 'fulltext', 'fields': ['title']},
                    {'type': 'fulltext', 'fields': ['abstract']},
                    {'type': 'persistent', 'fields': ['categories[*]']},
                    {'type': 'persistent', 'fields': ['authors[*]']}
                ]
                
                for index in indexes:
                    try:
                        collection.add_index(index)
                    except Exception as e:
                        logger.warning(f"Could not create index {index}: {e}")
            
            collection = self.db.collection('abstract_metadata')
            
            # Prepare documents for insertion
            documents = []
            for result in results:
                # Load metadata from file
                try:
                    with open(result['file_path'], 'r') as f:
                        metadata_dict = json.load(f)
                    
                    # Create document with embedding
                    doc = {
                        '_key': result['arxiv_id'].replace('/', '_'),  # ArangoDB key format
                        'arxiv_id': result['arxiv_id'],
                        'title': metadata_dict.get('title', ''),
                        'authors': metadata_dict.get('authors', []),
                        'abstract': metadata_dict.get('abstract', ''),
                        'categories': metadata_dict.get('categories', []),
                        'published': metadata_dict.get('published'),
                        'updated': metadata_dict.get('updated'),
                        'comments': metadata_dict.get('comments'),
                        'doi': metadata_dict.get('doi'),
                        'abstract_embedding': result['embedding'],
                        'gpu_id': result.get('gpu_id'),
                        'process_time': result.get('process_time'),
                        'processed_at': datetime.now().isoformat()
                    }
                    
                    documents.append(doc)
                    
                except Exception as e:
                    logger.error(f"Error preparing document {result['arxiv_id']}: {e}")
                    continue
            
            if documents:
                # Batch insert with error handling
                batch_size = 1000
                total_inserted = 0
                total_errors = 0
                
                for i in range(0, len(documents), batch_size):
                    batch = documents[i:i + batch_size]
                    
                    try:
                        # Insert with overwrite to handle duplicates
                        result = collection.insert_many(
                            batch,
                            overwrite=True,
                            return_new=False,
                            silent=False
                        )
                        
                        # Count successes and errors
                        if isinstance(result, list):
                            for r in result:
                                if r.get('error'):
                                    total_errors += 1
                                    logger.warning(f"Insert error for {r.get('_key')}: {r.get('errorMessage')}")
                                else:
                                    total_inserted += 1
                        else:
                            total_inserted += len(batch)
                            
                    except Exception as e:
                        logger.error(f"Batch insert error: {e}")
                        total_errors += len(batch)
                        
                        # Try to insert individually on batch failure
                        for doc in batch:
                            try:
                                collection.insert(doc, overwrite=True, silent=True)
                                total_inserted += 1
                            except Exception as individual_error:
                                logger.error(f"Individual insert error for {doc['arxiv_id']}: {individual_error}")
                                total_errors += 1
                
                logger.info(f"Stored {total_inserted} documents, {total_errors} errors")
                
        except Exception as e:
            logger.error(f"Critical error in _store_results: {e}")
            raise


def main():
    """Run production pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Production-ready dual-GPU processing pipeline"
    )
    parser.add_argument('--metadata-dir', type=str, required=True)
    parser.add_argument('--db-name', type=str, required=True)
    parser.add_argument('--db-host', type=str, default='localhost')
    parser.add_argument('--checkpoint-dir', type=str, required=True)
    parser.add_argument('--count', type=int, help='Number of documents to process')
    
    args = parser.parse_args()
    
    # Get metadata files
    metadata_dir = Path(args.metadata_dir)
    metadata_files = sorted(metadata_dir.glob("*.json"))
    
    if args.count:
        metadata_files = metadata_files[:args.count]
    
    print(f"Found {len(metadata_files)} metadata files")
    
    # Run pipeline
    pipeline = ProductionPipeline(
        db_host=args.db_host,
        db_name=args.db_name,
        checkpoint_dir=Path(args.checkpoint_dir)
    )
    
    results = pipeline.process_documents(metadata_files)
    
    print(f"\nProcessing complete:")
    print(f"  Processed: {results['total_processed']}")
    print(f"  Failed: {results['total_failed']}")
    print(f"  Checkpoint: {results['checkpoint_path']}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()