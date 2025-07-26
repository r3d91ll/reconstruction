#!/usr/bin/env python3
"""
Enhanced Dual-GPU Processing Pipeline
Optimized for RTX A6000 with NVLink for maximum throughput and reliability.

Key improvements:
- NVLink-aware memory management
- Dynamic load balancing
- Robust error recovery
- GPU health monitoring
- Checkpoint/resume capability
- Memory-aware batching
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
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple, Union
from tqdm import tqdm
from arango import ArangoClient
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue, Empty, Full
import threading
import multiprocessing as mp
from dataclasses import dataclass
import gc
import signal
import atexit

# Add infrastructure directory to path
infrastructure_dir = Path(__file__).parent.parent
sys.path.append(str(infrastructure_dir))

from irec_infrastructure.models.metadata import ArxivMetadata

# Setup logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_dual_gpu_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class GPUStatus:
    """Track GPU status and health."""
    gpu_id: int
    is_healthy: bool = True
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    temperature: int = 0
    utilization: int = 0
    processed_count: int = 0
    error_count: int = 0
    last_update: datetime = None
    
    @property
    def memory_percent(self) -> float:
        return (self.memory_used_gb / self.memory_total_gb * 100) if self.memory_total_gb > 0 else 0


class GPUMemoryManager:
    """Manages GPU memory and NVLink optimization."""
    
    def __init__(self):
        self.nvlink_available = self._check_nvlink()
        self.memory_thresholds = {
            'warning': 85.0,  # Warning at 85% memory
            'critical': 95.0,  # Critical at 95% memory
            'batch_reduce': 80.0  # Reduce batch size at 80%
        }
        
    def _check_nvlink(self) -> bool:
        """Check if NVLink is available between GPUs."""
        if torch.cuda.device_count() < 2:
            return False
            
        try:
            # Check if GPUs can access each other's memory
            torch.cuda.set_device(0)
            can_access = torch.cuda.can_device_access_peer(0, 1)
            
            if can_access:
                # Enable peer access for NVLink
                for i in range(torch.cuda.device_count()):
                    for j in range(torch.cuda.device_count()):
                        if i != j:
                            try:
                                torch.cuda.set_device(i)
                                torch.cuda.synchronize(i)
                                # This enables peer memory access
                                # Note: This might fail if already enabled
                            except:
                                pass
                
                logger.info("NVLink detected and enabled between GPUs")
                return True
            else:
                logger.info("NVLink not available - using standard multi-GPU")
                return False
                
        except Exception as e:
            logger.warning(f"Could not check NVLink status: {e}")
            return False
    
    def get_gpu_memory_status(self, gpu_id: int) -> Dict[str, float]:
        """Get detailed memory status for a GPU."""
        torch.cuda.set_device(gpu_id)
        
        # Get memory stats
        allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
        reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
        total = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
        
        # Get temperature and utilization if available
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            pynvml.nvmlShutdown()
        except:
            temp = 0
            util = 0
            
        
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'free_gb': total - allocated,
            'percent': (allocated / total * 100),
            'temperature': temp,
            'utilization': util
        }
    
    def should_reduce_batch_size(self, gpu_id: int) -> bool:
        """Check if batch size should be reduced due to memory pressure."""
        status = self.get_gpu_memory_status(gpu_id)
        return status['percent'] > self.memory_thresholds['batch_reduce']
    
    def cleanup_gpu_memory(self, gpu_id: int):
        """Force cleanup of GPU memory."""
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(gpu_id)
        gc.collect()


class ProcessingCheckpoint:
    """Manages checkpoints for resume capability."""
    
    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()
        
    def _load(self) -> Dict:
        """Load checkpoint data."""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
        
        return {
            'processed_files': set(),
            'failed_files': {},
            'gpu_stats': {0: {'processed': 0, 'errors': 0}, 
                         1: {'processed': 0, 'errors': 0}},
            'last_update': None
        }
    
    def save(self):
        """Save checkpoint data."""
        self.data['last_update'] = datetime.now()
        try:
            with open(self.checkpoint_path, 'wb') as f:
                pickle.dump(self.data, f)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def mark_processed(self, file_path: str):
        """Mark a file as processed."""
        self.data['processed_files'].add(file_path)
        
    def mark_failed(self, file_path: str, error: str):
        """Mark a file as failed."""
        self.data['failed_files'][file_path] = {
            'error': error,
            'timestamp': datetime.now(),
            'attempts': self.data['failed_files'].get(file_path, {}).get('attempts', 0) + 1
        }
    
    def should_process(self, file_path: str) -> bool:
        """Check if file should be processed."""
        if file_path in self.data['processed_files']:
            return False
        
        # Check if failed too many times
        if file_path in self.data['failed_files']:
            failed_info = self.data['failed_files'][file_path]
            if failed_info['attempts'] >= 3:
                return False
                
        return True
    
    def update_gpu_stats(self, gpu_id: int, processed: int = 0, errors: int = 0):
        """Update GPU statistics."""
        self.data['gpu_stats'][gpu_id]['processed'] += processed
        self.data['gpu_stats'][gpu_id]['errors'] += errors


class EnhancedGPUWorker:
    """Enhanced GPU worker with better error handling and monitoring."""
    
    def __init__(self, gpu_id: int, input_queue: mp.Queue, output_queue: mp.Queue,
                 status_queue: mp.Queue, config: Dict):
        self.gpu_id = gpu_id
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.status_queue = status_queue
        self.config = config
        
        self.processor = None
        self.memory_manager = GPUMemoryManager()
        self.status = GPUStatus(gpu_id=gpu_id)
        
        # Dynamic batch sizing
        self.current_batch_size = config['batch_size']
        self.min_batch_size = max(1, config['batch_size'] // 4)
        self.max_batch_size = config['batch_size'] * 2
        
        # Error tracking
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
        
    def initialize(self):
        """Initialize Jina on the assigned GPU with error recovery."""
        # Set this process to use only the assigned GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        
        # Set process priority
        try:
            os.nice(10)  # Lower priority to be nice to other processes
        except:
            pass
        
        # Import here to ensure correct GPU binding
        from irec_infrastructure.embeddings.local_jina_gpu import LocalJinaConfig, LocalJinaGPU
        
        logger.info(f"Initializing enhanced Jina on GPU {self.gpu_id}")
        
        try:
            config = LocalJinaConfig()
            config.device_ids = [0]  # Always 0 since we set CUDA_VISIBLE_DEVICES
            config.use_fp16 = True
            config.max_length = 8192  # Limit for abstracts
            config.chunk_size = 1024
            
            self.processor = LocalJinaGPU(config)
            
            # Test the processor
            test_result = self.processor.encode_batch(["Test initialization"])
            if test_result is None or len(test_result) == 0:
                raise RuntimeError("Processor test failed")
            
            # Update GPU status
            self._update_status()
            self.status.is_healthy = True
            
            logger.info(f"GPU {self.gpu_id} initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize GPU {self.gpu_id}: {e}")
            self.status.is_healthy = False
            raise
    
    def _update_status(self):
        """Update GPU status information."""
        try:
            mem_status = self.memory_manager.get_gpu_memory_status(0)  # Local GPU is always 0
            
            self.status.memory_used_gb = mem_status['allocated_gb']
            self.status.memory_total_gb = mem_status['total_gb']
            self.status.temperature = mem_status['temperature']
            self.status.utilization = mem_status['utilization']
            self.status.last_update = datetime.now()
            
            # Send status update
            self.status_queue.put({
                'gpu_id': self.gpu_id,
                'status': self.status
            })
            
        except Exception as e:
            logger.warning(f"Failed to update GPU {self.gpu_id} status: {e}")
    
    def _adjust_batch_size(self):
        """Dynamically adjust batch size based on memory usage."""
        if self.memory_manager.should_reduce_batch_size(0):
            # Reduce batch size
            self.current_batch_size = max(
                self.min_batch_size,
                int(self.current_batch_size * 0.8)
            )
            logger.info(f"GPU {self.gpu_id}: Reduced batch size to {self.current_batch_size}")
            
        elif self.status.memory_percent < 60 and self.consecutive_errors == 0:
            # Increase batch size if memory is available and no recent errors
            self.current_batch_size = min(
                self.max_batch_size,
                int(self.current_batch_size * 1.2)
            )
            logger.debug(f"GPU {self.gpu_id}: Increased batch size to {self.current_batch_size}")
    
    def process_batch(self, batch_data: List[Tuple[str, str, str]]) -> List[Dict]:
        """Process a batch with error handling and retries."""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Extract data
                file_paths = [item[0] for item in batch_data]
                arxiv_ids = [item[1] for item in batch_data]
                abstracts = [item[2] for item in batch_data]
                
                # Check for empty abstracts
                valid_indices = [i for i, abstract in enumerate(abstracts) if abstract and len(abstract) > 10]
                
                if not valid_indices:
                    logger.warning(f"GPU {self.gpu_id}: No valid abstracts in batch")
                    return []
                
                # Filter to valid items
                valid_abstracts = [abstracts[i] for i in valid_indices]
                valid_arxiv_ids = [arxiv_ids[i] for i in valid_indices]
                valid_file_paths = [file_paths[i] for i in valid_indices]
                
                # Generate embeddings with timeout
                start_time = time.time()
                embeddings = self.processor.encode_batch(valid_abstracts, batch_size=32)
                encode_time = time.time() - start_time
                
                # Validate embeddings
                if embeddings is None or len(embeddings) != len(valid_abstracts):
                    raise ValueError(f"Invalid embeddings: got {len(embeddings) if embeddings else 0}, expected {len(valid_abstracts)}")
                
                # Convert to list format
                if torch.is_tensor(embeddings):
                    embeddings = embeddings.cpu().numpy()
                
                embeddings_list = embeddings.tolist()
                
                # Create results
                results = []
                for i, (file_path, arxiv_id, embedding) in enumerate(zip(valid_file_paths, valid_arxiv_ids, embeddings_list)):
                    # Validate embedding
                    if len(embedding) != 2048:
                        logger.error(f"Invalid embedding dimension for {arxiv_id}: {len(embedding)}")
                        continue
                        
                    results.append({
                        'file_path': file_path,
                        'arxiv_id': arxiv_id,
                        'embedding': embedding,
                        'gpu_id': self.gpu_id,
                        'encode_time': encode_time / len(valid_abstracts)
                    })
                
                # Success - reset error counter
                self.consecutive_errors = 0
                self.status.processed_count += len(results)
                
                return results
                
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"GPU {self.gpu_id} OOM on attempt {attempt + 1}: {e}")
                
                # Clear memory
                self.memory_manager.cleanup_gpu_memory(0)
                
                # Reduce batch size significantly
                self.current_batch_size = max(1, self.current_batch_size // 2)
                
                if attempt == max_retries - 1:
                    self.consecutive_errors += 1
                    self.status.error_count += len(batch_data)
                    return []
                    
                time.sleep(2 ** attempt)  # Exponential backoff
                
            except Exception as e:
                logger.error(f"GPU {self.gpu_id} processing error on attempt {attempt + 1}: {e}")
                
                if attempt == max_retries - 1:
                    self.consecutive_errors += 1
                    self.status.error_count += len(batch_data)
                    
                    # Mark individual items as failed
                    failed_results = []
                    for file_path, arxiv_id, _ in batch_data:
                        failed_results.append({
                            'file_path': file_path,
                            'arxiv_id': arxiv_id,
                            'error': str(e),
                            'gpu_id': self.gpu_id
                        })
                    return failed_results
                    
                time.sleep(1)
        
        return []
    
    def run(self):
        """Main worker loop with health monitoring."""
        try:
            self.initialize()
        except Exception as e:
            logger.error(f"GPU {self.gpu_id} failed to initialize: {e}")
            # Send unhealthy status
            self.status.is_healthy = False
            self.status_queue.put({
                'gpu_id': self.gpu_id,
                'status': self.status,
                'error': str(e)
            })
            return
        
        # Main processing loop
        last_status_update = time.time()
        status_update_interval = 30  # seconds
        
        while True:
            try:
                # Update status periodically
                if time.time() - last_status_update > status_update_interval:
                    self._update_status()
                    last_status_update = time.time()
                
                # Check if we should stop due to errors
                if self.consecutive_errors >= self.max_consecutive_errors:
                    logger.error(f"GPU {self.gpu_id} shutting down due to repeated errors")
                    self.status.is_healthy = False
                    break
                
                # Get batch from queue with timeout
                try:
                    batch = self.input_queue.get(timeout=5)
                except Empty:
                    continue
                
                # Check for stop signal
                if batch is None:
                    logger.info(f"GPU {self.gpu_id} received stop signal")
                    break
                
                # Adjust batch size if needed
                self._adjust_batch_size()
                
                # Process batch
                start_time = time.time()
                results = self.process_batch(batch)
                process_time = time.time() - start_time
                
                # Send results
                self.output_queue.put({
                    'results': results,
                    'gpu_id': self.gpu_id,
                    'process_time': process_time,
                    'batch_size': len(batch),
                    'actual_batch_size': self.current_batch_size
                })
                
                # Periodic memory cleanup
                if self.status.processed_count % 1000 == 0:
                    self.memory_manager.cleanup_gpu_memory(0)
                    
            except Exception as e:
                logger.error(f"GPU {self.gpu_id} worker error: {e}")
                self.consecutive_errors += 1
                
        # Final status update
        self._update_status()
        logger.info(f"GPU {self.gpu_id} processed {self.status.processed_count} documents total")


def enhanced_gpu_worker_process(gpu_id: int, input_queue: mp.Queue, 
                               output_queue: mp.Queue, status_queue: mp.Queue,
                               config: Dict):
    """Enhanced worker process with signal handling."""
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"GPU {gpu_id} worker received signal {signum}")
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and run worker
    worker = EnhancedGPUWorker(gpu_id, input_queue, output_queue, status_queue, config)
    
    try:
        worker.run()
    except KeyboardInterrupt:
        logger.info(f"GPU {gpu_id} worker interrupted")
    except Exception as e:
        logger.error(f"GPU {gpu_id} worker crashed: {e}")
    finally:
        # Ensure queues are closed properly
        input_queue.close()
        output_queue.close()
        status_queue.close()


class LoadBalancer:
    """Dynamic load balancer for GPU assignment."""
    
    def __init__(self):
        self.gpu_loads = {0: 0, 1: 0}
        self.gpu_speeds = {0: [], 1: []}  # Track processing speeds
        self.window_size = 100
        
    def update_stats(self, gpu_id: int, items_processed: int, time_taken: float):
        """Update GPU statistics."""
        if time_taken > 0:
            speed = items_processed / time_taken
            self.gpu_speeds[gpu_id].append(speed)
            
            # Keep only recent measurements
            if len(self.gpu_speeds[gpu_id]) > self.window_size:
                self.gpu_speeds[gpu_id].pop(0)
    
    def get_next_gpu(self) -> int:
        """Get the next GPU to use based on load and performance."""
        # If no performance data, use round-robin
        if not self.gpu_speeds[0] and not self.gpu_speeds[1]:
            return 0 if self.gpu_loads[0] <= self.gpu_loads[1] else 1
        
        # Calculate average speeds
        avg_speed_0 = sum(self.gpu_speeds[0]) / len(self.gpu_speeds[0]) if self.gpu_speeds[0] else 1.0
        avg_speed_1 = sum(self.gpu_speeds[1]) / len(self.gpu_speeds[1]) if self.gpu_speeds[1] else 1.0
        
        # Calculate expected completion time for current load
        expected_time_0 = self.gpu_loads[0] / avg_speed_0 if avg_speed_0 > 0 else float('inf')
        expected_time_1 = self.gpu_loads[1] / avg_speed_1 if avg_speed_1 > 0 else float('inf')
        
        # Choose GPU with lower expected completion time
        return 0 if expected_time_0 <= expected_time_1 else 1
    
    def add_load(self, gpu_id: int, items: int):
        """Add load to a GPU."""
        self.gpu_loads[gpu_id] += items
        
    def remove_load(self, gpu_id: int, items: int):
        """Remove load from a GPU."""
        self.gpu_loads[gpu_id] = max(0, self.gpu_loads[gpu_id] - items)


class EnhancedDualGPUPipeline:
    """Enhanced pipeline with NVLink optimization and reliability features."""
    
    def __init__(self, db_host: str, db_name: str, 
                 batch_size: int = 200,
                 checkpoint_dir: Optional[Path] = None):
        self.db_host = db_host
        self.db_port = int(os.environ.get('ARANGO_PORT', '8529'))
        self.db_name = db_name
        self.username = os.environ.get('ARANGO_USERNAME', 'root')
        self.password = os.environ.get('ARANGO_PASSWORD', '')
        self.batch_size = batch_size
        
        if not self.password:
            raise ValueError("ARANGO_PASSWORD environment variable not set!")
        
        # Connect to database
        self._connect_database()
        
        # Setup multiprocessing with larger queues
        self.mp_manager = mp.Manager()
        self.input_queues = {
            0: mp.Queue(maxsize=20),
            1: mp.Queue(maxsize=20)
        }
        self.output_queue = mp.Queue(maxsize=100)
        self.status_queue = mp.Queue()
        
        # Worker configuration
        self.worker_config = {
            'batch_size': batch_size,
            'timeout': 300  # 5 minute timeout
        }
        
        # Workers and monitoring
        self.workers = {}
        self.gpu_status = {0: GPUStatus(gpu_id=0), 1: GPUStatus(gpu_id=1)}
        self.load_balancer = LoadBalancer()
        
        # Checkpointing
        checkpoint_path = checkpoint_dir or Path(f"./checkpoints/{db_name}")
        self.checkpoint = ProcessingCheckpoint(checkpoint_path / "checkpoint.pkl")
        
        # Stats tracking
        self.stats = {
            'start_time': time.time(),
            'total_processed': 0,
            'total_failed': 0,
            'gpu_stats': {0: {'processed': 0, 'failed': 0, 'time': 0},
                         1: {'processed': 0, 'failed': 0, 'time': 0}}
        }
        
        # Setup cleanup on exit
        atexit.register(self.cleanup)
        
        logger.info(f"Initialized enhanced dual-GPU pipeline for {self.db_name}")
        
    def _connect_database(self):
        """Connect to database with retry."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.client = ArangoClient(hosts=f'http://{self.db_host}:{self.db_port}')
                self.sys_db = self.client.db('_system', username=self.username, password=self.password)
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    raise ValueError(f"Database connection failed after {max_retries} attempts: {e}")
                time.sleep(2 ** attempt)
    
    def setup_database(self, clean_start=False):
        """Setup database with error handling."""
        try:
            if self.sys_db.has_database(self.db_name):
                if clean_start:
                    logger.warning(f"Dropping existing database: {self.db_name}")
                    self.sys_db.delete_database(self.db_name)
                else:
                    logger.info(f"Using existing database: {self.db_name}")
                    self.db = self.client.db(self.db_name, username=self.username, password=self.password)
                    return
            
            # Create database
            self.sys_db.create_database(self.db_name)
            self.db = self.client.db(self.db_name, username=self.username, password=self.password)
            
            # Create collection with optimized indexes
            collection = self.db.create_collection('abstract_metadata')
            
            indexes = [
                {'type': 'hash', 'fields': ['arxiv_id'], 'unique': True},
                {'type': 'persistent', 'fields': ['categories[*]'], 'sparse': True},
                {'type': 'persistent', 'fields': ['published']},
                {'type': 'persistent', 'fields': ['processed_gpu']},
                {'type': 'fulltext', 'fields': ['title'], 'minLength': 3},
                {'type': 'fulltext', 'fields': ['abstract'], 'minLength': 3}
            ]
            
            for index in indexes:
                collection.add_index(index)
            
            logger.info("Database schema created for enhanced dual-GPU pipeline")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            raise
    
    def start_gpu_workers(self):
        """Start GPU worker processes with monitoring."""
        for gpu_id in [0, 1]:
            process = mp.Process(
                target=enhanced_gpu_worker_process,
                args=(gpu_id, self.input_queues[gpu_id], self.output_queue, 
                      self.status_queue, self.worker_config),
                name=f"GPU-{gpu_id}-Worker"
            )
            process.daemon = False  # Don't make daemon so we can clean up properly
            process.start()
            self.workers[gpu_id] = process
            logger.info(f"Started enhanced worker process for GPU {gpu_id} (PID: {process.pid})")
            
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_workers, daemon=True)
        self.monitor_thread.start()
    
    def _monitor_workers(self):
        """Monitor worker health and restart if needed."""
        while any(p.is_alive() for p in self.workers.values()):
            try:
                # Check for status updates
                while not self.status_queue.empty():
                    try:
                        update = self.status_queue.get_nowait()
                        gpu_id = update['gpu_id']
                        self.gpu_status[gpu_id] = update['status']
                        
                        # Log warnings
                        if update['status'].memory_percent > 90:
                            logger.warning(f"GPU {gpu_id} memory critical: {update['status'].memory_percent:.1f}%")
                            
                    except Empty:
                        break
                
                # Check worker health
                for gpu_id, process in self.workers.items():
                    if not process.is_alive():
                        logger.error(f"GPU {gpu_id} worker died! Exit code: {process.exitcode}")
                        
                        # Could implement auto-restart here if desired
                        self.gpu_status[gpu_id].is_healthy = False
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Monitor thread error: {e}")
    
    def stop_gpu_workers(self):
        """Stop GPU worker processes gracefully."""
        logger.info("Stopping GPU workers...")
        
        # Send stop signals
        for gpu_id in [0, 1]:
            try:
                self.input_queues[gpu_id].put(None)
            except:
                pass
        
        # Wait for workers to finish with timeout
        for gpu_id, process in self.workers.items():
            process.join(timeout=30)
            if process.is_alive():
                logger.warning(f"Force terminating GPU {gpu_id} worker")
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    
        logger.info("All GPU workers stopped")
    
    def load_metadata_batch(self, paths: List[Path]) -> List[Dict]:
        """Load metadata files with validation."""
        metadata_list = []
        
        for path in paths:
            try:
                # Skip if already processed
                if not self.checkpoint.should_process(str(path)):
                    continue
                    
                with open(path, 'r') as f:
                    data = json.load(f)
                
                # Validate required fields
                if not data.get('abstract') or len(data['abstract']) < 10:
                    self.checkpoint.mark_failed(str(path), "No valid abstract")
                    continue
                
                # Parse dates
                for date_field in ['published', 'updated']:
                    if date_field in data and data[date_field]:
                        try:
                            data[date_field] = datetime.fromisoformat(
                                data[date_field].replace('Z', '+00:00')
                            )
                        except:
                            data[date_field] = None
                
                # Add file path for tracking
                data['_file_path'] = str(path)
                metadata_list.append(data)
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON error in {path}: {e}")
                self.checkpoint.mark_failed(str(path), f"JSON error: {e}")
            except Exception as e:
                logger.error(f"Error loading {path}: {e}")
                self.checkpoint.mark_failed(str(path), str(e))
                
        return metadata_list
    
    def process_parallel(self, metadata_files: List[Path]) -> Dict[str, Any]:
        """Process metadata files using both GPUs with enhanced reliability."""
        # Start GPU workers
        self.start_gpu_workers()
        
        # Wait for workers to initialize
        logger.info("Waiting for GPU workers to initialize...")
        time.sleep(5)
        
        # Result collection
        results_buffer = []
        failed_items = []
        stop_collection = threading.Event()
        
        def collect_results():
            """Collect and process results from GPU workers."""
            while not stop_collection.is_set() or not self.output_queue.empty():
                try:
                    result = self.output_queue.get(timeout=1)
                    
                    # Update load balancer stats
                    gpu_id = result['gpu_id']
                    self.load_balancer.remove_load(gpu_id, result['batch_size'])
                    self.load_balancer.update_stats(
                        gpu_id, 
                        len([r for r in result['results'] if 'embedding' in r]),
                        result['process_time']
                    )
                    
                    # Separate successful and failed results
                    successful = []
                    failed = []
                    
                    for r in result['results']:
                        if 'error' in r:
                            failed.append(r)
                            self.checkpoint.mark_failed(r['file_path'], r['error'])
                        else:
                            successful.append(r)
                            self.checkpoint.mark_processed(r['file_path'])
                    
                    if successful:
                        results_buffer.append({
                            'results': successful,
                            'gpu_id': gpu_id
                        })
                    
                    if failed:
                        failed_items.extend(failed)
                    
                    # Update stats
                    self.stats['gpu_stats'][gpu_id]['processed'] += len(successful)
                    self.stats['gpu_stats'][gpu_id]['failed'] += len(failed)
                    self.stats['gpu_stats'][gpu_id]['time'] += result['process_time']
                    
                    # Save checkpoint periodically
                    total_processed = sum(s['processed'] for s in self.stats['gpu_stats'].values())
                    if total_processed % 1000 == 0:
                        self.checkpoint.save()
                        
                except Empty:
                    continue
                except Exception as e:
                    logger.error(f"Result collection error: {e}")
        
        # Start result collection thread
        collector_thread = threading.Thread(target=collect_results)
        collector_thread.start()
        
        # Process files
        batch_buffer = []
        pending_batches = 0
        
        with tqdm(total=len(metadata_files), desc="Processing abstracts") as pbar:
            for i in range(0, len(metadata_files), self.batch_size):
                batch_files = metadata_files[i:i + self.batch_size]
                
                # Load metadata
                metadata_batch = self.load_metadata_batch(batch_files)
                
                if not metadata_batch:
                    pbar.update(len(batch_files))
                    continue
                
                # Prepare batch data
                batch_data = [
                    (m['_file_path'], m['arxiv_id'], m['abstract']) 
                    for m in metadata_batch
                ]
                
                # Choose GPU using load balancer
                gpu_id = self.load_balancer.get_next_gpu()
                
                # Wait if queue is full
                while self.input_queues[gpu_id].full():
                    time.sleep(0.1)
                    # Try other GPU
                    other_gpu = 1 - gpu_id
                    if not self.input_queues[other_gpu].full():
                        gpu_id = other_gpu
                        break
                
                # Send to GPU
                self.input_queues[gpu_id].put(batch_data)
                self.load_balancer.add_load(gpu_id, len(batch_data))
                pending_batches += 1
                
                # Store metadata for database insertion
                batch_buffer.append(metadata_batch)
                
                # Process results when buffer is large enough
                if len(batch_buffer) >= 4 or pending_batches >= 8:
                    self._process_results_buffer(results_buffer, batch_buffer)
                    results_buffer.clear()
                    batch_buffer.clear()
                    pending_batches = max(0, pending_batches - 4)
                
                pbar.update(len(batch_files))
                
                # Log progress periodically
                if i % (self.batch_size * 10) == 0:
                    self._log_progress()
        
        # Wait for final results
        logger.info("Waiting for final GPU results...")
        
        # Wait for queues to empty
        for gpu_id in [0, 1]:
            while not self.input_queues[gpu_id].empty():
                time.sleep(0.1)
        
        # Wait for pending results with timeout
        timeout = 60  # 60 seconds timeout
        start_wait = time.time()
        
        while pending_batches > 0 and (time.time() - start_wait) < timeout:
            time.sleep(0.5)
            if len(results_buffer) > 0:
                pending_batches = max(0, pending_batches - len(results_buffer))
        
        # Process final results
        if batch_buffer:
            self._process_results_buffer(results_buffer, batch_buffer)
        
        # Stop workers
        self.stop_gpu_workers()
        
        # Stop collection thread
        stop_collection.set()
        collector_thread.join(timeout=10)
        
        # Final checkpoint save
        self.checkpoint.save()
        
        # Calculate final stats
        self.stats['total_processed'] = sum(s['processed'] for s in self.stats['gpu_stats'].values())
        self.stats['total_failed'] = sum(s['failed'] for s in self.stats['gpu_stats'].values())
        self.stats['elapsed_time'] = time.time() - self.stats['start_time']
        
        return self.stats
    
    def _process_results_buffer(self, results_buffer: List[Dict], metadata_buffer: List[List[Dict]]):
        """Process and store results with transaction safety."""
        # Create lookup for metadata
        metadata_lookup = {}
        for batch in metadata_buffer:
            for m in batch:
                metadata_lookup[m['arxiv_id']] = m
        
        # Prepare records
        records = []
        
        for result_batch in results_buffer:
            for result in result_batch['results']:
                arxiv_id = result['arxiv_id']
                if arxiv_id not in metadata_lookup:
                    continue
                    
                metadata = metadata_lookup[arxiv_id]
                
                record = {
                    '_key': arxiv_id,
                    'arxiv_id': arxiv_id,
                    'title': metadata.get('title', ''),
                    'authors': metadata.get('authors', []),
                    'abstract': metadata['abstract'],
                    'categories': metadata.get('categories', []),
                    'published': metadata['published'].isoformat() if metadata.get('published') else None,
                    'updated': metadata['updated'].isoformat() if metadata.get('updated') else None,
                    'doi': metadata.get('doi'),
                    'journal_ref': metadata.get('journal_ref'),
                    'pdf_url': metadata.get('pdf_url'),
                    'abs_url': metadata.get('abs_url'),
                    'abstract_embedding': result['embedding'],
                    'processed_gpu': result['gpu_id'],
                    'encode_time': result.get('encode_time', 0),
                    'processed_at': datetime.now().isoformat()
                }
                records.append(record)
        
        # Batch insert with retry
        if records:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Use transaction for atomicity
                    self.db.collection('abstract_metadata').insert_many(
                        records, 
                        overwrite=True, 
                        silent=False
                    )
                    logger.info(f"Stored {len(records)} records")
                    break
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to store {len(records)} records after {max_retries} attempts: {e}")
                        self.stats['total_failed'] += len(records)
                    else:
                        logger.warning(f"Database insert attempt {attempt + 1} failed, retrying...")
                        time.sleep(2 ** attempt)
    
    def _log_progress(self):
        """Log detailed progress information."""
        elapsed = time.time() - self.stats['start_time']
        
        # Calculate rates
        gpu_rates = {}
        for gpu_id in [0, 1]:
            processed = self.stats['gpu_stats'][gpu_id]['processed']
            gpu_time = self.stats['gpu_stats'][gpu_id]['time']
            gpu_rates[gpu_id] = processed / elapsed if elapsed > 0 else 0
        
        total_rate = sum(gpu_rates.values())
        
        # Memory status
        mem_status = []
        for gpu_id, status in self.gpu_status.items():
            if status.last_update:
                mem_status.append(f"GPU{gpu_id}: {status.memory_percent:.1f}%")
        
        logger.info(
            f"Progress - Total: {self.stats['total_processed']} | "
            f"Rate: {total_rate:.1f} docs/s | "
            f"GPU0: {gpu_rates[0]:.1f}/s | "
            f"GPU1: {gpu_rates[1]:.1f}/s | "
            f"Memory: [{', '.join(mem_status)}] | "
            f"Failed: {self.stats['total_failed']}"
        )
    
    def verify_database(self):
        """Comprehensive database verification."""
        print("\nDatabase Verification:")
        print("="*60)
        
        # Total count
        total_count = self.db.collection('abstract_metadata').count()
        print(f"Total documents: {total_count:,}")
        
        # GPU distribution
        cursor = self.db.aql.execute("""
            FOR doc IN abstract_metadata
                COLLECT gpu = doc.processed_gpu WITH COUNT INTO count
                RETURN {gpu: gpu, count: count}
        """)
        
        print("\nDocuments per GPU:")
        gpu_counts = {}
        for item in cursor:
            gpu_counts[item['gpu']] = item['count']
            print(f"  GPU {item['gpu']}: {item['count']:,}")
        
        # Load balance efficiency
        if len(gpu_counts) == 2:
            balance = min(gpu_counts.values()) / max(gpu_counts.values())
            print(f"\nLoad balance: {balance:.2%}")
        
        # Processing time statistics
        cursor = self.db.aql.execute("""
            FOR doc IN abstract_metadata
                FILTER doc.encode_time != null
                COLLECT gpu = doc.processed_gpu
                AGGREGATE 
                    avg_time = AVG(doc.encode_time),
                    min_time = MIN(doc.encode_time),
                    max_time = MAX(doc.encode_time)
                RETURN {
                    gpu: gpu,
                    avg_time: avg_time,
                    min_time: min_time,
                    max_time: max_time
                }
        """)
        
        print("\nProcessing time per GPU:")
        for item in cursor:
            print(f"  GPU {item['gpu']}: avg={item['avg_time']*1000:.1f}ms, "
                  f"min={item['min_time']*1000:.1f}ms, max={item['max_time']*1000:.1f}ms")
        
        # Embedding validation
        cursor = self.db.aql.execute("""
            FOR doc IN abstract_metadata
                LIMIT 100
                RETURN {
                    has_embedding: doc.abstract_embedding != null,
                    embedding_length: LENGTH(doc.abstract_embedding)
                }
        """)
        
        results = list(cursor)
        valid_embeddings = sum(1 for r in results if r['has_embedding'] and r['embedding_length'] == 2048)
        print(f"\nEmbedding validation (100 samples): {valid_embeddings}/100 valid")
        
        # Category distribution
        cursor = self.db.aql.execute("""
            FOR doc IN abstract_metadata
                FOR cat IN doc.categories
                    COLLECT category = cat WITH COUNT INTO count
                    SORT count DESC
                    LIMIT 10
                    RETURN {category: category, count: count}
        """)
        
        print("\nTop 10 categories:")
        for item in cursor:
            print(f"  {item['category']}: {item['count']:,}")
    
    def cleanup(self):
        """Cleanup resources on exit."""
        logger.info("Cleaning up pipeline resources...")
        
        try:
            # Stop workers if running
            for process in self.workers.values():
                if process.is_alive():
                    process.terminate()
                    
            # Close queues
            for q in self.input_queues.values():
                q.close()
            self.output_queue.close()
            self.status_queue.close()
            
            # Save final checkpoint
            self.checkpoint.save()
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

def main():
    """Run the enhanced dual-GPU pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enhanced dual-GPU processing pipeline with NVLink optimization"
    )
    parser.add_argument('--metadata-dir', type=str, 
                        default='/mnt/data/arxiv_data/metadata',
                        help='Directory containing metadata JSON files')
    parser.add_argument('--count', type=int, 
                        help='Number of documents to process (default: all)')
    parser.add_argument('--db-name', type=str, 
                        default='arxiv_abstracts_enhanced',
                        help='Database name')
    parser.add_argument('--db-host', type=str, 
                        default='192.168.1.69',
                        help='Database host/IP')
    parser.add_argument('--clean-start', action='store_true',
                        help='Drop existing database and start fresh')
    parser.add_argument('--batch-size', type=int, default=200,
                        help='Base batch size per GPU')
    parser.add_argument('--checkpoint-dir', type=str,
                        help='Directory for checkpoints')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    print("\nENHANCED DUAL-GPU PIPELINE WITH NVLINK")
    print("="*60)
    print(f"Metadata source: {args.metadata_dir}")
    print(f"Database: {args.db_name} @ {args.db_host}")
    print(f"Base batch size: {args.batch_size} per GPU")
    print(f"Checkpoint directory: {args.checkpoint_dir or 'default'}")
    print(f"Resume mode: {args.resume}")
    print("="*60)
    
    # Check GPU availability
    if torch.cuda.device_count() < 2:
        print("ERROR: This pipeline requires 2 GPUs")
        sys.exit(1)
    
    print(f"\nDetected GPUs:")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    
    # Initialize pipeline
    try:
        pipeline = EnhancedDualGPUPipeline(
            db_host=args.db_host,
            db_name=args.db_name,
            batch_size=args.batch_size,
            checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None
        )
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        sys.exit(1)
    
    # Setup database
    if not args.resume:
        pipeline.setup_database(clean_start=args.clean_start)
    else:
        # Connect to existing database
        pipeline.db = pipeline.client.db(pipeline.db_name, 
                                       username=pipeline.username, 
                                       password=pipeline.password)
    
    # Get metadata files
    metadata_dir = Path(args.metadata_dir)
    if not metadata_dir.exists():
        print(f"ERROR: Metadata directory not found: {metadata_dir}")
        sys.exit(1)
        
    metadata_files = sorted(metadata_dir.glob("*.json"))
    
    if args.count:
        metadata_files = metadata_files[:args.count]
    
    print(f"\nFound {len(metadata_files)} metadata files")
    
    if args.resume:
        # Filter already processed files
        original_count = len(metadata_files)
        metadata_files = [f for f in metadata_files if pipeline.checkpoint.should_process(str(f))]
        print(f"Resuming: {original_count - len(metadata_files)} already processed")
        print(f"Remaining: {len(metadata_files)} files to process")
    
    if not metadata_files:
        print("No files to process!")
        return
    
    print("\nStarting enhanced dual-GPU processing...")
    print("="*60)
    
    # Process
    start_time = time.time()
    
    try:
        stats = pipeline.process_parallel(metadata_files)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        pipeline.cleanup()
        sys.exit(0)
    except Exception as e:
        print(f"\nPipeline error: {e}")
        pipeline.cleanup()
        sys.exit(1)
    
    elapsed = time.time() - start_time
    
    # Print results
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Total processed: {stats['total_processed']:,}")
    print(f"Total failed: {stats['total_failed']:,}")
    
    if stats['total_processed'] > 0:
        print(f"\nPerformance:")
        print(f"  Combined rate: {stats['total_processed']/elapsed:.1f} docs/second")
        
        for gpu_id in [0, 1]:
            gpu_stats = stats['gpu_stats'][gpu_id]
            print(f"  GPU {gpu_id}: {gpu_stats['processed']:,} docs, "
                  f"{gpu_stats['processed']/elapsed:.1f} docs/s")
        
        # Efficiency calculation
        ideal_speedup = 2.0
        single_gpu_estimate = stats['total_processed'] / elapsed / ideal_speedup
        actual_speedup = (stats['total_processed'] / elapsed) / single_gpu_estimate
        efficiency = actual_speedup / ideal_speedup * 100
        
        print(f"\nEfficiency:")
        print(f"  Estimated single GPU: {single_gpu_estimate:.1f} docs/s")
        print(f"  Actual speedup: {actual_speedup:.2f}x")
        print(f"  Parallel efficiency: {efficiency:.1f}%")
    
    # Verify database
    pipeline.verify_database()
    
    # Show checkpoint info
    print(f"\nCheckpoint saved: {pipeline.checkpoint.checkpoint_path}")
    print(f"  Processed files: {len(pipeline.checkpoint.data['processed_files']):,}")
    print(f"  Failed files: {len(pipeline.checkpoint.data['failed_files']):,}")
    
    print("\n✅ Enhanced dual-GPU pipeline completed successfully!")


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Set up better error handling
    import faulthandler
    faulthandler.enable()
    
    main()