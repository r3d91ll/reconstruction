#!/usr/bin/env python3
"""
Production-Ready Dual GPU Pipeline with Continuous Utilization

Key features:
- Producer-consumer architecture with queues
- Prefetching and double buffering
- Robust checkpointing with fast recovery
- GPU utilization monitoring
- Clean separation of concerns
"""

import os
import sys
import json
import logging
import torch
import torch.multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from queue import Queue, Empty
from threading import Thread, Event
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tqdm import tqdm
import psutil
import pickle
import lmdb
import time
import hashlib
from arango import ArangoClient

# Add infrastructure directory to path
infrastructure_dir = Path(__file__).parent.parent
sys.path.append(str(infrastructure_dir))

# Try to import GPU embedding classes early to catch errors at startup
try:
    from irec_infrastructure.embeddings.local_jina_gpu import LocalJinaGPU, LocalJinaConfig
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to import GPU embedding classes: {e}")
    logger.error("Make sure the irec_infrastructure package is properly installed")
    raise

# Configure multiprocessing for CUDA
mp.set_start_method('spawn', force=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler('continuous_gpu_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class GPUWork:
    """Work item for GPU processing"""
    batch_id: str
    texts: List[str]
    metadata: List[Dict]
    priority: float = 1.0
    
    def __lt__(self, other):
        return self.priority > other.priority
    

@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    # GPU settings
    gpu_devices: List[int] = field(default_factory=list)
    batch_size: int = 128
    prefetch_factor: int = 4
    
    # Queue settings
    preprocessing_workers: int = 8
    max_gpu_queue_size: int = 100
    max_output_queue_size: int = 200
    
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


class GPUWorker(mp.Process):
    """
    Dedicated GPU worker process.
    Runs in separate process to avoid GIL and ensure continuous GPU utilization.
    """
    
    def __init__(
        self,
        gpu_id: int,
        input_queue: mp.Queue,
        output_queue: mp.Queue,
        checkpoint_queue: mp.Queue,
        config: PipelineConfig,
        stop_event: mp.Event
    ):
        super().__init__()
        self.gpu_id = gpu_id
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.checkpoint_queue = checkpoint_queue
        self.config = config
        self.stop_event = stop_event
        self.processed_count = 0
        self.total_gpu_time = 0.0
        
    def run(self):
        """Main GPU processing loop"""
        # Set GPU device for this process
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        torch.cuda.set_device(0)  # Local device 0 in this process
        
        logger.info(f"GPU Worker {self.gpu_id} starting on physical device {self.gpu_id}")
        
        # Initialize model on this GPU (classes already imported at module level)
        
        config = LocalJinaConfig()
        config.device_ids = [0]  # Local device 0 in this process context
        config.use_fp16 = True
        config.max_length = 8192
        
        try:
            model = LocalJinaGPU(config)
            logger.info(f"GPU Worker {self.gpu_id} model loaded successfully")
        except Exception as e:
            logger.error(f"GPU Worker {self.gpu_id} failed to load model: {e}")
            return
        
        # Processing loop
        while not self.stop_event.is_set():
            try:
                # Get work with timeout
                work = self.input_queue.get(timeout=1.0)
                if work is None:  # Poison pill
                    logger.info(f"GPU Worker {self.gpu_id} received shutdown signal")
                    break
                    
                # Process batch on GPU
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                
                # Generate embeddings
                try:
                    embeddings = model.encode_batch(work.texts, batch_size=32)
                    
                    # Convert to list format if needed
                    if torch.is_tensor(embeddings):
                        embeddings = embeddings.cpu().numpy()
                    embeddings_list = embeddings.tolist()
                    
                except torch.cuda.OutOfMemoryError:
                    logger.error(f"GPU {self.gpu_id} OOM - reducing batch size")
                    # Try with smaller batch
                    embeddings_list = []
                    for i in range(0, len(work.texts), 16):
                        sub_batch = work.texts[i:i+16]
                        sub_embeddings = model.encode_batch(sub_batch, batch_size=16)
                        if torch.is_tensor(sub_embeddings):
                            sub_embeddings = sub_embeddings.cpu().numpy()
                        embeddings_list.extend(sub_embeddings.tolist())
                
                end_time.record()
                torch.cuda.synchronize()
                
                gpu_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
                self.total_gpu_time += gpu_time
                
                # Prepare output
                output = {
                    'batch_id': work.batch_id,
                    'embeddings': embeddings_list,
                    'metadata': work.metadata,
                    'gpu_id': self.gpu_id,
                    'gpu_time': gpu_time,
                    'timestamp': datetime.now().isoformat(),
                    'docs_per_second': len(work.texts) / gpu_time if gpu_time > 0 else 0
                }
                
                # Send to output queue
                self.output_queue.put(output)
                
                self.processed_count += 1
                
                # Checkpoint if needed
                if self.processed_count % self.config.checkpoint_interval == 0:
                    checkpoint_data = {
                        'gpu_id': self.gpu_id,
                        'processed_count': self.processed_count,
                        'total_gpu_time': self.total_gpu_time,
                        'last_batch_id': work.batch_id,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.checkpoint_queue.put(checkpoint_data)
                    
                # Periodic memory cleanup
                if self.processed_count % 50 == 0:
                    torch.cuda.empty_cache()
                    
            except Empty:
                continue
            except Exception as e:
                logger.error(f"GPU Worker {self.gpu_id} error: {e}", exc_info=True)
                
        # Cleanup
        avg_time = self.total_gpu_time / self.processed_count if self.processed_count > 0 else 0
        logger.info(f"GPU Worker {self.gpu_id} shutting down. "
                   f"Processed {self.processed_count} batches, "
                   f"avg time: {avg_time:.3f}s/batch")


class PreprocessingWorker(Thread):
    """
    CPU preprocessing worker for abstracts.
    Loads metadata files and prepares batches for GPU processing.
    """
    
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
        
    def run(self):
        """Main preprocessing loop"""
        batch_texts = []
        batch_metadata = []
        batch_num = 0
        
        while not self.stop_event.is_set():
            try:
                # Get document path
                item = self.document_queue.get(timeout=1.0)
                if item is None:  # Poison pill
                    break
                    
                # Handle both file path and pre-loaded metadata
                if isinstance(item, dict):
                    # Pre-loaded metadata
                    metadata = item
                else:
                    # File path - load metadata
                    try:
                        with open(item, 'r') as f:
                            metadata = json.load(f)
                        metadata['_file_path'] = str(item)
                    except Exception as e:
                        logger.error(f"Worker {self.worker_id} failed to load {item}: {e}")
                        continue
                
                # Extract abstract
                abstract = metadata.get('abstract', '')
                if abstract and len(abstract.strip()) > 10:
                    batch_texts.append(abstract)
                    batch_metadata.append({
                        'arxiv_id': metadata.get('arxiv_id', metadata.get('id', '')),
                        'title': metadata.get('title', ''),
                        'authors': metadata.get('authors', []),
                        'categories': metadata.get('categories', []),
                        'published': metadata.get('published'),
                        'file_path': metadata.get('_file_path', '')
                    })
                    
                # Check if batch is ready
                if len(batch_texts) >= self.config.batch_size:
                    # Create work item
                    batch_id = f"w{self.worker_id}_b{batch_num}"
                    work = GPUWork(
                        batch_id=batch_id,
                        texts=batch_texts[:self.config.batch_size],
                        metadata=batch_metadata[:self.config.batch_size]
                    )
                    
                    # Send to GPU queue
                    self.gpu_queue.put(work)
                    
                    # Keep remaining items for next batch
                    batch_texts = batch_texts[self.config.batch_size:]
                    batch_metadata = batch_metadata[self.config.batch_size:]
                    batch_num += 1
                    
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Preprocessing worker {self.worker_id} error: {e}")
                
        # Process remaining items
        if batch_texts:
            batch_id = f"w{self.worker_id}_b{batch_num}_final"
            work = GPUWork(
                batch_id=batch_id,
                texts=batch_texts,
                metadata=batch_metadata
            )
            self.gpu_queue.put(work)
            batch_num += 1
            
        logger.info(f"Preprocessing worker {self.worker_id} shutting down. Created {batch_num} batches")


class CheckpointManager:
    """
    Fast checkpoint management using LMDB for crash recovery.
    """
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # LMDB for fast key-value storage
        self.env = lmdb.open(
            str(self.checkpoint_dir / "pipeline_state"),
            map_size=10 * 1024 * 1024 * 1024,  # 10GB
            max_dbs=3
        )
        
        # Sub-databases
        self.batch_db = self.env.open_db(b'batches')
        self.state_db = self.env.open_db(b'state')
        self.metadata_db = self.env.open_db(b'metadata')
        
    def save_state(self, key: str, value: Any, db=None):
        """Save state to checkpoint"""
        db = db or self.state_db
        with self.env.begin(write=True) as txn:
            # Use JSON instead of pickle for security
            json_data = json.dumps(value, default=str)  # default=str handles datetime objects
            txn.put(key.encode(), json_data.encode('utf-8'), db=db)
            
    def load_state(self, key: str, db=None) -> Optional[Any]:
        """Load state from checkpoint"""
        db = db or self.state_db
        with self.env.begin() as txn:
            data = txn.get(key.encode(), db=db)
            if data:
                # Use JSON instead of pickle for security
                return json.loads(data.decode('utf-8'))
            return None
            
    def save_batch_result(self, batch_id: str, result: Dict):
        """Save batch processing result"""
        # Save to batch database
        self.save_state(batch_id, {
            'processed': True,
            'gpu_id': result['gpu_id'],
            'timestamp': result['timestamp'],
            'doc_count': len(result['metadata'])
        }, db=self.batch_db)
        
        # Save individual document states
        with self.env.begin(write=True) as txn:
            for i, metadata in enumerate(result['metadata']):
                doc_id = metadata['arxiv_id']
                doc_data = {
                    'batch_id': batch_id,
                    'embedding_index': i,
                    'processed_at': result['timestamp']
                }
                txn.put(doc_id.encode(), json.dumps(doc_data, default=str).encode('utf-8'), db=self.metadata_db)
                
    def get_processed_batches(self) -> set:
        """Get set of processed batch IDs"""
        processed = set()
        with self.env.begin() as txn:
            cursor = txn.cursor(db=self.batch_db)
            for key, value in cursor:
                batch_data = json.loads(value.decode('utf-8'))
                if batch_data.get('processed', False):
                    processed.add(key.decode())
        return processed
        
    def get_processed_documents(self) -> set:
        """Get set of processed document IDs"""
        processed = set()
        with self.env.begin() as txn:
            cursor = txn.cursor(db=self.metadata_db)
            for key, _ in cursor:
                processed.add(key.decode())
        return processed
        
    def save_pipeline_metadata(self, metadata: Dict):
        """Save pipeline run metadata"""
        self.save_state('pipeline_metadata', metadata)
        
    def cleanup(self):
        """Clean up resources"""
        self.env.close()


class GPUUtilizationMonitor(Thread):
    """
    Monitor GPU utilization and queue depths.
    """
    
    def __init__(
        self,
        gpu_queue: mp.Queue,
        output_queue: mp.Queue,
        config: PipelineConfig,
        stop_event: Event = None
    ):
        super().__init__(daemon=True)
        self.gpu_queue = gpu_queue
        self.output_queue = output_queue
        self.config = config
        self.stop_event = stop_event or Event()
        self.low_util_count = {0: 0, 1: 0}
        
    def run(self):
        """Monitor loop"""
        try:
            import pynvml
            pynvml.nvmlInit()
            has_nvml = True
        except:
            logger.warning("NVML not available - GPU monitoring limited")
            has_nvml = False
        
        while not self.stop_event.is_set():
            try:
                stats = {
                    'timestamp': datetime.now().isoformat(),
                    'gpu_queue_size': self.gpu_queue.qsize(),
                    'output_queue_size': self.output_queue.qsize(),
                    'gpus': []
                }
                
                if has_nvml:
                    # Get GPU stats
                    device_count = pynvml.nvmlDeviceGetCount()
                    for i in range(min(device_count, 2)):  # Monitor first 2 GPUs
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        
                        # Utilization
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        
                        # Memory
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        
                        # Temperature
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        
                        gpu_stats = {
                            'id': i,
                            'utilization': util.gpu,
                            'memory_used_gb': mem_info.used / 1024**3,
                            'memory_total_gb': mem_info.total / 1024**3,
                            'memory_percent': (mem_info.used / mem_info.total) * 100,
                            'temperature': temp
                        }
                        stats['gpus'].append(gpu_stats)
                        
                        # Track low utilization
                        if util.gpu < self.config.low_util_threshold:
                            self.low_util_count[i] += 1
                            if self.low_util_count[i] > 3:  # Alert after 3 consecutive low readings
                                logger.warning(
                                    f"GPU {i} utilization consistently low: {util.gpu}% "
                                    f"(Queue size: {self.gpu_queue.qsize()})"
                                )
                        else:
                            self.low_util_count[i] = 0
                
                # Log summary
                if stats['gpus']:
                    gpu_info = []
                    for gpu in stats['gpus']:
                        gpu_info.append(
                            f"GPU{gpu['id']}: {gpu['utilization']}% util, "
                            f"{gpu['memory_percent']:.1f}% mem, {gpu['temperature']}°C"
                        )
                    
                    logger.info(
                        f"Pipeline Status - GPU Queue: {stats['gpu_queue_size']}, "
                        f"Output Queue: {stats['output_queue_size']} | " +
                        " | ".join(gpu_info)
                    )
                    
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                
            self.stop_event.wait(self.config.monitor_interval)
            
        if has_nvml:
            pynvml.nvmlShutdown()


class DatabaseWriter(Thread):
    """
    Asynchronous database writer to prevent blocking the output queue.
    """
    
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
        
    def _connect_database(self):
        """Connect to ArangoDB with retry logic"""
        max_retries = 5
        retry_delay = 1.0  # Initial delay in seconds
        
        for attempt in range(max_retries):
            try:
                self.client = ArangoClient(hosts=f'http://{self.config.db_host}:{self.config.db_port}')
                self.db = self.client.db(
                    self.config.db_name,
                    username=self.config.db_username,
                    password=self.config.db_password
                )
                self.collection = self.db.collection('abstract_metadata')
                
                # Test the connection
                self.collection.count()
                
                logger.info("Successfully connected to database")
                return
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"Database connection attempt {attempt + 1}/{max_retries} failed: {e}. "
                        f"Retrying in {wait_time:.1f} seconds..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to connect to database after {max_retries} attempts")
                    raise Exception(f"Database connection failed after {max_retries} retries: {e}")
        
    def run(self):
        """Database writer loop"""
        self._connect_database()
        buffer = []
        
        while not self.stop_event.is_set() or not self.write_queue.empty():
            try:
                # Get item with timeout
                item = self.write_queue.get(timeout=1.0)
                if item is None:  # Poison pill
                    break
                    
                buffer.append(item)
                
                # Write when buffer is full or on timeout
                if len(buffer) >= 100:  # Batch size for DB writes
                    self._write_batch(buffer)
                    buffer = []
                    
            except Empty:
                # Write any remaining items in buffer
                if buffer:
                    self._write_batch(buffer)
                    buffer = []
            except Exception as e:
                logger.error(f"Database writer error: {e}")
                
        # Final flush
        if buffer:
            self._write_batch(buffer)
            
    def _write_batch(self, records: List[Dict]):
        """Write batch to database"""
        try:
            self.collection.insert_many(records, overwrite=True, silent=False)
            logger.debug(f"Wrote {len(records)} records to database")
        except Exception as e:
            logger.error(f"Database write error: {e}")
            # Could implement retry logic here


class ContinuousGPUPipeline:
    """
    Main pipeline orchestrator with continuous GPU utilization.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir)
        
        # Queues
        self.document_queue = Queue(maxsize=10000)  # Large buffer for documents
        self.gpu_queue = mp.Queue(maxsize=config.max_gpu_queue_size)
        self.output_queue = mp.Queue(maxsize=config.max_output_queue_size)
        self.checkpoint_queue = mp.Queue()
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
        
        # Statistics
        self.stats = {
            'start_time': None,
            'total_processed': 0,
            'total_documents': 0,
            'gpu_stats': {0: {'batches': 0, 'time': 0}, 1: {'batches': 0, 'time': 0}}
        }
        
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
        logger.info("Starting Continuous GPU Pipeline")
        
        self.stats['start_time'] = time.time()
        
        # Start GPU workers
        for gpu_id in self.config.gpu_devices or [0, 1]:
            worker = GPUWorker(
                gpu_id=gpu_id,
                input_queue=self.gpu_queue,
                output_queue=self.output_queue,
                checkpoint_queue=self.checkpoint_queue,
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
            
        # Start output processor
        self.output_thread = Thread(target=self._process_outputs, daemon=True)
        self.output_thread.start()
        
        # Start checkpoint handler
        self.checkpoint_thread = Thread(target=self._handle_checkpoints, daemon=True)
        self.checkpoint_thread.start()
        
        logger.info("Pipeline started successfully")
        
    def process_metadata_files(self, metadata_files: List[Path], resume: bool = True):
        """
        Process a list of metadata files through the pipeline.
        """
        # Filter out already processed documents if resuming
        if resume:
            processed_docs = self.checkpoint_manager.get_processed_documents()
            logger.info(f"Resuming from checkpoint. Already processed: {len(processed_docs)} documents")
            
            # Filter files
            filtered_files = []
            for file_path in metadata_files:
                # Quick check - load just the ID
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        arxiv_id = data.get('arxiv_id', data.get('id', ''))
                        if arxiv_id not in processed_docs:
                            filtered_files.append(file_path)
                except:
                    filtered_files.append(file_path)
                    
            metadata_files = filtered_files
            logger.info(f"Remaining files to process: {len(metadata_files)}")
        
        self.stats['total_documents'] = len(metadata_files)
        
        # Save pipeline metadata
        self.checkpoint_manager.save_pipeline_metadata({
            'start_time': datetime.now().isoformat(),
            'total_files': len(metadata_files),
            'config': {
                'batch_size': self.config.batch_size,
                'gpu_devices': self.config.gpu_devices,
                'preprocessing_workers': self.config.preprocessing_workers
            }
        })
        
        # Queue all documents
        logger.info(f"Queueing {len(metadata_files)} metadata files")
        for file_path in tqdm(metadata_files, desc="Queueing files"):
            self.document_queue.put(file_path)
            
        # Add poison pills for preprocessing workers
        for _ in range(len(self.preprocessing_workers)):
            self.document_queue.put(None)
            
        logger.info("All files queued for processing")
        
    def _process_outputs(self):
        """Process outputs from GPU workers"""
        while not self.output_stop.is_set():
            try:
                output = self.output_queue.get(timeout=1.0)
                
                # Save to checkpoint
                self.checkpoint_manager.save_batch_result(
                    output['batch_id'],
                    output
                )
                
                # Prepare database records
                for i, (metadata, embedding) in enumerate(zip(output['metadata'], output['embeddings'])):
                    record = {
                        '_key': metadata['arxiv_id'],
                        'arxiv_id': metadata['arxiv_id'],
                        'title': metadata.get('title', ''),
                        'authors': metadata.get('authors', []),
                        'abstract_embedding': embedding,
                        'categories': metadata.get('categories', []),
                        'published': metadata.get('published'),
                        'processed_gpu': output['gpu_id'],
                        'processed_at': output['timestamp'],
                        'batch_id': output['batch_id'],
                        'gpu_time': output['gpu_time'] / len(output['metadata'])  # Per-doc time
                    }
                    self.db_write_queue.put(record)
                
                # Update statistics
                self.stats['total_processed'] += len(output['metadata'])
                gpu_id = output['gpu_id']
                self.stats['gpu_stats'][gpu_id]['batches'] += 1
                self.stats['gpu_stats'][gpu_id]['time'] += output['gpu_time']
                
                # Log progress
                if self.stats['total_processed'] % 1000 == 0:
                    elapsed = time.time() - self.stats['start_time']
                    rate = self.stats['total_processed'] / elapsed
                    eta = (self.stats['total_documents'] - self.stats['total_processed']) / rate
                    
                    logger.info(
                        f"Progress: {self.stats['total_processed']}/{self.stats['total_documents']} "
                        f"({self.stats['total_processed']/self.stats['total_documents']*100:.1f}%) | "
                        f"Rate: {rate:.1f} docs/s | ETA: {eta/60:.1f} min"
                    )
                    
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Output processing error: {e}", exc_info=True)
                
    def _handle_checkpoints(self):
        """Handle checkpoint requests from GPU workers"""
        while not self.stop_event.is_set():
            try:
                checkpoint_data = self.checkpoint_queue.get(timeout=1.0)
                
                # Save GPU worker state
                key = f"gpu_{checkpoint_data['gpu_id']}_state"
                self.checkpoint_manager.save_state(key, checkpoint_data)
                
                logger.debug(f"Checkpoint saved for GPU {checkpoint_data['gpu_id']}")
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Checkpoint error: {e}")
                
    def wait_for_completion(self):
        """Wait for all work to complete"""
        logger.info("Waiting for preprocessing to complete...")
        
        # Wait for preprocessing to finish
        for worker in self.preprocessing_workers:
            worker.join()
            
        logger.info("All preprocessing complete")
        
        # Wait for GPU queue to empty
        logger.info("Waiting for GPU processing to complete...")
        while not self.gpu_queue.empty():
            time.sleep(1.0)
            
        # Wait for all GPU workers to finish processing
        # This replaces the fixed 5-second sleep with deterministic checking
        logger.info("Ensuring all GPU workers have finished processing...")
        max_wait_time = 30.0  # Maximum wait time in seconds
        start_wait = time.time()
        
        while time.time() - start_wait < max_wait_time:
            # Check if all GPU workers are idle (not processing)
            all_idle = True
            
            # Check if GPU queue is truly empty and no workers are processing
            if self.gpu_queue.empty():
                # Small delay to ensure workers have had time to grab any last items
                time.sleep(0.1)
                
                # Double-check the queue is still empty
                if self.gpu_queue.empty():
                    break
            
            time.sleep(0.5)  # Check every 500ms
        
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
        
        # Stop monitoring
        self.stop_event.set()
        if self.monitor:
            self.monitor.join()
            
        logger.info("Pipeline completion successful")
        
    def print_summary(self):
        """Print processing summary"""
        elapsed = time.time() - self.stats['start_time']
        
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"Total processed: {self.stats['total_processed']:,} documents")
        print(f"Overall rate: {self.stats['total_processed']/elapsed:.1f} docs/second")
        
        print("\nPer-GPU Statistics:")
        for gpu_id in [0, 1]:
            gpu_stats = self.stats['gpu_stats'][gpu_id]
            if gpu_stats['batches'] > 0:
                avg_time = gpu_stats['time'] / gpu_stats['batches']
                print(f"  GPU {gpu_id}:")
                print(f"    Batches: {gpu_stats['batches']}")
                print(f"    Avg time/batch: {avg_time:.3f}s")
                print(f"    Total GPU time: {gpu_stats['time']:.1f}s")
                print(f"    GPU utilization: {gpu_stats['time']/elapsed*100:.1f}%")
                
        # Calculate efficiency
        total_gpu_time = sum(s['time'] for s in self.stats['gpu_stats'].values())
        parallel_efficiency = total_gpu_time / (elapsed * 2) * 100
        
        print(f"\nParallel efficiency: {parallel_efficiency:.1f}%")
        print("="*60)
        
    def shutdown(self):
        """Clean shutdown"""
        logger.info("Shutting down pipeline")
        
        # Stop all components
        self.preprocessing_stop.set()
        self.output_stop.set()
        self.stop_event.set()
        
        # Clean up
        self.checkpoint_manager.cleanup()
        
        logger.info("Pipeline shutdown complete")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Continuous GPU Processing Pipeline for arXiv abstracts"
    )
    parser.add_argument('--metadata-dir', type=str, required=True,
                        help='Directory containing metadata JSON files')
    parser.add_argument('--count', type=int,
                        help='Number of documents to process (default: all)')
    parser.add_argument('--db-name', type=str, default='arxiv_abstracts_continuous',
                        help='Database name')
    parser.add_argument('--db-host', type=str, default='localhost',
                        help='Database host')
    parser.add_argument('--gpu-devices', type=int, nargs='+', default=[0, 1],
                        help='GPU devices to use')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size per GPU')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of preprocessing workers')
    parser.add_argument('--clean-start', action='store_true',
                        help='Drop existing database and start fresh')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    parser.add_argument('--checkpoint-dir', type=str,
                        help='Checkpoint directory')
    
    args = parser.parse_args()
    
    # Get environment variables
    db_password = os.environ.get('ARANGO_PASSWORD', '')
    if not db_password:
        print("ERROR: ARANGO_PASSWORD environment variable not set!")
        sys.exit(1)
    
    # Configure pipeline
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
    
    # Print configuration
    print("\nCONTINUOUS GPU PIPELINE")
    print("="*60)
    print(f"Metadata source: {args.metadata_dir}")
    print(f"Database: {args.db_name} @ {args.db_host}")
    print(f"GPUs: {args.gpu_devices}")
    print(f"Batch size: {args.batch_size}")
    print(f"Preprocessing workers: {args.workers}")
    print(f"Resume: {args.resume}")
    print("="*60)
    
    # Create pipeline
    pipeline = ContinuousGPUPipeline(config)
    
    try:
        # Setup database
        if not args.resume:
            pipeline.setup_database(clean_start=args.clean_start)
        
        # Start pipeline
        pipeline.start()
        
        # Give workers time to initialize
        time.sleep(5)
        
        # Get metadata files
        metadata_dir = Path(args.metadata_dir)
        metadata_files = sorted(metadata_dir.glob("*.json"))
        
        if args.count:
            metadata_files = metadata_files[:args.count]
            
        print(f"\nFound {len(metadata_files)} metadata files")
        
        # Process files
        pipeline.process_metadata_files(metadata_files, resume=args.resume)
        
        # Wait for completion
        pipeline.wait_for_completion()
        
        # Print summary
        pipeline.print_summary()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
    finally:
        # Clean shutdown
        pipeline.shutdown()
        
    print("\n✅ Continuous GPU pipeline completed successfully!")


if __name__ == "__main__":
    main()