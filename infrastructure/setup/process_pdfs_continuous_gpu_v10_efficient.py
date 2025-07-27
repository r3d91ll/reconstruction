#!/usr/bin/env python3
"""
PDF Processing Pipeline V10 EFFICIENT - Balanced for RTX A6000 (48GB)
- Multiple workers per GPU with memory pools
- Pipeline parallelism for continuous GPU utilization  
- Larger batches with dynamic adjustment
- Zero-copy transfers between stages
"""

import os
import sys
import json
import time
import queue
import logging
import argparse
import warnings
import threading
import multiprocessing as mp
import multiprocessing.shared_memory as shm
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import deque

# Set GPU visibility BEFORE any imports
def set_worker_gpu(gpu_id: int):
    """Set GPU visibility for worker process"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Check if docling is installed
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
except ImportError:
    print("ERROR: Docling not installed. Install with: pip install docling")
    sys.exit(1)

from tqdm import tqdm
import lmdb
from arango import ArangoClient
import numpy as np

# GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except:
    NVML_AVAILABLE = False
    print("WARNING: pynvml not available, GPU monitoring disabled")

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline_v10_efficient.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Efficient configuration for RTX A6000 48GB"""
    # Directories
    pdf_dir: str = "/mnt/data/arxiv_data/pdf"
    checkpoint_dir: str = "checkpoints/pdf_pipeline_v10_efficient"
    
    # Database
    db_name: str = "irec_three_collections"
    db_host: str = "localhost"
    db_port: int = 8529
    db_username: str = "root"
    db_password: str = os.getenv("ARANGO_PASSWORD", "rootpassword")
    db_batch_size: int = 500  # Larger batches for efficiency
    
    # GPU Configuration - Multiple workers with memory pools
    docling_gpu: int = 0
    embedding_gpu: int = 1
    
    # Workers - More workers but controlled memory
    pdf_workers: int = 2  # 2 workers on GPU 0 (Docling is lighter)
    late_workers: int = 3  # 3 workers on GPU 1 (embedding is compute heavy)
    
    # Memory allocation per worker (total must be < 1.0)
    pdf_worker_memory: float = 0.4  # 40% each = 80% total for GPU 0
    late_worker_memory: float = 0.3  # 30% each = 90% total for GPU 1
    
    # Batching - Optimized for throughput
    embedding_batch_size: int = 8  # Per worker
    pdf_batch_size: int = 4  # Per worker
    
    # Pipeline settings
    prefetch_size: int = 50  # Pre-load PDFs
    pipeline_buffer_size: int = 100  # Buffer between stages
    
    # Late Chunking - Balanced settings
    max_context_length: int = 16384  # Reduced for better throughput
    chunk_size_tokens: int = 512
    chunk_stride_tokens: int = 256
    
    # Dynamic batching
    enable_dynamic_batching: bool = True
    batch_timeout_seconds: float = 2.0  # Process partial batches after timeout
    
    # Memory management
    max_document_chars: int = 300000  # Balanced for batch processing
    use_memory_pool: bool = True
    pool_size_gb: float = 10.0  # Memory pool size per GPU
    
    # Performance
    use_tf32: bool = True
    compile_model: bool = False
    zero_copy_transfer: bool = True
    use_quantization: bool = True  # 8-bit quantization for efficiency
    
    # Limits
    max_pdfs: Optional[int] = None
    max_file_size_mb: float = 75.0  # Slightly reduced for batch efficiency
    
    # Resume
    resume: bool = True
    clean_start: bool = False


@dataclass
class DocumentWork:
    """Document ready for late chunking"""
    arxiv_id: str
    full_text: str
    metadata: Dict[str, Any]
    extraction_time: float
    char_count: int
    
    
@dataclass 
class LateChunkOutput:
    """Output from late chunking"""
    arxiv_id: str
    chunk_embeddings: List[np.ndarray]
    chunk_texts: List[str]
    chunk_metadata: List[Dict[str, Any]]
    total_tokens: int
    processing_time: float


class GPUMemoryPool:
    """Pre-allocated memory pool for stable multi-worker operation"""
    
    def __init__(self, gpu_id: int, pool_size_gb: float):
        self.gpu_id = gpu_id
        self.pool_size_gb = pool_size_gb
        self.allocated_buffers = {}
        self.free_buffers = queue.Queue()
        self.lock = threading.Lock()
        
        logger.info(f"Initializing memory pool for GPU {gpu_id} with {pool_size_gb}GB")
        
    def initialize(self):
        """Initialize memory pool (must be called in worker process)"""
        import torch
        
        # Pre-allocate buffers
        buffer_size = 100 * 1024 * 1024  # 100MB buffers
        num_buffers = int(self.pool_size_gb * 1024 / 100)
        
        for i in range(num_buffers):
            try:
                buffer = torch.cuda.ByteTensor(buffer_size)
                self.free_buffers.put(buffer)
            except Exception as e:
                logger.warning(f"Could only allocate {i} buffers: {e}")
                break
                
        logger.info(f"Memory pool initialized with {self.free_buffers.qsize()} buffers")
        
    def get_buffer(self, size_bytes: int):
        """Get buffer from pool"""
        import torch
        
        try:
            if size_bytes <= 100 * 1024 * 1024:  # Use pool for small allocations
                buffer = self.free_buffers.get_nowait()
                return buffer[:size_bytes]
        except queue.Empty:
            pass
            
        # Fallback to direct allocation
        return torch.cuda.ByteTensor(size_bytes)
        
    def return_buffer(self, buffer):
        """Return buffer to pool"""
        if buffer.numel() == 100 * 1024 * 1024:
            try:
                self.free_buffers.put_nowait(buffer)
            except queue.Full:
                pass  # Let it be garbage collected
                
    def cleanup(self):
        """Release all allocated GPU buffers"""
        import torch
        
        with self.lock:
            # Track buffers to release
            released_count = 0
            
            # Clear all free buffers
            while not self.free_buffers.empty():
                try:
                    buffer = self.free_buffers.get_nowait()
                    del buffer
                    released_count += 1
                except queue.Empty:
                    break
                    
            # Clear allocated buffers
            for buffer_id in list(self.allocated_buffers.keys()):
                try:
                    del self.allocated_buffers[buffer_id]
                    released_count += 1
                except Exception as e:
                    logger.warning(f"Failed to release buffer {buffer_id}: {e}")
                    
            # Force garbage collection and clear GPU cache
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            logger.info(f"GPU memory pool cleanup: released {released_count} buffers")
            
    def get_stats(self):
        """Get pool statistics"""
        return {
            'free_buffers': self.free_buffers.qsize(),
            'allocated_buffers': len(self.allocated_buffers),
            'pool_size_gb': self.pool_size_gb
        }


class ZeroCopyQueue:
    """Efficient queue using shared memory for large arrays"""
    
    def __init__(self, maxsize: int):
        self.queue = mp.Queue(maxsize)
        self.shm_refs = {}
        self.lock = threading.Lock()
        
    def put_array(self, array: np.ndarray, metadata: dict):
        """Put numpy array using shared memory"""
        # Validate array size
        MAX_ARRAY_SIZE_GB = 2.0  # Maximum 2GB per array
        array_size_gb = array.nbytes / (1024**3)
        
        if array_size_gb > MAX_ARRAY_SIZE_GB:
            raise ValueError(f"Array too large: {array_size_gb:.2f}GB exceeds limit of {MAX_ARRAY_SIZE_GB}GB")
        
        # For small arrays, use regular queue
        if array.nbytes < 1024 * 1024:  # < 1MB
            self.queue.put({'array': array, 'metadata': metadata, 'zero_copy': False})
            return
            
        try:
            # Create shared memory
            shm_buffer = shm.SharedMemory(create=True, size=array.nbytes)
            shm_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm_buffer.buf)
            shm_array[:] = array[:]
            
            # Send reference
            ref = {
                'shm_name': shm_buffer.name,
                'shape': array.shape,
                'dtype': str(array.dtype),
                'metadata': metadata,
                'zero_copy': True
            }
            
            with self.lock:
                self.shm_refs[shm_buffer.name] = shm_buffer
                
            self.queue.put(ref)
            
        except Exception as e:
            logger.warning(f"Zero-copy failed, using regular transfer: {e}")
            # Cleanup any partially created shared memory
            if 'shm_buffer' in locals():
                try:
                    shm_buffer.close()
                    shm_buffer.unlink()
                except:
                    pass
            # Fallback to regular transfer
            self.queue.put({'array': array, 'metadata': metadata, 'zero_copy': False})
            
    def get_array(self) -> Tuple[np.ndarray, dict]:
        """Get array from shared memory or regular queue"""
        ref = self.queue.get()
        
        if not ref.get('zero_copy', False):
            return ref['array'], ref['metadata']
            
        try:
            # Access shared memory
            shm_buffer = shm.SharedMemory(name=ref['shm_name'])
            array = np.ndarray(ref['shape'], dtype=ref['dtype'], buffer=shm_buffer.buf)
            
            # Copy data (recipient should copy if needed)
            array_copy = array.copy()
            
            # Clean up
            shm_buffer.close()
            with self.lock:
                if ref['shm_name'] in self.shm_refs:
                    self.shm_refs[ref['shm_name']].close()
                    self.shm_refs[ref['shm_name']].unlink()
                    del self.shm_refs[ref['shm_name']]
                    
            return array_copy, ref['metadata']
            
        except Exception as e:
            logger.error(f"Failed to access shared memory: {e}")
            raise
            
    def cleanup(self):
        """Clean up shared memory"""
        with self.lock:
            for shm_name, shm_buffer in list(self.shm_refs.items()):
                try:
                    shm_buffer.close()
                    shm_buffer.unlink()
                except Exception as e:
                    logger.warning(f"Failed to cleanup shared memory {shm_name}: {e}")
                finally:
                    # Always remove from dict even if cleanup fails
                    del self.shm_refs[shm_name]
            
            # Also clean any remaining items in queue
            while not self.queue.empty():
                try:
                    ref = self.queue.get_nowait()
                    if ref.get('zero_copy', False) and 'shm_name' in ref:
                        try:
                            temp_shm = shm.SharedMemory(name=ref['shm_name'])
                            temp_shm.close()
                            temp_shm.unlink()
                        except:
                            pass
                except:
                    break


class PipelineCoordinator:
    """Coordinate pipeline stages for maximum throughput"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        # Create pipeline stages with zero-copy queues
        self.stages = {
            'prefetch': mp.Queue(config.prefetch_size),
            'extraction': mp.Queue(config.pipeline_buffer_size),
            'embedding': ZeroCopyQueue(config.pipeline_buffer_size),
            'database': ZeroCopyQueue(config.pipeline_buffer_size * 2)
        }
        
        # Work assignment queues for load balancing
        self.pdf_assignments = [mp.Queue() for _ in range(config.pdf_workers)]
        self.embed_assignments = [mp.Queue() for _ in range(config.late_workers)]
        
        # Worker load tracking
        self.pdf_worker_loads = mp.Array('i', config.pdf_workers)
        self.embed_worker_loads = mp.Array('i', config.late_workers)
        
        # Statistics
        self.stats = {
            'pdfs_queued': mp.Value('i', 0),
            'pdfs_extracted': mp.Value('i', 0),
            'docs_embedded': mp.Value('i', 0),
            'chunks_written': mp.Value('i', 0)
        }
        
    def distribute_pdf_work(self):
        """Distribute PDF work to least loaded workers"""
        while True:
            try:
                pdf_path = self.stages['prefetch'].get(timeout=0.1)
                
                if pdf_path is None:
                    # Signal end to all PDF workers
                    for q in self.pdf_assignments:
                        q.put(None)
                    break
                    
                # Find least loaded worker
                min_load_idx = min(
                    range(self.config.pdf_workers),
                    key=lambda i: self.pdf_worker_loads[i]
                )
                
                self.pdf_assignments[min_load_idx].put(pdf_path)
                self.pdf_worker_loads[min_load_idx] += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"PDF distribution error: {e}")
                
    def distribute_embedding_work(self):
        """Distribute embedding work to least loaded workers"""
        while True:
            try:
                doc = self.stages['extraction'].get(timeout=0.1)
                
                if doc is None:
                    # Signal end to all embedding workers
                    for q in self.embed_assignments:
                        q.put(None)
                    break
                    
                # Find least loaded worker
                min_load_idx = min(
                    range(self.config.late_workers),
                    key=lambda i: self.embed_worker_loads[i]
                )
                
                self.embed_assignments[min_load_idx].put(doc)
                self.embed_worker_loads[min_load_idx] += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Embedding distribution error: {e}")


class CheckpointManager:
    """Manages processing checkpoints"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.checkpoint_dir / "checkpoints.lmdb"
        
        # Initialize LMDB
        self.env = lmdb.open(
            str(self.db_path),
            map_size=2 * 1024 * 1024 * 1024,  # 2GB
            max_dbs=3
        )
        
        # Create sub-databases
        with self.env.begin(write=True) as txn:
            self.processed_db = self.env.open_db(b'processed', txn=txn)
            self.failed_db = self.env.open_db(b'failed', txn=txn)
            self.stats_db = self.env.open_db(b'stats', txn=txn)
            
        logger.info(f"Checkpoint database initialized at {self.db_path}")
        
    def is_processed(self, arxiv_id: str) -> bool:
        """Check if PDF has been processed"""
        with self.env.begin() as txn:
            return txn.get(arxiv_id.encode(), db=self.processed_db) is not None
            
    def mark_pdf_processed(self, arxiv_id: str, chunks: int, tokens: int):
        """Mark PDF as processed"""
        data = json.dumps({
            'chunks': chunks,
            'tokens': tokens,
            'timestamp': datetime.now().isoformat()
        })
        
        with self.env.begin(write=True) as txn:
            txn.put(
                arxiv_id.encode(),
                data.encode(),
                db=self.processed_db
            )
            
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        with self.env.begin() as txn:
            processed = sum(1 for _ in txn.cursor(db=self.processed_db))
            failed = sum(1 for _ in txn.cursor(db=self.failed_db))
            
        return {
            'processed': processed,
            'failed': failed,
            'total': processed + failed
        }


def pdf_worker_process_efficient(
    worker_id: int,
    gpu_id: int,
    memory_fraction: float,
    work_queue: mp.Queue,
    output_queue: mp.Queue,
    checkpoint_dir: str,
    config: PipelineConfig,
    worker_loads: mp.Array,
    stats: dict,
    stop_event: mp.Event
):
    """PDF worker with memory pool and efficient batching"""
    set_worker_gpu(gpu_id)
    
    import torch
    
    # Set memory fraction for this worker
    torch.cuda.set_per_process_memory_fraction(memory_fraction)
    
    logger.info(f"PDF worker {worker_id} starting on GPU {gpu_id} with {memory_fraction*100}% memory")
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(checkpoint_dir)
    
    # Initialize memory pool
    memory_pool = GPUMemoryPool(gpu_id, config.pool_size_gb / config.pdf_workers)
    memory_pool.initialize()
    
    # Initialize Docling
    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True
    )
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )
    
    logger.info(f"Worker {worker_id} initialized Docling")
    
    # Batch accumulator
    batch = []
    last_batch_time = time.time()
    
    try:
        while not stop_event.is_set():
            # Collect batch
            while len(batch) < config.pdf_batch_size:
                try:
                    pdf_path = work_queue.get(timeout=0.1)
                    if pdf_path is None:  # Sentinel value - process remaining and exit
                        if batch:
                            process_pdf_batch_efficient(
                                converter, batch, output_queue, checkpoint_manager,
                                config, worker_id, memory_pool, worker_loads, stats
                            )
                        return
                    batch.append(pdf_path)
                except queue.Empty:
                    if time.time() - last_batch_time > config.batch_timeout_seconds and batch:
                        break
                        
            if batch:
                # Process batch efficiently
                process_pdf_batch_efficient(
                    converter, batch, output_queue, checkpoint_manager,
                    config, worker_id, memory_pool, worker_loads, stats
                )
                
                batch = []
                last_batch_time = time.time()
                
    finally:
        # Cleanup on exit
        try:
            # Process any remaining batch
            if batch:
                logger.info(f"Processing final batch of {len(batch)} PDFs before exit")
                process_pdf_batch_efficient(
                    converter, batch, output_queue, checkpoint_manager,
                    config, worker_id, memory_pool, worker_loads, stats
                )
        except Exception as e:
            logger.error(f"Error processing final batch: {e}")
            
        # Clean up memory pool
        if memory_pool:
            memory_pool.cleanup()
            
        logger.info(f"PDF worker {worker_id} stopped and cleaned up")


def process_pdf_batch_efficient(
    converter, batch, output_queue, checkpoint_manager, 
    config, worker_id, memory_pool, worker_loads, stats
):
    """Process batch of PDFs efficiently"""
    logger.info(f"Worker {worker_id} processing batch of {len(batch)} PDFs")
    
    for pdf_path in batch:
        try:
            arxiv_id = Path(pdf_path).stem
            
            # Update load tracking
            worker_loads[worker_id] -= 1
            
            # Check if already processed
            if checkpoint_manager.is_processed(arxiv_id):
                logger.info(f"Skipping already processed: {arxiv_id}")
                continue
                
            # Check file size
            file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
            if file_size_mb > config.max_file_size_mb:
                logger.warning(f"Skipping {arxiv_id}: file too large ({file_size_mb:.1f} MB)")
                continue
                
            # Extract text
            start_time = time.time()
            result = converter.convert(pdf_path)
            
            # Get full text
            full_text = result.document.export_to_markdown()
            
            if not full_text or len(full_text) < 100:
                logger.warning(f"Skipping {arxiv_id}: insufficient text")
                continue
                
            # Check document size
            if len(full_text) > config.max_document_chars:
                logger.warning(f"Skipping {arxiv_id}: text too long ({len(full_text)} chars)")
                continue
                
            extraction_time = time.time() - start_time
            
            # Create work item
            doc_work = DocumentWork(
                arxiv_id=arxiv_id,
                full_text=full_text,
                metadata={
                    'title': getattr(result.document, 'title', ''),
                    'num_pages': len(result.document.pages),
                    'extraction_time': extraction_time
                },
                extraction_time=extraction_time,
                char_count=len(full_text)
            )
            
            # Queue for late chunking
            output_queue.put(doc_work)
            stats['pdfs_extracted'].value += 1
            
            logger.info(
                f"Extracted {arxiv_id}: {len(full_text)} chars "
                f"(~{len(full_text)//4} tokens) in {extraction_time:.1f}s"
            )
            
        except Exception as e:
            logger.error(f"Error extracting {pdf_path}: {e}")


def late_chunking_worker_efficient(
    worker_id: int,
    gpu_id: int,
    memory_fraction: float,
    work_queue: mp.Queue,
    output_queue: ZeroCopyQueue,
    config: PipelineConfig,
    worker_loads: mp.Array,
    stats: dict,
    stop_event: mp.Event
):
    """Efficient late chunking with batched processing"""
    set_worker_gpu(gpu_id)
    
    import torch
    from transformers import AutoTokenizer, AutoModel
    
    torch.cuda.set_per_process_memory_fraction(memory_fraction)
    
    logger.info(f"Late worker {worker_id} starting on GPU {gpu_id} with {memory_fraction*100}% memory")
    
    # Memory optimization
    torch.cuda.empty_cache()
    
    # Enable optimizations
    if config.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
    # Initialize Jina v4
    model_name = "jinaai/jina-embeddings-v4"
    
    # Load model with quantization if enabled
    if config.use_quantization:
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                quantization_config=quantization_config,
                device_map={'': gpu_id}
            )
            logger.info(f"Successfully loaded model with 8-bit quantization")
        except ImportError as e:
            logger.warning(f"BitsAndBytesConfig not available: {e}")
            logger.warning("Quantization requires 'bitsandbytes' package. Install with: pip install bitsandbytes")
            logger.info("Falling back to fp16 model without quantization")
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16
            ).cuda()
        except Exception as e:
            logger.warning(f"Quantization failed with error: {e}")
            logger.info("Falling back to fp16 model")
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16
            ).cuda()
    else:
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).cuda()
    
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Initialize batch processor
    batch_processor = EfficientBatchProcessor(model, tokenizer, config)
    
    logger.info(f"Worker {worker_id} initialized for late chunking")
    
    # Process batches
    while not stop_event.is_set():
        batch = collect_batch(
            work_queue, config.embedding_batch_size, 
            config.batch_timeout_seconds, worker_id, worker_loads
        )
        
        if batch is None:
            break
            
        if batch:
            try:
                # Process multiple documents in single forward pass
                results = batch_processor.process_batch(batch)
                
                for result in results:
                    # Use zero-copy transfer for embeddings
                    for i, embedding in enumerate(result.chunk_embeddings):
                        output_queue.put_array(
                            embedding,
                            {
                                'arxiv_id': result.arxiv_id,
                                'chunk_idx': i,
                                'chunk_text': result.chunk_texts[i],
                                'chunk_metadata': result.chunk_metadata[i],
                                'total_chunks': len(result.chunk_embeddings),
                                'total_tokens': result.total_tokens,
                                'processing_time': result.processing_time
                            }
                        )
                    
                    stats['docs_embedded'].value += 1
                    
            except torch.cuda.OutOfMemoryError:
                logger.error(f"OOM in worker {worker_id}, clearing cache")
                torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Batch processing error in worker {worker_id}: {e}")
                
    logger.info(f"Late worker {worker_id} stopped")


def collect_batch(work_queue, batch_size, timeout, worker_id, worker_loads):
    """Collect batch with timeout"""
    batch = []
    start_time = time.time()
    
    while len(batch) < batch_size:
        try:
            doc = work_queue.get(timeout=0.1)
            
            if doc is None:
                return None  # Shutdown signal
                
            batch.append(doc)
            worker_loads[worker_id] -= 1
            
        except queue.Empty:
            if time.time() - start_time > timeout:
                break
                
    return batch


class EfficientBatchProcessor:
    """Process multiple documents in single forward pass"""
    
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
    def process_batch(self, documents: List[DocumentWork]) -> List[LateChunkOutput]:
        """Process batch with padding and attention masks"""
        import torch
        
        start_time = time.time()
        
        # Limit text length for batch processing
        texts = []
        for doc in documents:
            # Truncate to reasonable length for batching
            max_chars = min(len(doc.full_text), 100000)
            texts.append(doc.full_text[:max_chars])
        
        # Tokenize all documents at once
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            max_length=self.config.max_context_length,
            truncation=True,
            padding=True,
            return_offsets_mapping=True
        )
        
        # Move to GPU
        inputs = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in inputs.items()}
        
        # Single forward pass for all documents
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    task_label='retrieval'
                )
        
        # Extract chunks for each document
        results = []
        batch_time = time.time() - start_time
        
        for i, doc in enumerate(documents):
            try:
                # Get embeddings for this document
                # Handle different output formats
                if hasattr(outputs, 'multi_vec_emb'):
                    embeddings = outputs.multi_vec_emb[i]
                else:
                    embeddings = outputs.last_hidden_state[i]
                
                # Get valid tokens (not padding)
                attention_mask = inputs['attention_mask'][i]
                valid_length = attention_mask.sum().item()
                embeddings = embeddings[:valid_length]
                
                # Get offset mapping
                offset_mapping = inputs['offset_mapping'][i].cpu().numpy()
                offset_mapping = offset_mapping[:valid_length]
                
                # Extract chunks
                chunks = self.extract_chunks(
                    embeddings, offset_mapping, texts[i], doc.arxiv_id
                )
                
                results.append(LateChunkOutput(
                    arxiv_id=doc.arxiv_id,
                    chunk_embeddings=[c['embedding'] for c in chunks],
                    chunk_texts=[c['text'] for c in chunks],
                    chunk_metadata=[c['metadata'] for c in chunks],
                    total_tokens=sum(c['metadata']['tokens'] for c in chunks),
                    processing_time=batch_time / len(documents)
                ))
                
                logger.info(
                    f"Processed {doc.arxiv_id}: {len(chunks)} chunks, "
                    f"{sum(c['metadata']['tokens'] for c in chunks)} tokens"
                )
                
            except Exception as e:
                logger.error(f"Error processing document {doc.arxiv_id}: {e}")
                
        return results
        
    def extract_chunks(self, embeddings, offset_mapping, text, arxiv_id):
        """Extract chunks from embeddings with thorough validation"""
        chunks = []
        seq_len = embeddings.shape[0]
        text_len = len(text)
        
        chunk_size = self.config.chunk_size_tokens
        stride = self.config.chunk_stride_tokens
        
        # Validate inputs
        if seq_len == 0 or text_len == 0:
            logger.warning(f"Empty input for {arxiv_id}: seq_len={seq_len}, text_len={text_len}")
            return chunks
            
        if chunk_size <= 0 or stride <= 0:
            logger.error(f"Invalid chunk parameters: size={chunk_size}, stride={stride}")
            return chunks
        
        for start_idx in range(0, seq_len - chunk_size + 1, stride):
            end_idx = min(start_idx + chunk_size, seq_len)
            
            # Validate token indices
            if start_idx < 0 or end_idx > seq_len or start_idx >= end_idx:
                logger.warning(f"Invalid token indices for {arxiv_id}: start={start_idx}, end={end_idx}, seq_len={seq_len}")
                continue
            
            # Mean pool tokens
            try:
                chunk_embedding = embeddings[start_idx:end_idx].mean(dim=0)
            except Exception as e:
                logger.error(f"Error pooling embeddings for {arxiv_id}: {e}")
                continue
            
            # Get text boundaries with validation
            try:
                # Ensure indices are within offset_mapping bounds
                if start_idx >= len(offset_mapping) or end_idx - 1 >= len(offset_mapping):
                    logger.warning(f"Token indices out of offset_mapping bounds for {arxiv_id}")
                    continue
                    
                start_char = int(offset_mapping[start_idx][0])
                end_char = int(offset_mapping[end_idx - 1][1])
                
                # Validate character indices
                if start_char < 0 or end_char > text_len:
                    logger.warning(f"Character indices out of bounds for {arxiv_id}: start_char={start_char}, end_char={end_char}, text_len={text_len}")
                    continue
                    
                if start_char >= end_char:
                    logger.warning(f"Invalid character range for {arxiv_id}: start_char={start_char} >= end_char={end_char}")
                    continue
                    
            except (IndexError, TypeError) as e:
                logger.error(f"Error accessing offset_mapping for {arxiv_id}: {e}")
                continue
            
            # Extract text
            try:
                chunk_text = text[start_char:end_char]
            except Exception as e:
                logger.error(f"Error extracting text for {arxiv_id}: {e}")
                continue
            
            # Validate chunk text
            if not chunk_text or not chunk_text.strip():
                continue
                
            chunks.append({
                'embedding': chunk_embedding.cpu().numpy(),
                'text': chunk_text,
                'metadata': {
                    'arxiv_id': arxiv_id,
                    'start_token': start_idx,
                    'end_token': end_idx,
                    'start_char': start_char,
                    'end_char': end_char,
                    'tokens': end_idx - start_idx
                }
            })
            
        return chunks


class DatabaseWriter:
    """High-throughput database writer"""
    
    def __init__(self, config: PipelineConfig, checkpoint_manager: CheckpointManager):
        self.config = config
        self.checkpoint_manager = checkpoint_manager
        self.client = None
        self.db = None
        self.collections = {}
        
        # Chunk accumulator
        self.chunk_accumulator = {}
        self.batch_buffer = []
        
        # Stats
        self.stats = {
            'total_documents': 0,
            'total_chunks': 0,
            'total_tokens': 0,
            'start_time': time.time()
        }
        
        # Initialize database
        self._initialize_db()
        
    def _initialize_db(self) -> bool:
        """Initialize database connection"""
        try:
            self.client = ArangoClient(hosts=f'http://{self.config.db_host}:{self.config.db_port}')
            self.db = self.client.db(
                self.config.db_name,
                username=self.config.db_username,
                password=self.config.db_password
            )
            
            self.collections = {
                'chunks': self.db.collection('chunks')
            }
            
            logger.info(f"Chunks collection has {self.collections['chunks'].count()} documents")
                
            return True
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            return False
            
    def process_outputs(self, input_queue: ZeroCopyQueue, coordinator: PipelineCoordinator, stop_event: mp.Event):
        """Process outputs with high throughput"""
        logger.info("Database Writer started")
        
        last_log_time = time.time()
        log_interval = 30
        
        while not stop_event.is_set() or self.chunk_accumulator:
            try:
                # Get chunk with embedding
                embedding, metadata = input_queue.get_array()
                
                arxiv_id = metadata['arxiv_id']
                
                # Accumulate chunks for document
                if arxiv_id not in self.chunk_accumulator:
                    self.chunk_accumulator[arxiv_id] = {
                        'chunks': [],
                        'total_chunks': metadata['total_chunks'],
                        'total_tokens': metadata['total_tokens'],
                        'processing_time': metadata['processing_time']
                    }
                    
                self.chunk_accumulator[arxiv_id]['chunks'].append({
                    'idx': metadata['chunk_idx'],
                    'embedding': embedding,
                    'text': metadata['chunk_text'],
                    'metadata': metadata['chunk_metadata']
                })
                
                # Check if document is complete
                if len(self.chunk_accumulator[arxiv_id]['chunks']) == metadata['total_chunks']:
                    self._process_complete_document(arxiv_id)
                    coordinator.stats['chunks_written'].value += metadata['total_chunks']
                    
                # Periodic progress logging
                if time.time() - last_log_time > log_interval:
                    self._log_progress()
                    last_log_time = time.time()
                    
            except queue.Empty:
                # Flush any remaining data
                self._flush_batch()
            except Exception as e:
                logger.error(f"Database writer error: {e}")
                
        # Final flush
        self._flush_batch()
        self._log_progress()
        
        logger.info(
            f"Database Writer stopped. "
            f"Documents: {self.stats['total_documents']}, "
            f"Chunks: {self.stats['total_chunks']}"
        )
        
    def _process_complete_document(self, arxiv_id: str):
        """Process a complete document"""
        doc_data = self.chunk_accumulator[arxiv_id]
        
        # Sort chunks by index
        chunks = sorted(doc_data['chunks'], key=lambda x: x['idx'])
        
        # Create chunk records
        for chunk in chunks:
            chunk_record = {
                '_key': f"{arxiv_id}_chunk_{chunk['idx']:04d}",
                'chunk_id': f"{arxiv_id}_chunk_{chunk['idx']:04d}",
                'arxiv_id': arxiv_id,
                'text': chunk['text'],
                'embedding': chunk['embedding'].tolist(),
                'chunk_index': chunk['idx'],
                'chunk_metadata': chunk['metadata'],
                'late_chunking': True,
                'processing_time': doc_data['processing_time'],
                'processed_at': datetime.now().isoformat()
            }
            
            self.batch_buffer.append(chunk_record)
            
        # Update stats
        self.stats['total_documents'] += 1
        self.stats['total_chunks'] += len(chunks)
        self.stats['total_tokens'] += doc_data['total_tokens']
        
        # Mark as processed
        self.checkpoint_manager.mark_pdf_processed(
            arxiv_id, len(chunks), doc_data['total_tokens']
        )
        
        # Remove from accumulator
        del self.chunk_accumulator[arxiv_id]
        
        # Check if batch is full
        if len(self.batch_buffer) >= self.config.db_batch_size:
            self._flush_batch()
            
    def _flush_batch(self):
        """Write batch to database"""
        if not self.batch_buffer:
            return
            
        try:
            self.collections['chunks'].insert_many(self.batch_buffer, overwrite=True)
            logger.info(f"Wrote {len(self.batch_buffer)} chunks to database")
            self.batch_buffer = []
        except Exception as e:
            logger.error(f"Failed to write batch: {e}")
            self.batch_buffer = []
            
    def _log_progress(self):
        """Log processing progress"""
        elapsed = time.time() - self.stats['start_time']
        docs_per_min = (self.stats['total_documents'] / elapsed) * 60 if elapsed > 0 else 0
        chunks_per_min = (self.stats['total_chunks'] / elapsed) * 60 if elapsed > 0 else 0
        
        logger.info(
            f"Progress: {self.stats['total_documents']} docs, "
            f"{self.stats['total_chunks']} chunks "
            f"({docs_per_min:.1f} docs/min, {chunks_per_min:.0f} chunks/min)"
        )


class EfficientPipeline:
    """Main pipeline with maximum efficiency"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir)
        
        # Setup multiprocessing
        mp.set_start_method('spawn', force=True)
        
        # Create coordinator
        self.coordinator = PipelineCoordinator(config)
        
        # Control
        self.stop_event = mp.Event()
        
        # Workers and threads
        self.workers = []
        self.threads = []
        
        # Stats
        self.start_time = None
        
    def setup_database(self) -> bool:
        """Setup database"""
        try:
            client = ArangoClient(hosts=f'http://{self.config.db_host}:{self.config.db_port}')
            sys_db = client.db('_system', username=self.config.db_username, password=self.config.db_password)
            
            if not sys_db.has_database(self.config.db_name):
                sys_db.create_database(self.config.db_name)
                logger.info(f"Created database: {self.config.db_name}")
            else:
                logger.info(f"Using existing database: {self.config.db_name}")
                
            # Connect to database
            db = client.db(self.config.db_name, username=self.config.db_username, password=self.config.db_password)
            
            # Create collections
            if not db.has_collection('chunks'):
                db.create_collection('chunks')
                logger.info(f"Created collection: chunks")
                    
            return True
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            return False
            
    def start(self):
        """Start the pipeline"""
        self.start_time = time.time()
        
        # Start distribution threads
        logger.info("Starting work distribution threads")
        
        pdf_dist_thread = threading.Thread(
            target=self.coordinator.distribute_pdf_work,
            daemon=True
        )
        pdf_dist_thread.start()
        self.threads.append(pdf_dist_thread)
        
        embed_dist_thread = threading.Thread(
            target=self.coordinator.distribute_embedding_work,
            daemon=True
        )
        embed_dist_thread.start()
        self.threads.append(embed_dist_thread)
        
        # Start PDF workers
        logger.info(f"Starting {self.config.pdf_workers} PDF workers on GPU {self.config.docling_gpu}")
        for i in range(self.config.pdf_workers):
            p = mp.Process(
                target=pdf_worker_process_efficient,
                args=(
                    i, self.config.docling_gpu, self.config.pdf_worker_memory,
                    self.coordinator.pdf_assignments[i], 
                    self.coordinator.stages['extraction'],
                    self.config.checkpoint_dir, self.config,
                    self.coordinator.pdf_worker_loads,
                    self.coordinator.stats,
                    self.stop_event
                )
            )
            p.start()
            self.workers.append(p)
            
        # Start embedding workers
        logger.info(f"Starting {self.config.late_workers} embedding workers on GPU {self.config.embedding_gpu}")
        for i in range(self.config.late_workers):
            p = mp.Process(
                target=late_chunking_worker_efficient,
                args=(
                    i, self.config.embedding_gpu, self.config.late_worker_memory,
                    self.coordinator.embed_assignments[i],
                    self.coordinator.stages['database'],
                    self.config,
                    self.coordinator.embed_worker_loads,
                    self.coordinator.stats,
                    self.stop_event
                )
            )
            p.start()
            self.workers.append(p)
            
        # Start database writer
        self.db_writer = DatabaseWriter(self.config, self.checkpoint_manager)
        self.db_thread = threading.Thread(
            target=self.db_writer.process_outputs,
            args=(self.coordinator.stages['database'], self.coordinator, self.stop_event)
        )
        self.db_thread.start()
        
        logger.info("Efficient pipeline started successfully")
        
    def queue_pdfs(self):
        """Queue PDFs for processing"""
        pdf_dir = Path(self.config.pdf_dir)
        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        
        if self.config.max_pdfs:
            pdf_files = pdf_files[:self.config.max_pdfs]
            
        logger.info(f"Processing {len(pdf_files)} PDFs")
        
        # Queue PDFs for processing
        for pdf_path in tqdm(pdf_files, desc="Queueing PDFs"):
            if not self.config.resume or not self.checkpoint_manager.is_processed(pdf_path.stem):
                self.coordinator.stages['prefetch'].put(str(pdf_path))
                self.coordinator.stats['pdfs_queued'].value += 1
                
        # Signal end of PDFs
        self.coordinator.stages['prefetch'].put(None)
        
    def monitor_progress(self):
        """Monitor pipeline progress"""
        last_stats = {
            'extracted': 0,
            'embedded': 0,
            'written': 0
        }
        
        while any(w.is_alive() for w in self.workers):
            time.sleep(10)
            
            current_stats = {
                'extracted': self.coordinator.stats['pdfs_extracted'].value,
                'embedded': self.coordinator.stats['docs_embedded'].value,
                'written': self.coordinator.stats['chunks_written'].value
            }
            
            # Calculate rates
            extract_rate = (current_stats['extracted'] - last_stats['extracted']) / 10.0
            embed_rate = (current_stats['embedded'] - last_stats['embedded']) / 10.0
            
            logger.info(
                f"Pipeline Status - "
                f"Extracted: {current_stats['extracted']} ({extract_rate:.1f}/s), "
                f"Embedded: {current_stats['embedded']} ({embed_rate:.1f}/s), "
                f"Chunks Written: {current_stats['written']}"
            )
            
            # Check worker loads
            pdf_loads = list(self.coordinator.pdf_worker_loads)
            embed_loads = list(self.coordinator.embed_worker_loads)
            
            logger.info(
                f"Worker Loads - PDF: {pdf_loads}, Embedding: {embed_loads}"
            )
            
            last_stats = current_stats
            
    def wait_for_completion(self):
        """Wait for all work to complete"""
        # Monitor progress in separate thread
        monitor_thread = threading.Thread(target=self.monitor_progress, daemon=True)
        monitor_thread.start()
        
        # Wait for all workers
        for worker in self.workers:
            worker.join()
            
        # Signal database writer to stop
        self.stop_event.set()
        self.db_thread.join()
        
        # Clean up
        self.coordinator.stages['embedding'].cleanup()
        self.coordinator.stages['database'].cleanup()
        
        # Final stats
        total_time = time.time() - self.start_time
        final_stats = self.checkpoint_manager.get_stats()
        
        logger.info(f"""
============================================================
Pipeline completed in {total_time:.1f} seconds
Processed: {final_stats['processed']} PDFs
Rate: {final_stats['processed'] / total_time:.2f} PDFs/second
Peak Performance: {self.coordinator.stats['pdfs_extracted'].value / total_time:.2f} extractions/second
============================================================
""")


def main():
    parser = argparse.ArgumentParser(
        description="PDF Processing V10 EFFICIENT - Maximum throughput"
    )
    
    # Directories
    parser.add_argument("--pdf-dir", default="/mnt/data/arxiv_data/pdf")
    
    # Database
    parser.add_argument("--db-name", default="irec_three_collections")
    parser.add_argument("--db-host", default="localhost")
    
    # GPU
    parser.add_argument("--docling-gpu", type=int, default=0)
    parser.add_argument("--embedding-gpu", type=int, default=1)
    
    # Workers
    parser.add_argument("--pdf-workers", type=int, default=2)
    parser.add_argument("--late-workers", type=int, default=3)
    
    # Memory
    parser.add_argument("--pdf-worker-memory", type=float, default=0.4)
    parser.add_argument("--late-worker-memory", type=float, default=0.3)
    
    # Limits
    parser.add_argument("--max-pdfs", type=int, help="Maximum PDFs to process")
    
    # Control
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--clean-start", action="store_true")
    
    args = parser.parse_args()
    
    # Check environment
    if not os.getenv("ARANGO_PASSWORD"):
        print("ERROR: ARANGO_PASSWORD environment variable not set!")
        return
        
    # Validate GPU availability
    import torch
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"Found {gpu_count} GPU(s)")
        
        # Check docling GPU
        if args.docling_gpu >= gpu_count:
            logger.error(f"ERROR: Docling GPU {args.docling_gpu} not available. Only {gpu_count} GPU(s) found.")
            logger.error(f"Available GPUs: 0-{gpu_count-1}")
            return
            
        # Check embedding GPU
        if args.embedding_gpu >= gpu_count:
            logger.error(f"ERROR: Embedding GPU {args.embedding_gpu} not available. Only {gpu_count} GPU(s) found.")
            logger.error(f"Available GPUs: 0-{gpu_count-1}")
            return
            
        # Log GPU info
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)")
    else:
        logger.error("ERROR: No CUDA GPUs available!")
        logger.error("This pipeline requires CUDA-capable GPUs.")
        return
        
    # Create config
    config = PipelineConfig(
        pdf_dir=args.pdf_dir,
        db_name=args.db_name,
        db_host=args.db_host,
        docling_gpu=args.docling_gpu,
        embedding_gpu=args.embedding_gpu,
        pdf_workers=args.pdf_workers,
        late_workers=args.late_workers,
        pdf_worker_memory=args.pdf_worker_memory,
        late_worker_memory=args.late_worker_memory,
        max_pdfs=args.max_pdfs,
        resume=args.resume and not args.clean_start,
        clean_start=args.clean_start
    )
    
    # Log configuration
    logger.info(f"""
============================================================
EFFICIENT Pipeline V10 - Maximum Throughput
============================================================
PDF Workers: {config.pdf_workers} on GPU {config.docling_gpu} ({config.pdf_worker_memory*100}% each)
Embedding Workers: {config.late_workers} on GPU {config.embedding_gpu} ({config.late_worker_memory*100}% each)
Total GPU Memory: {config.pdf_workers * config.pdf_worker_memory * 100}% (GPU 0), {config.late_workers * config.late_worker_memory * 100}% (GPU 1)
Max context: {config.max_context_length} tokens
Batch sizes: PDF={config.pdf_batch_size}, Embedding={config.embedding_batch_size}
Pipeline buffer: {config.pipeline_buffer_size}
Zero-copy transfer: {config.zero_copy_transfer}
Quantization: {config.use_quantization}
============================================================
""")
    
    # Initialize pipeline
    pipeline = EfficientPipeline(config)
    
    # Setup database
    if not pipeline.setup_database():
        logger.error("Failed to setup database")
        return
        
    try:
        # Start pipeline
        pipeline.start()
        
        # Queue PDFs
        pipeline.queue_pdfs()
        
        # Wait for completion
        pipeline.wait_for_completion()
        
        logger.info("✅ Efficient Pipeline Completed Successfully")
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
        pipeline.stop_event.set()
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        

if __name__ == "__main__":
    main()