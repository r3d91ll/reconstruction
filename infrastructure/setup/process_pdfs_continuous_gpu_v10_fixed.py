#!/usr/bin/env python3
"""
PDF Processing Pipeline V10 FIXED - Efficient and stable for RTX A6000
- Multiple workers per GPU without pickling issues
- Simple queue-based parallelism
- Optimized batching
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
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

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
        logging.FileHandler('pipeline_v10_fixed.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Efficient configuration for RTX A6000 48GB"""
    # Directories
    pdf_dir: str = "/mnt/data/arxiv_data/pdf"
    checkpoint_dir: str = "checkpoints/pdf_pipeline_v10_fixed"
    
    # Database
    db_name: str = "irec_three_collections"
    db_host: str = "localhost"
    db_port: int = 8529
    db_username: str = "root"
    db_password: str = os.getenv("ARANGO_PASSWORD", "")
    db_batch_size: int = 300  # Larger batches
    
    # GPU Configuration
    docling_gpu: int = 0
    embedding_gpu: int = 1
    
    # Workers - balanced for stability and throughput
    pdf_workers: int = 2  # 2 workers on GPU 0
    late_workers: int = 2  # 2 workers on GPU 1 (more stable than 3)
    
    # Memory allocation
    pdf_worker_memory: float = 0.45  # 45% each = 90% total
    late_worker_memory: float = 0.45  # 45% each = 90% total
    
    # Power management
    gpu_power_limit: int = 300000  # Milliwatts per GPU (default 300W = 300000mW)
    inter_batch_delay: float = 0.1  # Seconds between batches - small delay for stability
    thermal_check_interval: int = 30  # Check temperature every N seconds
    gradual_startup_delay: float = 2.0  # Delay between starting workers
    
    # Batching
    embedding_batch_size: int = 6  # Balanced batch size
    pdf_batch_size: int = 3  # Smaller PDF batches
    
    # Queue sizes
    pdf_queue_size: int = 100
    document_queue_size: int = 200
    output_queue_size: int = 300
    
    # Late Chunking
    max_context_length: int = 32768  # Balanced context
    chunk_size_tokens: int = 512
    chunk_stride_tokens: int = 256
    
    # Memory management
    max_document_chars: int = 300000
    batch_timeout_seconds: float = 3.0
    
    # Performance
    use_tf32: bool = True
    compile_model: bool = False
    
    # Limits
    max_pdfs: Optional[int] = None
    max_file_size_mb: float = 75.0
    
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


def pdf_worker_process(
    worker_id: int,
    gpu_id: int,
    memory_fraction: float,
    pdf_queue: mp.Queue,
    document_queue: mp.Queue,
    checkpoint_dir: str,
    config: PipelineConfig,
    stop_event: mp.Event,
    db_queue: mp.Queue = None
):
    """PDF extraction worker"""
    set_worker_gpu(gpu_id)
    
    # Set power limit
    if hasattr(config, 'gpu_power_limit'):
        try:
            set_gpu_power_limit(gpu_id, config.gpu_power_limit)
        except Exception as e:
            logger.warning(f"Failed to set GPU {gpu_id} power limit: {e}. Continuing without power management.")
    
    import torch
    
    # Set memory fraction
    torch.cuda.set_per_process_memory_fraction(memory_fraction)
    
    logger.info(f"PDF worker {worker_id} starting on GPU {gpu_id} with {memory_fraction*100}% memory")
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(checkpoint_dir)
    
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
    
    # Process PDFs
    batch = []
    last_batch_time = time.time()
    last_thermal_check = time.time()
    
    while not stop_event.is_set():
        try:
            # Collect batch
            while len(batch) < config.pdf_batch_size and not stop_event.is_set():
                try:
                    pdf_path = pdf_queue.get(timeout=0.5)
                    if pdf_path is None:
                        if batch:
                            process_pdf_batch(
                                converter, batch, document_queue, 
                                checkpoint_manager, config, worker_id, db_queue
                            )
                        return
                    batch.append(pdf_path)
                except queue.Empty:
                    if time.time() - last_batch_time > config.batch_timeout_seconds:
                        break
                        
            if batch:
                # Check temperature before processing
                if time.time() - last_thermal_check > getattr(config, 'thermal_check_interval', 30):
                    temp = monitor_gpu_temperature(gpu_id)
                    if temp > 80:
                        logger.warning(f"GPU {gpu_id} temperature high: {temp}°C, throttling...")
                        time.sleep(5)
                    last_thermal_check = time.time()
                
                process_pdf_batch(
                    converter, batch, document_queue, 
                    checkpoint_manager, config, worker_id, db_queue
                )
                batch = []
                last_batch_time = time.time()
                
                # Add inter-batch delay for power management
                if hasattr(config, 'inter_batch_delay'):
                    time.sleep(config.inter_batch_delay)
                
        except Exception as e:
            logger.error(f"PDF worker {worker_id} error: {e}")
            
    logger.info(f"PDF worker {worker_id} stopped")


def process_pdf_batch(converter, batch, document_queue, checkpoint_manager, config, worker_id, db_queue=None):
    """Process batch of PDFs"""
    logger.info(f"Worker {worker_id} processing batch of {len(batch)} PDFs")
    
    for pdf_path in batch:
        try:
            arxiv_id = Path(pdf_path).stem
            
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
            
            # Extract metadata
            metadata = {
                'title': getattr(result.document, 'title', ''),
                'num_pages': len(result.document.pages),
                'extraction_time': extraction_time
            }
            
            # Create metadata record for database
            metadata_record = {
                '_key': arxiv_id,
                'arxiv_id': arxiv_id,
                'title': metadata.get('title', f'Document {arxiv_id}'),
                'pdf_path': str(pdf_path),
                'page_count': metadata['num_pages'],
                'char_count': len(full_text),
                'extraction_date': datetime.now().isoformat(),
                'extraction_time': extraction_time
            }
            
            # Create document record for database
            document_record = {
                '_key': arxiv_id,
                'arxiv_id': arxiv_id,
                'full_text': full_text,
                'processed_at': datetime.now().isoformat()
            }
            
            # Send to database queue if available
            if db_queue:
                db_queue.put(('metadata', metadata_record))
                db_queue.put(('documents', document_record))
            
            # Create work item for chunking
            doc_work = DocumentWork(
                arxiv_id=arxiv_id,
                full_text=full_text,
                metadata=metadata,
                extraction_time=extraction_time,
                char_count=len(full_text)
            )
            
            # Queue for late chunking
            document_queue.put(doc_work)
            
            logger.info(
                f"Extracted {arxiv_id}: {len(full_text)} chars "
                f"(~{len(full_text)//4} tokens) in {extraction_time:.1f}s"
            )
            
        except Exception as e:
            logger.error(f"Error extracting {pdf_path}: {e}")


def set_gpu_power_limit(gpu_id: int, power_limit_mw: int):
    """Set GPU power limit to prevent power spikes
    
    Args:
        gpu_id: GPU index
        power_limit_mw: Power limit in milliwatts (e.g., 300000 for 300W)
    """
    # Validate inputs
    if not isinstance(gpu_id, int) or gpu_id < 0:
        logger.warning(f"Invalid GPU ID: {gpu_id}. Skipping power limit setting.")
        return
        
    if not isinstance(power_limit_mw, int) or power_limit_mw < 0:
        logger.warning(f"Invalid power limit: {power_limit_mw}. Skipping power limit setting.")
        return
        
    # Check GPU exists
    if NVML_AVAILABLE:
        try:
            gpu_count = pynvml.nvmlDeviceGetCount()
            if gpu_id >= gpu_count:
                logger.warning(f"GPU {gpu_id} not found (only {gpu_count} GPUs available). Skipping power limit setting.")
                return
                
            # Get GPU handle and check power management support
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            
            # Check if power management is supported
            try:
                min_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[0]
                max_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1]
                
                # Validate power limit is within acceptable range
                if power_limit_mw < min_limit or power_limit_mw > max_limit:
                    logger.warning(f"Power limit {power_limit_mw}mW ({power_limit_mw/1000}W) is outside valid range [{min_limit}mW, {max_limit}mW] for GPU {gpu_id}. Skipping.")
                    return
            except pynvml.NVMLError as e:
                logger.warning(f"GPU {gpu_id} does not support power management: {e}. Skipping power limit setting.")
                return
                
        except Exception as e:
            logger.warning(f"Error checking GPU {gpu_id}: {e}")
            
    try:
        import subprocess
        # nvidia-smi expects watts, so convert from milliwatts
        power_limit_watts = power_limit_mw // 1000
        cmd = f"nvidia-smi -i {gpu_id} -pl {power_limit_watts}"
        result = subprocess.run(cmd.split(), check=True, capture_output=True, text=True)
        logger.info(f"Set GPU {gpu_id} power limit to {power_limit_watts}W")
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        if "is not supported" in error_msg or "not supported for GPU" in error_msg:
            logger.warning(f"GPU {gpu_id} does not support power management. Continuing without power limits.")
        else:
            logger.warning(f"Could not set GPU {gpu_id} power limit: {error_msg}")
    except Exception as e:
        logger.warning(f"Could not set GPU power limit: {e}")


def monitor_gpu_temperature(gpu_id: int) -> float:
    """Monitor GPU temperature"""
    if NVML_AVAILABLE:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            return temp
        except:
            pass
    return 0


def late_chunking_worker_process(
    worker_id: int,
    gpu_id: int,
    memory_fraction: float,
    document_queue: mp.Queue,
    output_queue: mp.Queue,
    config: PipelineConfig,
    stop_event: mp.Event
):
    """Late chunking worker"""
    set_worker_gpu(gpu_id)
    
    # Set power limit
    if hasattr(config, 'gpu_power_limit'):
        try:
            set_gpu_power_limit(gpu_id, config.gpu_power_limit)
        except Exception as e:
            logger.warning(f"Failed to set GPU {gpu_id} power limit: {e}. Continuing without power management.")
    
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
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).cuda()
    
    model.eval()
    
    logger.info(f"Worker {worker_id} initialized for late chunking")
    
    # Process documents
    batch = []
    last_batch_time = time.time()
    last_thermal_check = time.time()
    
    while not stop_event.is_set():
        try:
            # Collect batch
            while len(batch) < config.embedding_batch_size and not stop_event.is_set():
                try:
                    doc = document_queue.get(timeout=0.5)
                    if doc is None:
                        if batch:
                            process_late_chunk_batch(
                                batch, model, tokenizer, output_queue, 
                                config, worker_id
                            )
                        return
                    batch.append(doc)
                except queue.Empty:
                    if time.time() - last_batch_time > config.batch_timeout_seconds:
                        break
                        
            if batch:
                # Check temperature before processing
                if time.time() - last_thermal_check > getattr(config, 'thermal_check_interval', 30):
                    temp = monitor_gpu_temperature(gpu_id)
                    if temp > 80:
                        logger.warning(f"GPU {gpu_id} temperature high: {temp}°C, throttling...")
                        time.sleep(5)
                    last_thermal_check = time.time()
                
                process_late_chunk_batch(
                    batch, model, tokenizer, output_queue, 
                    config, worker_id
                )
                batch = []
                last_batch_time = time.time()
                
                # Add inter-batch delay for power management
                if hasattr(config, 'inter_batch_delay'):
                    time.sleep(config.inter_batch_delay)
                
        except Exception as e:
            logger.error(f"Late worker {worker_id} error: {e}", exc_info=True)
            batch = []
            torch.cuda.empty_cache()
            
    logger.info(f"Late worker {worker_id} stopped")


def process_late_chunk_batch(batch, model, tokenizer, output_queue, config, worker_id):
    """Process batch of documents with late chunking"""
    import torch
    
    logger.info(f"Worker {worker_id} processing batch of {len(batch)} documents")
    start_time = time.time()
    
    for doc in batch:
        try:
            # Track time for this specific document
            doc_start_time = time.time()
            
            # Late chunk the document
            chunks = late_chunk_document(doc, model, tokenizer, config)
            
            # Create output
            output = LateChunkOutput(
                arxiv_id=doc.arxiv_id,
                chunk_embeddings=[c['embedding'] for c in chunks],
                chunk_texts=[c['text'] for c in chunks],
                chunk_metadata=[c['metadata'] for c in chunks],
                total_tokens=sum(c['metadata']['tokens'] for c in chunks),
                processing_time=time.time() - doc_start_time
            )
            
            output_queue.put(output)
            
            logger.info(
                f"Late chunked {doc.arxiv_id}: {len(chunks)} chunks, "
                f"{output.total_tokens} tokens"
            )
            
        except torch.cuda.OutOfMemoryError:
            logger.error(f"OOM for {doc.arxiv_id}, clearing cache")
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Late chunking failed for {doc.arxiv_id}: {e}")
            
    batch_time = time.time() - start_time
    logger.info(
        f"Worker {worker_id} batch of {len(batch)} docs "
        f"in {batch_time:.1f}s ({batch_time/len(batch):.1f}s per doc)"
    )


def late_chunk_document(doc, model, tokenizer, config):
    """Late chunking for a document"""
    import torch
    
    # Tokenize with limited context
    inputs = tokenizer(
        doc.full_text,
        return_tensors='pt',
        max_length=config.max_context_length,
        truncation=True,
        return_offsets_mapping=True,
        padding=True
    ).to('cuda')
    
    seq_len = inputs['input_ids'].shape[1]
    
    # Process with mixed precision
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            outputs = model(
                **{k: v for k, v in inputs.items() if k != 'offset_mapping'},
                task_label='retrieval'
            )
            all_token_embeddings = outputs.multi_vec_emb[0]
    
    # Extract chunks
    chunks = []
    offset_mapping = inputs['offset_mapping'][0].cpu().numpy()
    
    chunk_size = config.chunk_size_tokens
    stride = config.chunk_stride_tokens
    
    for start_idx in range(0, seq_len - chunk_size + 1, stride):
        end_idx = min(start_idx + chunk_size, seq_len)
        
        # Mean pool tokens
        chunk_embedding = all_token_embeddings[start_idx:end_idx].mean(dim=0)
        
        # Get text boundaries
        start_char = int(offset_mapping[start_idx][0])
        end_char = int(offset_mapping[end_idx - 1][1])
        
        if start_char < 0 or end_char > len(doc.full_text):
            continue
            
        chunk_text = doc.full_text[start_char:end_char]
        
        if not chunk_text.strip():
            continue
            
        chunks.append({
            'embedding': chunk_embedding.cpu().numpy(),
            'text': chunk_text,
            'metadata': {
                'start_token': start_idx,
                'end_token': end_idx,
                'start_char': start_char,
                'end_char': end_char,
                'tokens': end_idx - start_idx
            }
        })
        
    return chunks


class DatabaseWriter:
    """Database writer with batching"""
    
    def __init__(self, config: PipelineConfig, checkpoint_manager: CheckpointManager):
        self.config = config
        self.checkpoint_manager = checkpoint_manager
        self.client = None
        self.db = None
        self.collections = {}
        
        # Batch buffers for each collection
        self.batch_buffers = {
            'metadata': [],
            'documents': [],
            'chunks': []
        }
        
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
            
            # Initialize all three collections
            self.collections = {
                'metadata': self.db.collection('metadata'),
                'documents': self.db.collection('documents'),
                'chunks': self.db.collection('chunks')
            }
            
            # Log collection counts
            for name, collection in self.collections.items():
                logger.info(f"{name.capitalize()} collection has {collection.count()} documents")
                
            return True
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            return False
            
    def process_outputs(self, output_queue: mp.Queue, stop_event: mp.Event, db_queue: mp.Queue = None):
        """Process outputs"""
        logger.info("Database Writer started")
        
        last_log_time = time.time()
        log_interval = 30
        
        # Start a thread to handle db_queue if provided
        db_thread = None
        if db_queue:
            db_thread = threading.Thread(
                target=self._process_db_queue,
                args=(db_queue, stop_event),
                daemon=True
            )
            db_thread.start()
        
        while not stop_event.is_set() or not output_queue.empty():
            try:
                output = output_queue.get(timeout=1.0)
                
                if isinstance(output, LateChunkOutput):
                    self._process_late_chunks(output)
                    
                self._check_and_flush()
                
                # Log progress periodically
                if time.time() - last_log_time > log_interval:
                    self._log_progress()
                    last_log_time = time.time()
                    
            except queue.Empty:
                self._check_and_flush()
            except Exception as e:
                logger.error(f"Database writer error: {e}")
                
        self._flush_all_batches()
        self._log_progress()
        
        # Wait for db_thread if it exists
        if db_thread:
            db_thread.join()
        
        logger.info(
            f"Database Writer stopped. "
            f"Documents: {self.stats['total_documents']}, "
            f"Chunks: {self.stats['total_chunks']}"
        )
        
    def _log_progress(self):
        """Log processing progress"""
        elapsed = time.time() - self.stats['start_time']
        docs_per_min = (self.stats['total_documents'] / elapsed) * 60
        chunks_per_min = (self.stats['total_chunks'] / elapsed) * 60
        
        logger.info(
            f"Progress: {self.stats['total_documents']} docs, "
            f"{self.stats['total_chunks']} chunks "
            f"({docs_per_min:.1f} docs/min, {chunks_per_min:.0f} chunks/min)"
        )
        
    def _process_db_queue(self, db_queue: mp.Queue, stop_event: mp.Event):
        """Process items from db_queue (metadata and documents)"""
        while not stop_event.is_set() or not db_queue.empty():
            try:
                item = db_queue.get(timeout=1.0)
                if item is None:
                    break
                    
                collection_type, record = item
                if collection_type in self.batch_buffers:
                    self.batch_buffers[collection_type].append(record)
                    
                # Check if any buffer needs flushing
                for coll_name, buffer in self.batch_buffers.items():
                    if len(buffer) >= self.config.db_batch_size:
                        self._flush_batch(coll_name)
                        
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"DB queue processing error: {e}")
    
    def _process_late_chunks(self, output: LateChunkOutput):
        """Process late chunked output"""
        arxiv_id = output.arxiv_id
        
        # Update stats
        self.stats['total_chunks'] += len(output.chunk_embeddings)
        self.stats['total_documents'] += 1
        self.stats['total_tokens'] += output.total_tokens
        
        # Convert to records
        for i in range(len(output.chunk_texts)):
            chunk_record = {
                '_key': f"{arxiv_id}_chunk_{i:04d}",
                'chunk_id': f"{arxiv_id}_chunk_{i:04d}",
                'arxiv_id': arxiv_id,
                'text': output.chunk_texts[i],
                'embedding': output.chunk_embeddings[i].tolist(),
                'chunk_index': i,
                'chunk_metadata': output.chunk_metadata[i],
                'late_chunking': True,
                'processing_time': output.processing_time / len(output.chunk_texts),
                'processed_at': datetime.now().isoformat()
            }
            
            self.batch_buffers['chunks'].append(chunk_record)
            
        # Mark as processed
        self.checkpoint_manager.mark_pdf_processed(
            arxiv_id, 
            len(output.chunk_texts), 
            output.total_tokens
        )
        
    def _check_and_flush(self):
        """Flush buffers if full"""
        for collection_name, buffer in self.batch_buffers.items():
            if len(buffer) >= self.config.db_batch_size:
                self._flush_batch(collection_name)
    
    def _flush_all_batches(self):
        """Flush all buffers"""
        for collection_name, buffer in self.batch_buffers.items():
            if buffer:
                self._flush_batch(collection_name)
                
    def _flush_batch(self, collection_name: str):
        """Write batch to database with retry mechanism"""
        buffer = self.batch_buffers.get(collection_name, [])
        if not buffer:
            return
            
        collection = self.collections.get(collection_name)
        if not collection:
            logger.error(f"Collection {collection_name} not found")
            return
            
        max_retries = 3
        retry_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                collection.insert_many(buffer, overwrite=True)
                logger.info(f"Wrote {len(buffer)} records to {collection_name}")
                self.batch_buffers[collection_name] = []  # Clear buffer after successful write
                return
            except Exception as e:
                logger.error(f"Failed to write to {collection_name} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 1.5  # Exponential backoff
                else:
                    logger.error(f"Failed to write to {collection_name} after {max_retries} attempts, clearing buffer")
                    self.batch_buffers[collection_name] = []  # Clear buffer after exhausting retries


class FixedPipeline:
    """Main pipeline - fixed for stability"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir)
        
        # Setup multiprocessing
        mp.set_start_method('spawn', force=True)
        
        # Create queues
        self.pdf_queue = mp.Queue(maxsize=config.pdf_queue_size)
        self.document_queue = mp.Queue(maxsize=config.document_queue_size)
        self.output_queue = mp.Queue(maxsize=config.output_queue_size)
        self.db_queue = mp.Queue(maxsize=200)  # For metadata and documents
        
        # Control
        self.stop_event = mp.Event()
        
        # Workers
        self.workers = []
        
        # Stats
        self.start_time = None
        self.pdfs_queued = 0
        
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
            
            # Create all three collections
            collections_to_create = ['metadata', 'documents', 'chunks']
            
            for collection_name in collections_to_create:
                if not db.has_collection(collection_name):
                    db.create_collection(collection_name)
                    logger.info(f"Created collection: {collection_name}")
                else:
                    logger.info(f"Collection already exists: {collection_name}")
                    
            return True
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            return False
            
    def start(self):
        """Start the pipeline"""
        self.start_time = time.time()
        
        # Start PDF workers
        logger.info(f"Starting {self.config.pdf_workers} PDF workers on GPU {self.config.docling_gpu}")
        for i in range(self.config.pdf_workers):
            p = mp.Process(
                target=pdf_worker_process,
                args=(
                    i, self.config.docling_gpu, self.config.pdf_worker_memory,
                    self.pdf_queue, self.document_queue,
                    self.config.checkpoint_dir, self.config, self.stop_event,
                    self.db_queue
                )
            )
            p.start()
            self.workers.append(p)
            # Gradual startup to avoid power spikes
            if hasattr(self.config, 'gradual_startup_delay'):
                time.sleep(self.config.gradual_startup_delay)
            
        # Start embedding workers
        logger.info(f"Starting {self.config.late_workers} embedding workers on GPU {self.config.embedding_gpu}")
        for i in range(self.config.late_workers):
            p = mp.Process(
                target=late_chunking_worker_process,
                args=(
                    i, self.config.embedding_gpu, self.config.late_worker_memory,
                    self.document_queue, self.output_queue,
                    self.config, self.stop_event
                )
            )
            p.start()
            self.workers.append(p)
            # Gradual startup to avoid power spikes
            if hasattr(self.config, 'gradual_startup_delay'):
                time.sleep(self.config.gradual_startup_delay)
            
        # Start database writer
        self.db_writer = DatabaseWriter(self.config, self.checkpoint_manager)
        self.db_thread = threading.Thread(
            target=self.db_writer.process_outputs,
            args=(self.output_queue, self.stop_event, self.db_queue)
        )
        self.db_thread.start()
        
        logger.info("Pipeline started successfully")
        
    def queue_pdfs(self):
        """Queue PDFs for processing"""
        pdf_dir = Path(self.config.pdf_dir)
        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        
        if self.config.max_pdfs:
            pdf_files = pdf_files[:self.config.max_pdfs]
            
        logger.info(f"Processing {len(pdf_files)} PDFs")
        
        for pdf_path in tqdm(pdf_files, desc="Queueing PDFs"):
            if not self.config.resume or not self.checkpoint_manager.is_processed(pdf_path.stem):
                self.pdf_queue.put(str(pdf_path))
                self.pdfs_queued += 1
                
        # Signal end to PDF workers
        for _ in range(self.config.pdf_workers):
            self.pdf_queue.put(None)
            
    def wait_for_completion(self):
        """Wait for all work to complete"""
        # Monitor progress
        last_stats = self.checkpoint_manager.get_stats()
        
        while True:
            time.sleep(10)
            
            current_stats = self.checkpoint_manager.get_stats()
            
            # Check if PDF workers are done
            pdf_workers_done = all(not w.is_alive() for w in self.workers[:self.config.pdf_workers])
            
            if pdf_workers_done and self.document_queue.empty():
                # Signal embedding workers to stop
                for _ in range(self.config.late_workers):
                    self.document_queue.put(None)
                break
                
            # Log progress
            if current_stats['processed'] > last_stats['processed']:
                rate = (current_stats['processed'] - last_stats['processed']) / 10.0
                logger.info(
                    f"Progress: {current_stats['processed']}/{self.pdfs_queued} PDFs "
                    f"({rate:.1f} docs/sec)"
                )
            last_stats = current_stats
            
        # Wait for all workers
        for worker in self.workers:
            worker.join()
            
        # Stop database writer
        self.stop_event.set()
        self.db_thread.join()
        
        # Final stats
        total_time = time.time() - self.start_time
        final_stats = self.checkpoint_manager.get_stats()
        
        logger.info(f"""
============================================================
Pipeline completed in {total_time:.1f} seconds
Processed: {final_stats['processed']} PDFs
Rate: {final_stats['processed'] / total_time:.2f} PDFs/second
============================================================
""")


def main():
    parser = argparse.ArgumentParser(
        description="PDF Processing V10 FIXED - Efficient and stable"
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
    parser.add_argument("--late-workers", type=int, default=2)
    
    # Memory
    parser.add_argument("--pdf-worker-memory", type=float, default=0.45)
    parser.add_argument("--late-worker-memory", type=float, default=0.45)
    
    # Limits
    parser.add_argument("--max-pdfs", type=int, help="Maximum PDFs to process")
    
    # Control
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--clean-start", action="store_true")
    
    args = parser.parse_args()
    
    # Check environment
    arango_password = os.getenv("ARANGO_PASSWORD")
    if not arango_password:
        logger.error("ERROR: ARANGO_PASSWORD environment variable not set!")
        sys.exit(1)
        
    # Create config
    config = PipelineConfig(
        pdf_dir=args.pdf_dir,
        db_name=args.db_name,
        db_host=args.db_host,
        db_password=arango_password,
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
FIXED Pipeline V10 - Efficient and Stable
============================================================
PDF Workers: {config.pdf_workers} on GPU {config.docling_gpu} ({config.pdf_worker_memory*100}% each)
Embedding Workers: {config.late_workers} on GPU {config.embedding_gpu} ({config.late_worker_memory*100}% each)
Total GPU Memory: {config.pdf_workers * config.pdf_worker_memory * 100}% (GPU 0), {config.late_workers * config.late_worker_memory * 100}% (GPU 1)
Max context: {config.max_context_length} tokens
Batch sizes: PDF={config.pdf_batch_size}, Embedding={config.embedding_batch_size}
============================================================
""")
    
    # Initialize pipeline
    pipeline = FixedPipeline(config)
    
    # Setup database before starting workers
    if not pipeline.setup_database():
        logger.error("Failed to setup database")
        sys.exit(1)
        
    try:
        # Start pipeline
        pipeline.start()
        
        # Queue PDFs
        pipeline.queue_pdfs()
        
        # Wait for completion
        pipeline.wait_for_completion()
        
        logger.info("✅ Pipeline Completed Successfully")
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
        pipeline.stop_event.set()
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        

if __name__ == "__main__":
    main()