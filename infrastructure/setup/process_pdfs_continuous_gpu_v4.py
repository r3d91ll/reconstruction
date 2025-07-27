#!/usr/bin/env python3
"""
Production-Ready Continuous GPU Pipeline V4 - Full PDF Processing
Self-contained implementation with all components included.

Key improvements over V3:
- No external dependencies on other scripts
- Complete three-collection storage implementation
- Smart chunking that respects document structure
- Proper PDF and chunk tracking
- Memory management for large PDFs
"""

import os
import sys

# Set default Docling device to CPU to avoid GPU conflicts
os.environ.setdefault('DOCLING_DEVICE', 'cpu')  # lowercase required
os.environ.setdefault('DOCLING_NUM_THREADS', '12')  # Use 12 threads for better performance
import json
import logging
import torch
import torch.multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Protocol
from dataclasses import dataclass, field
from queue import Queue, Empty, PriorityQueue as ThreadPriorityQueue
from threading import Thread, Event, Lock
import numpy as np
from tqdm import tqdm
import time
import math
import gc
import heapq
import re
from arango import ArangoClient
import lmdb
from multiprocessing.managers import BaseManager
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Add infrastructure directory to path
infrastructure_dir = Path(__file__).parent.parent
sys.path.append(str(infrastructure_dir))

# Import Docling
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
except ImportError:
    print("ERROR: Docling not installed. Install with: pip install docling")
    sys.exit(1)

# Configure multiprocessing for CUDA
mp.set_start_method('spawn', force=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler('pdf_processing_pipeline_v4.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============== Shared Components (from V3) ==============

class PriorityQueue:
    """Priority queue that works with multiprocessing"""
    def __init__(self):
        self.heap = []
        self.counter = 0
        self.lock = mp.Lock()
        self.not_empty = mp.Condition(self.lock)
        
    def put(self, item, priority=1.0):
        with self.lock:
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


BaseManager.register('PriorityQueue', PriorityQueue)


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
                max_dbs=6,  # Added pdf_progress
                lock=True
            )
            
            # Create named databases
            with self.env.begin(write=True) as txn:
                self.progress_db = self.env.open_db(b'progress', txn=txn)
                self.state_db = self.env.open_db(b'state', txn=txn)
                self.metadata_db = self.env.open_db(b'metadata', txn=txn)
                self.failed_db = self.env.open_db(b'failed', txn=txn)
                self.processed_db = self.env.open_db(b'processed', txn=txn)
                self.pdf_progress_db = self.env.open_db(b'pdf_progress', txn=txn)
                
            logger.info(f"Checkpoint database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize checkpoint database: {e}")
            raise
            
    def save_json(self, key: str, value: Any, db=None):
        """Save data as JSON"""
        db = db or self.state_db
        with self.lock:
            try:
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
        
    def mark_pdf_processed(self, pdf_id: str, chunk_count: int):
        """Mark PDF as processed with chunk count"""
        self.save_json(f"pdf_{pdf_id}", {
            'processed': True,
            'chunk_count': chunk_count,
            'timestamp': datetime.now().isoformat()
        }, db=self.pdf_progress_db)
        
    def is_pdf_processed(self, pdf_id: str) -> bool:
        """Check if PDF is already processed"""
        return self.load_json(f"pdf_{pdf_id}", db=self.pdf_progress_db) is not None
        
    def cleanup(self):
        """Cleanup resources"""
        if self.env:
            self.env.close()


# ============== PDF-Specific Components ==============

@dataclass
class PDFWork:
    """Work item for PDF processing"""
    batch_id: str
    pdf_paths: List[Path]
    metadata: List[Dict]
    priority: float = 1.0
    retry_count: int = 0
    doc_count: int = 0
    
    def __post_init__(self):
        self.doc_count = len(self.pdf_paths)


@dataclass
class ChunkWork:
    """Work item for GPU processing (embeddings)"""
    batch_id: str
    texts: List[str]
    metadata: List[Dict]
    chunks: List[Dict]
    priority: float = 1.0
    retry_count: int = 0
    doc_count: int = 0
    
    def __post_init__(self):
        self.doc_count = len(self.chunks) if self.chunks else len(self.texts)


@dataclass
class PipelineConfig:
    """Pipeline configuration for PDF processing"""
    # GPU settings
    gpu_devices: List[int] = field(default_factory=lambda: [0, 1])
    batch_size: int = 32
    prefetch_factor: int = 2
    
    # Worker settings
    pdf_extraction_workers: int = 4
    preprocessing_workers: int = 4
    max_gpu_queue_size: Optional[int] = None
    max_output_queue_size: Optional[int] = None
    
    # Database
    db_host: str = "localhost"
    db_port: int = 8529
    db_name: str = "irec_three_collections"
    db_username: str = "root"
    db_password: str = ""
    db_batch_size: int = 100
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints/pdf_pipeline"
    checkpoint_interval: int = 50
    
    # Monitoring
    enable_monitoring: bool = True
    monitor_interval: float = 10.0
    low_util_threshold: float = 50.0
    low_util_alert_threshold: int = 3
    
    # Memory management
    max_embeddings_in_memory: int = 1000
    embedding_cleanup_interval: int = 25
    max_pdf_size_mb: int = 100  # Maximum PDF size to process
    
    # Retry settings
    max_retries: int = 3
    retry_delay_base: float = 2.0
    
    # Validation
    min_document_length: int = 100
    embedding_dimension: int = 2048
    
    # Chunking settings
    chunk_size: int = 1024
    chunk_overlap: int = 128
    max_chunk_size: int = 2048  # Maximum chunk size
    
    # PDF extraction settings
    use_ocr: bool = False
    extract_tables: bool = True
    extract_figures: bool = False
    force_cpu_extraction: bool = True  # Force CPU for PDF extraction to avoid GPU conflicts
    docling_cpu_threads: int = 12  # Number of CPU threads for Docling (12 out of 48 available)
    
    def __post_init__(self):
        """Validate and adjust configuration"""
        if self.max_gpu_queue_size is None:
            self.max_gpu_queue_size = min(
                self.batch_size * self.prefetch_factor * len(self.gpu_devices),
                100
            )
            
        if self.max_output_queue_size is None:
            self.max_output_queue_size = self.max_gpu_queue_size * 2
            
        # Validate GPU devices
        available_gpus = torch.cuda.device_count()
        for gpu_id in self.gpu_devices:
            if gpu_id >= available_gpus:
                raise ValueError(f"GPU {gpu_id} not available")


class PDFChunkCompletionTracker:
    """Track both PDF and chunk completion"""
    
    def __init__(self):
        self.lock = Lock()
        
        # PDF tracking
        self.pdfs_queued = 0
        self.pdfs_completed = 0
        self.pdfs_failed = 0
        self.pdf_chunk_counts = {}  # pdf_id -> expected chunks
        
        # Chunk tracking
        self.chunks_expected = 0
        self.chunks_completed = 0
        self.chunks_failed = 0
        self.chunks_in_flight = {}  # gpu_id -> set of (batch_id, chunk_count)
        
    def add_pdf(self, pdf_id: str):
        """Add a PDF to tracking"""
        with self.lock:
            self.pdfs_queued += 1
            self.pdf_chunk_counts[pdf_id] = 0
            
    def set_pdf_chunks(self, pdf_id: str, chunk_count: int):
        """Set expected chunks for a PDF"""
        with self.lock:
            self.pdf_chunk_counts[pdf_id] = chunk_count
            self.chunks_expected += chunk_count
            
    def complete_pdf(self, pdf_id: str):
        """Mark PDF as completed"""
        with self.lock:
            self.pdfs_completed += 1
            
    def fail_pdf(self, pdf_id: str):
        """Mark PDF as failed"""
        with self.lock:
            self.pdfs_failed += 1
            
    def start_chunk_batch(self, gpu_id: int, batch_id: str, chunk_count: int):
        """Track chunk batch processing"""
        with self.lock:
            if gpu_id not in self.chunks_in_flight:
                self.chunks_in_flight[gpu_id] = set()
            self.chunks_in_flight[gpu_id].add((batch_id, chunk_count))
            
    def complete_chunk_batch(self, gpu_id: int, batch_id: str, chunk_count: int):
        """Mark chunk batch as completed"""
        with self.lock:
            if gpu_id in self.chunks_in_flight:
                self.chunks_in_flight[gpu_id].discard((batch_id, chunk_count))
            self.chunks_completed += chunk_count
            
    def fail_chunk_batch(self, gpu_id: int, batch_id: str, chunk_count: int):
        """Mark chunk batch as failed"""
        with self.lock:
            if gpu_id in self.chunks_in_flight:
                self.chunks_in_flight[gpu_id].discard((batch_id, chunk_count))
            self.chunks_failed += chunk_count
            
    def is_complete(self) -> bool:
        """Check if all processing is complete"""
        with self.lock:
            chunks_in_flight_count = sum(
                count for gpu_chunks in self.chunks_in_flight.values()
                for _, count in gpu_chunks
            )
            return (self.pdfs_completed + self.pdfs_failed >= self.pdfs_queued and
                    chunks_in_flight_count == 0)
            
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        with self.lock:
            chunks_in_flight_count = sum(
                count for gpu_chunks in self.chunks_in_flight.values()
                for _, count in gpu_chunks
            )
            
            return {
                'pdfs': {
                    'queued': self.pdfs_queued,
                    'completed': self.pdfs_completed,
                    'failed': self.pdfs_failed,
                    'remaining': self.pdfs_queued - self.pdfs_completed - self.pdfs_failed
                },
                'chunks': {
                    'expected': self.chunks_expected,
                    'completed': self.chunks_completed,
                    'failed': self.chunks_failed,
                    'in_flight': chunks_in_flight_count,
                    'remaining': self.chunks_expected - self.chunks_completed - self.chunks_failed
                }
            }


class PDFExtractionWorker(Thread):
    """Worker for extracting text from PDFs using Docling"""
    
    def __init__(
        self,
        pdf_queue: Queue,
        text_queue: Queue,
        db_queue: Queue,  # For document storage
        config: PipelineConfig,
        stop_event: Event,
        worker_id: int,
        completion_tracker: PDFChunkCompletionTracker
    ):
        super().__init__(daemon=True)
        self.pdf_queue = pdf_queue
        self.text_queue = text_queue
        self.db_queue = db_queue
        self.config = config
        self.stop_event = stop_event
        self.worker_id = worker_id
        self.completion_tracker = completion_tracker
        self.processed_count = 0
        self.failed_count = 0
        
        # Store CPU extraction preference
        self.force_cpu_extraction = config.force_cpu_extraction
        if self.force_cpu_extraction:
            logger.info(f"PDF Worker {worker_id} will use CPU mode for extraction")
        
        # Initialize Docling
        self.converter = self._init_docling()
        
    def _init_docling(self):
        """Initialize Docling converter"""
        try:
            # Import necessary components
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            
            # Create pipeline options
            pipeline_options = PdfPipelineOptions()
            pipeline_options.use_ocr = self.config.use_ocr
            pipeline_options.extract_tables = self.config.extract_tables
            pipeline_options.extract_figures = self.config.extract_figures
            
            # Configure accelerator for CPU if requested
            if self.force_cpu_extraction:
                try:
                    from docling.datamodel.pipeline_options import AcceleratorOptions, AcceleratorDevice
                    
                    # Force CPU usage with specified threads
                    accelerator_options = AcceleratorOptions(
                        num_threads=self.config.docling_cpu_threads,  # Use configured CPU threads
                        device='cpu'  # Explicitly use CPU (lowercase)
                    )
                    pipeline_options.accelerator_options = accelerator_options
                    logger.info(f"PDF Worker {self.worker_id} configured for CPU-only with {self.config.docling_cpu_threads} threads")
                    
                except ImportError:
                    logger.warning(f"AcceleratorOptions not available, falling back to environment variables")
                    # Fallback to environment variables
                    os.environ['DOCLING_DEVICE'] = 'cpu'  # lowercase
                    os.environ['DOCLING_NUM_THREADS'] = str(self.config.docling_cpu_threads)
            
            # Initialize converter with options
            converter = DocumentConverter(
                pdf_options=pipeline_options
            )
            
            logger.info(f"PDF Worker {self.worker_id} initialized Docling with CPU={self.force_cpu_extraction}")
            return converter
            
        except Exception as e:
            logger.error(f"Failed to initialize Docling: {e}")
            # Fallback to basic initialization
            try:
                logger.info("Attempting fallback initialization...")
                converter = DocumentConverter()
                return converter
            except Exception as e2:
                logger.error(f"Fallback initialization also failed: {e2}")
                raise
            
    def run(self):
        """Process PDFs and extract text"""
        logger.info(f"PDF extraction worker {self.worker_id} started")
        
        while not self.stop_event.is_set():
            try:
                pdf_path = self.pdf_queue.get(timeout=1.0)
                if pdf_path is None:
                    break
                    
                # Check PDF size
                pdf_size_mb = pdf_path.stat().st_size / (1024 * 1024)
                if pdf_size_mb > self.config.max_pdf_size_mb:
                    logger.warning(
                        f"Skipping large PDF {pdf_path} "
                        f"({pdf_size_mb:.1f}MB > {self.config.max_pdf_size_mb}MB limit)"
                    )
                    self.completion_tracker.fail_pdf(pdf_path.stem)
                    self.failed_count += 1
                    continue
                    
                # Extract with retry
                extracted_data = self._extract_pdf_with_retry(pdf_path)
                
                if extracted_data:
                    # Store document and metadata
                    self._store_document_data(extracted_data)
                    
                    # Pass to chunking
                    self.text_queue.put(extracted_data)
                    self.processed_count += 1
                else:
                    self.completion_tracker.fail_pdf(pdf_path.stem)
                    self.failed_count += 1
                    
            except Empty:
                continue
            except Exception as e:
                logger.error(f"PDF worker {self.worker_id} error: {e}")
                
        logger.info(
            f"PDF worker {self.worker_id} done. "
            f"Processed: {self.processed_count}, Failed: {self.failed_count}"
        )
        
    def _extract_pdf_with_retry(self, pdf_path: Path) -> Optional[Dict]:
        """Extract PDF with retry and fallback strategies"""
        for attempt in range(self.config.max_retries):
            try:
                if attempt == 0:
                    # Fast extraction without OCR
                    result = self._extract_pdf(pdf_path, use_ocr=False)
                elif attempt == 1:
                    # Try with OCR if text extraction failed
                    logger.info(f"Retrying {pdf_path} with OCR enabled")
                    result = self._extract_pdf(pdf_path, use_ocr=True)
                else:
                    # Final attempt with minimal options
                    logger.info(f"Final attempt for {pdf_path} with minimal options")
                    result = self._extract_pdf(pdf_path, use_ocr=False, minimal=True)
                    
                if result:
                    return result
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed for {pdf_path}: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay_base ** attempt)
                    
        return None
        
    def _extract_pdf(self, pdf_path: Path, use_ocr: bool = None, minimal: bool = False) -> Optional[Dict]:
        """Extract text and metadata from PDF"""
        try:
            # Temporarily override OCR setting if specified
            if use_ocr is not None and hasattr(self.converter, 'format_options'):
                original_ocr = self.converter.format_options[InputFormat.PDF].use_ocr
                self.converter.format_options[InputFormat.PDF].use_ocr = use_ocr
                
            if minimal and hasattr(self.converter, 'format_options'):
                # Disable expensive features for problematic PDFs
                self.converter.format_options[InputFormat.PDF].extract_tables = False
                self.converter.format_options[InputFormat.PDF].extract_figures = False
                
            # Convert PDF
            start_time = time.time()
            result = self.converter.convert(str(pdf_path))
            extraction_time = time.time() - start_time
            
            # Restore settings
            if use_ocr is not None and hasattr(self.converter, 'format_options'):
                self.converter.format_options[InputFormat.PDF].use_ocr = original_ocr
            if minimal and hasattr(self.converter, 'format_options'):
                self.converter.format_options[InputFormat.PDF].extract_tables = self.config.extract_tables
                self.converter.format_options[InputFormat.PDF].extract_figures = self.config.extract_figures
                
            if not result or not hasattr(result, 'document'):
                logger.warning(f"No content extracted from {pdf_path}")
                return None
                
            # Export to markdown
            full_text = result.document.export_to_markdown()
            
            if not full_text or len(full_text.strip()) < self.config.min_document_length:
                logger.warning(f"Insufficient text from {pdf_path}: {len(full_text)} chars")
                return None
                
            # Extract metadata
            doc_metadata = self._extract_metadata(result)
            
            # Get arXiv ID from filename
            arxiv_id = pdf_path.stem
            
            return {
                'arxiv_id': arxiv_id,
                'pdf_path': str(pdf_path),
                'full_text': full_text,
                'metadata': doc_metadata,
                'extraction_time': extraction_time,
                'doc_structure': self._extract_structure(result),
                'extraction_metadata': {
                    'page_count': len(result.document.pages) if hasattr(result.document, 'pages') else 0,
                    'has_tables': bool(result.document.tables) if hasattr(result.document, 'tables') else False,
                    'char_count': len(full_text),
                    'extraction_method': 'docling',
                    'used_ocr': use_ocr or False,
                    'worker_id': self.worker_id,
                    'pdf_size_mb': pdf_path.stat().st_size / (1024 * 1024)
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting {pdf_path}: {e}")
            return None
            
    def _extract_metadata(self, result) -> Dict:
        """Extract document metadata from Docling result"""
        metadata = {}
        
        try:
            if hasattr(result.document, 'metadata'):
                doc_meta = result.document.metadata
                if hasattr(doc_meta, 'title'):
                    metadata['title'] = doc_meta.title
                if hasattr(doc_meta, 'authors'):
                    metadata['authors'] = doc_meta.authors
                if hasattr(doc_meta, 'date'):
                    metadata['date'] = str(doc_meta.date)
        except Exception as e:
            logger.debug(f"Error extracting metadata: {e}")
            
        return metadata
        
    def _extract_structure(self, result) -> Dict:
        """Extract document structure"""
        structure = {
            'sections': [],
            'subsections': [],
            'has_abstract': False,
            'has_introduction': False,
            'has_conclusion': False,
            'has_references': False
        }
        
        try:
            text = result.document.export_to_markdown()
            
            # Find section headers
            sections = re.findall(r'^#+\s+(.+)$', text, re.MULTILINE)
            structure['sections'] = sections[:30]  # Limit to first 30
            
            # Find subsections
            subsections = re.findall(r'^###\s+(.+)$', text, re.MULTILINE)
            structure['subsections'] = subsections[:50]
            
            # Check for common sections
            text_lower = text.lower()
            structure['has_abstract'] = bool(re.search(r'\babstract\b', text_lower[:5000]))
            structure['has_introduction'] = bool(re.search(r'\bintroduction\b', text_lower[:10000]))
            structure['has_conclusion'] = bool(re.search(r'\bconclusion\b', text_lower[-10000:]))
            structure['has_references'] = bool(re.search(r'\breferences\b', text_lower[-15000:]))
            
        except Exception as e:
            logger.debug(f"Error extracting structure: {e}")
            
        return structure
        
    def _store_document_data(self, doc_data: Dict):
        """Store document and metadata to database"""
        arxiv_id = doc_data['arxiv_id']
        
        # Prepare metadata record
        metadata_record = {
            '_key': arxiv_id,
            'arxiv_id': arxiv_id,
            'title': doc_data['metadata'].get('title', f'Document {arxiv_id}'),
            'authors': doc_data['metadata'].get('authors', []),
            'extraction_date': datetime.now().isoformat(),
            'pdf_path': doc_data['pdf_path'],
            'char_count': doc_data['extraction_metadata']['char_count'],
            'page_count': doc_data['extraction_metadata']['page_count'],
            'has_tables': doc_data['extraction_metadata']['has_tables'],
            'structure_summary': {
                'sections': len(doc_data['doc_structure']['sections']),
                'has_abstract': doc_data['doc_structure']['has_abstract'],
                'has_references': doc_data['doc_structure']['has_references']
            }
        }
        
        # Prepare document record
        document_record = {
            '_key': arxiv_id,
            'arxiv_id': arxiv_id,
            'full_text_markdown': doc_data['full_text'],
            'document_structure': doc_data['doc_structure'],
            'extraction_metadata': doc_data['extraction_metadata'],
            'extraction_time': doc_data['extraction_time'],
            'processed_at': datetime.now().isoformat()
        }
        
        # Send to database queue
        self.db_queue.put({
            'type': 'metadata',
            'data': metadata_record
        })
        
        self.db_queue.put({
            'type': 'documents',
            'data': document_record
        })
        
        logger.debug(f"Queued document and metadata for {arxiv_id}")


class SmartDocumentChunker(Thread):
    """Smart chunking that respects document structure"""
    
    def __init__(
        self,
        text_queue: Queue,
        gpu_queue,  # PriorityQueue proxy
        config: PipelineConfig,
        stop_event: Event,
        worker_id: int,
        completion_tracker: PDFChunkCompletionTracker,
        checkpoint_manager: SecureCheckpointManager
    ):
        super().__init__(daemon=True)
        self.text_queue = text_queue
        self.gpu_queue = gpu_queue
        self.config = config
        self.stop_event = stop_event
        self.worker_id = worker_id
        self.completion_tracker = completion_tracker
        self.checkpoint_manager = checkpoint_manager
        self.processed_count = 0
        
    def run(self):
        """Chunk documents intelligently"""
        batch_chunks = []
        batch_metadata = []
        batch_texts = []
        batch_num = 0
        
        while not self.stop_event.is_set():
            try:
                doc_data = self.text_queue.get(timeout=1.0)
                if doc_data is None:
                    if batch_chunks:
                        self._send_batch(batch_texts, batch_metadata, batch_chunks, batch_num)
                    break
                    
                # Smart chunking
                chunks = self._smart_chunk_document(doc_data)
                
                if not chunks:
                    logger.warning(f"No chunks created for {doc_data['arxiv_id']}")
                    self.completion_tracker.complete_pdf(doc_data['arxiv_id'])
                    continue
                    
                # Update tracker with chunk count
                self.completion_tracker.set_pdf_chunks(doc_data['arxiv_id'], len(chunks))
                
                # Add chunks to batch
                for chunk in chunks:
                    batch_chunks.append(chunk)
                    batch_texts.append(chunk['text'])
                    batch_metadata.append({
                        'arxiv_id': doc_data['arxiv_id'],
                        'chunk_id': chunk['chunk_id'],
                        'chunk_index': chunk['index'],
                        'doc_metadata': doc_data['metadata']
                    })
                    
                    # Send batch when full
                    if len(batch_chunks) >= self.config.batch_size:
                        self._send_batch(batch_texts, batch_metadata, batch_chunks, batch_num)
                        batch_chunks = []
                        batch_metadata = []
                        batch_texts = []
                        batch_num += 1
                        
                # Mark PDF as complete
                self.completion_tracker.complete_pdf(doc_data['arxiv_id'])
                self.checkpoint_manager.mark_pdf_processed(doc_data['arxiv_id'], len(chunks))
                self.processed_count += 1
                
            except Empty:
                if batch_chunks:
                    self._send_batch(batch_texts, batch_metadata, batch_chunks, batch_num)
                    batch_chunks = []
                    batch_metadata = []
                    batch_texts = []
                    batch_num += 1
                    
            except Exception as e:
                logger.error(f"Chunking worker {self.worker_id} error: {e}")
                
        logger.info(f"Chunking worker {self.worker_id} done. Processed: {self.processed_count} documents")
        
    def _smart_chunk_document(self, doc_data: Dict) -> List[Dict]:
        """Smart chunking that respects document structure"""
        chunks = []
        text = doc_data['full_text']
        structure = doc_data['doc_structure']
        arxiv_id = doc_data['arxiv_id']
        
        # Try section-based chunking first
        sections = self._split_into_sections(text, structure)
        
        if sections:
            # Chunk within sections
            chunk_index = 0
            for section_name, section_text in sections:
                section_chunks = self._chunk_text(
                    section_text,
                    arxiv_id,
                    chunk_index,
                    section_name=section_name
                )
                chunks.extend(section_chunks)
                chunk_index += len(section_chunks)
        else:
            # Fall back to regular chunking
            chunks = self._chunk_text(text, arxiv_id, 0)
            
        logger.debug(f"Created {len(chunks)} chunks for {arxiv_id} using {'section-based' if sections else 'regular'} chunking")
        return chunks
        
    def _split_into_sections(self, text: str, structure: Dict) -> List[Tuple[str, str]]:
        """Split document into sections based on structure"""
        sections = []
        
        if not structure.get('sections'):
            return sections
            
        # Find section boundaries
        section_pattern = r'^(#+\s+)(.+)$'
        matches = list(re.finditer(section_pattern, text, re.MULTILINE))
        
        if not matches:
            return sections
            
        # Extract sections
        for i, match in enumerate(matches):
            section_name = match.group(2).strip()
            start = match.end()
            
            # Find end of section
            if i < len(matches) - 1:
                end = matches[i + 1].start()
            else:
                end = len(text)
                
            section_text = text[start:end].strip()
            
            if len(section_text) > 50:  # Minimum section size
                sections.append((section_name, section_text))
                
        return sections
        
    def _chunk_text(self, text: str, arxiv_id: str, start_index: int, section_name: str = None) -> List[Dict]:
        """Chunk text with overlap and sentence boundaries"""
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        max_chunk = self.config.max_chunk_size
        
        # Split into sentences for better boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = []
        current_size = 0
        chunk_index = start_index
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # Check if adding sentence exceeds chunk size
            if current_size + sentence_size > chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                
                if len(chunk_text) > 50:  # Minimum chunk size
                    chunk_id = f"{arxiv_id}_chunk_{chunk_index:04d}"
                    
                    chunk_data = {
                        'chunk_id': chunk_id,
                        'arxiv_id': arxiv_id,
                        'text': chunk_text[:max_chunk],  # Enforce max size
                        'index': chunk_index,
                        'metadata': {
                            'chunk_size': len(chunk_text),
                            'section': section_name,
                            'overlapping': chunk_index > start_index
                        }
                    }
                    
                    chunks.append(chunk_data)
                    chunk_index += 1
                    
                    # Start new chunk with overlap
                    if overlap > 0:
                        # Keep last few sentences for overlap
                        overlap_sentences = int(len(current_chunk) * 0.2)  # 20% overlap
                        current_chunk = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
                        current_size = sum(len(s) for s in current_chunk)
                    else:
                        current_chunk = []
                        current_size = 0
                        
            current_chunk.append(sentence)
            current_size += sentence_size
            
        # Handle remaining text
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) > 50:
                chunk_id = f"{arxiv_id}_chunk_{chunk_index:04d}"
                
                chunks.append({
                    'chunk_id': chunk_id,
                    'arxiv_id': arxiv_id,
                    'text': chunk_text[:max_chunk],
                    'index': chunk_index,
                    'metadata': {
                        'chunk_size': len(chunk_text),
                        'section': section_name,
                        'overlapping': False
                    }
                })
                
        return chunks
        
    def _send_batch(self, texts: List[str], metadata: List[Dict], chunks: List[Dict], batch_num: int):
        """Send batch to GPU queue"""
        if not texts:
            return
            
        batch_id = f"chunk_w{self.worker_id}_b{batch_num}_{int(time.time() * 1000)}"
        
        work = ChunkWork(
            batch_id=batch_id,
            texts=texts,
            metadata=metadata,
            chunks=chunks,
            priority=1.0
        )
        
        self.gpu_queue.put(work, priority=work.priority)
        logger.debug(f"Chunking worker {self.worker_id} sent batch {batch_id} with {len(texts)} chunks")


class GPUWorker(mp.Process):
    """GPU worker for embedding generation"""
    
    def __init__(
        self,
        gpu_id: int,
        input_queue,
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
            warmup_result = model.encode_batch(["warmup"] * 10, batch_size=10)
            torch.cuda.empty_cache()
            logger.info(f"GPU Worker {self.gpu_id} initialized successfully")
        except Exception as e:
            logger.error(f"GPU Worker {self.gpu_id} failed to initialize: {e}")
            self.stats_queue.put(('worker_failed', self.gpu_id, None, 0))
            return
            
        # Processing loop
        while not self.stop_event.is_set():
            try:
                work = self.input_queue.get(timeout=1.0)
                
                if work is None or (hasattr(work, 'batch_id') and work.batch_id is None):
                    logger.info(f"GPU Worker {self.gpu_id} received shutdown signal")
                    break
                    
                # Notify stats queue
                self.stats_queue.put(('processing', self.gpu_id, work.batch_id, work.doc_count))
                
                # Process batch
                try:
                    result = self._process_batch(model, work)
                    
                    if result:
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
                
        logger.info(f"GPU Worker {self.gpu_id} shutting down. Processed {self.processed_count} batches")
        
    def _process_batch(self, model, work: ChunkWork) -> Optional[Dict]:
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
                'chunks': work.chunks,
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


class AtomicDatabaseWriter(Thread):
    """Database writer with atomic three-collection updates"""
    
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
        self.collections = {}
        
        self.batch_buffers = {
            'metadata': [],
            'documents': [],
            'chunks': []
        }
        self.written_counts = {
            'metadata': 0,
            'documents': 0,
            'chunks': 0
        }
        
    def run(self):
        """Main database writing loop"""
        logger.info("Atomic Database Writer started")
        
        if not self._initialize_db_with_retry():
            logger.error("Failed to initialize database")
            return
            
        while not self.stop_event.is_set():
            try:
                item = self.write_queue.get(timeout=1.0)
                if item is None:
                    self._flush_all_buffers()
                    break
                    
                # Route to appropriate buffer
                self._add_to_buffer(item)
                
                # Check if any buffer needs flushing
                self._check_and_flush_buffers()
                
            except Empty:
                self._check_and_flush_buffers()
                
            except Exception as e:
                logger.error(f"Database writer error: {e}")
                
        logger.info(f"Database Writer stopped. Written: {self.written_counts}")
        
    def _initialize_db_with_retry(self) -> bool:
        """Initialize with three collections"""
        for attempt in range(self.config.max_retries):
            try:
                self.client = ArangoClient(hosts=f'http://{self.config.db_host}:{self.config.db_port}')
                
                self.db = self.client.db(
                    self.config.db_name,
                    username=self.config.db_username,
                    password=self.config.db_password
                )
                
                # Get/create collections
                self.collections = {
                    'metadata': self.db.collection('metadata'),
                    'documents': self.db.collection('documents'),
                    'chunks': self.db.collection('chunks')
                }
                
                # Test connections
                for name, coll in self.collections.items():
                    coll.count()
                    
                logger.info("Database connection established for three collections")
                return True
                
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    wait_time = self.config.retry_delay_base ** attempt
                    logger.warning(f"DB init attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to initialize DB after {self.config.max_retries} attempts")
                    
        return False
        
    def _add_to_buffer(self, item: Dict):
        """Add item to appropriate buffer"""
        item_type = item.get('type', 'chunks')
        
        if item_type == 'metadata':
            self.batch_buffers['metadata'].append(item['data'])
        elif item_type == 'documents':
            self.batch_buffers['documents'].append(item['data'])
        elif item_type == 'chunks':
            # Process GPU output with embeddings
            self._process_chunk_batch(item)
            
    def _process_chunk_batch(self, gpu_output: Dict):
        """Process GPU output and prepare chunk records"""
        for i, chunk_data in enumerate(gpu_output['chunks']):
            chunk_record = {
                '_key': chunk_data['chunk_id'].replace('/', '_'),
                'chunk_id': chunk_data['chunk_id'],
                'arxiv_id': chunk_data['arxiv_id'],
                'text': chunk_data['text'],
                'embedding': gpu_output['embeddings'][i],
                'chunk_index': chunk_data['index'],
                'chunk_metadata': chunk_data['metadata'],
                'processed_gpu': gpu_output['gpu_id'],
                'processing_time': gpu_output['gpu_time'] / len(gpu_output['chunks']),
                'processed_at': gpu_output['timestamp']
            }
            
            # Validate before adding
            if self._validate_chunk(chunk_record):
                self.batch_buffers['chunks'].append(chunk_record)
                
    def _validate_chunk(self, chunk: Dict) -> bool:
        """Validate chunk record"""
        if not chunk.get('chunk_id') or not chunk.get('arxiv_id'):
            return False
            
        embedding = chunk.get('embedding', [])
        if not isinstance(embedding, list) or len(embedding) != self.config.embedding_dimension:
            return False
            
        try:
            if any(math.isnan(x) or math.isinf(x) for x in embedding):
                return False
        except:
            return False
            
        return True
        
    def _check_and_flush_buffers(self):
        """Check and flush buffers that are full"""
        for collection_name, buffer in self.batch_buffers.items():
            if len(buffer) >= self.config.db_batch_size:
                self._flush_buffer(collection_name)
                
    def _flush_all_buffers(self):
        """Flush all buffers"""
        for collection_name in self.batch_buffers:
            if self.batch_buffers[collection_name]:
                self._flush_buffer(collection_name)
                
    def _flush_buffer(self, collection_name: str):
        """Flush a specific buffer with retry"""
        buffer = self.batch_buffers[collection_name]
        if not buffer:
            return
            
        success = self._write_batch_with_retry(collection_name, buffer)
        
        if success:
            self.written_counts[collection_name] += len(buffer)
            self.batch_buffers[collection_name] = []
        else:
            logger.error(f"Failed to write {len(buffer)} records to {collection_name}")
            self.batch_buffers[collection_name] = []
            
    def _write_batch_with_retry(self, collection_name: str, documents: List[Dict]) -> bool:
        """Write batch with exponential backoff"""
        collection = self.collections[collection_name]
        
        for attempt in range(self.config.max_retries):
            try:
                result = collection.insert_many(
                    documents,
                    overwrite=True,
                    return_new=False,
                    silent=False
                )
                
                logger.debug(f"Wrote {len(documents)} documents to {collection_name}")
                return True
                
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    wait_time = self.config.retry_delay_base ** attempt
                    logger.warning(f"Write to {collection_name} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Write to {collection_name} failed after {self.config.max_retries} attempts: {e}")
                    
        return False


class PDFExtractionMonitor(Thread):
    """Monitor PDF extraction performance"""
    
    def __init__(
        self,
        pdf_queue: Queue,
        text_queue: Queue,
        config: PipelineConfig,
        stop_event: Event,
        extraction_times: List[float]
    ):
        super().__init__(daemon=True)
        self.pdf_queue = pdf_queue
        self.text_queue = text_queue
        self.config = config
        self.stop_event = stop_event
        self.extraction_times = extraction_times
        
    def run(self):
        """Monitor extraction performance"""
        logger.info("PDF Extraction Monitor started")
        
        while not self.stop_event.is_set():
            try:
                pdf_queue_size = self.pdf_queue.qsize()
                text_queue_size = self.text_queue.qsize()
                
                # Calculate average extraction time
                if self.extraction_times:
                    recent_times = self.extraction_times[-20:]  # Last 20
                    avg_time = sum(recent_times) / len(recent_times)
                    
                    logger.info(
                        f"PDF Extraction - Queue: {pdf_queue_size}, "
                        f"Extracted: {text_queue_size}, "
                        f"Avg time: {avg_time:.1f}s"
                    )
                    
                    # Alert on slow extraction
                    if avg_time > 30:
                        logger.warning(f"PDF extraction is slow: {avg_time:.1f}s average")
                        
                # Check for stalls
                if pdf_queue_size > 0 and text_queue_size == 0:
                    logger.warning("PDF extraction may be stalled")
                    
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                
            self.stop_event.wait(self.config.monitor_interval)


class PDFProcessingPipeline:
    """Main pipeline for processing PDF documents"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.checkpoint_manager = SecureCheckpointManager(config.checkpoint_dir)
        
        # Create manager for priority queue
        self.manager = BaseManager()
        self.manager.start()
        
        # Queues
        self.pdf_queue = Queue(maxsize=100)
        self.text_queue = Queue(maxsize=50)
        self.db_queue = Queue(maxsize=200)  # For document/metadata storage
        self.gpu_queue = self.manager.PriorityQueue()
        self.output_queue = mp.Queue(maxsize=config.max_output_queue_size)
        self.checkpoint_queue = mp.Queue()
        self.stats_queue = mp.Queue()
        self.db_write_queue = Queue(maxsize=100)
        
        # Events
        self.stop_event = mp.Event()
        self.extraction_stop = Event()
        self.chunking_stop = Event()
        self.output_stop = Event()
        
        # Workers
        self.pdf_workers = []
        self.chunking_workers = []
        self.gpu_workers = []
        self.db_writer = None
        self.extraction_monitor = None
        
        # Tracking
        self.completion_tracker = PDFChunkCompletionTracker()
        self.extraction_times = []
        
        # Stats
        self.stats = {
            'start_time': None,
            'total_pdfs': 0,
            'total_chunks': 0,
            'gpu_stats': {gpu: {'batches': 0, 'time': 0} for gpu in config.gpu_devices}
        }
        
    def setup_database(self, clean_start=False):
        """Setup three-collection database"""
        try:
            client = ArangoClient(hosts=f'http://{self.config.db_host}:{self.config.db_port}')
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
            
            # Create collections
            collections = {
                'metadata': {
                    'indexes': [
                        {'type': 'hash', 'fields': ['arxiv_id'], 'unique': True},
                        {'type': 'persistent', 'fields': ['extraction_date']},
                        {'type': 'persistent', 'fields': ['page_count']}
                    ]
                },
                'documents': {
                    'indexes': [
                        {'type': 'hash', 'fields': ['arxiv_id'], 'unique': True},
                        {'type': 'fulltext', 'fields': ['full_text_markdown'], 'minLength': 3}
                    ]
                },
                'chunks': {
                    'indexes': [
                        {'type': 'hash', 'fields': ['chunk_id'], 'unique': True},
                        {'type': 'persistent', 'fields': ['arxiv_id']},
                        {'type': 'persistent', 'fields': ['chunk_index']}
                    ]
                }
            }
            
            for coll_name, coll_config in collections.items():
                collection = db.create_collection(coll_name)
                for index in coll_config['indexes']:
                    collection.add_index(index)
                logger.info(f"Created collection '{coll_name}' with indexes")
                
            logger.info("Three-collection database setup complete")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            raise
            
    def start(self):
        """Start all pipeline components"""
        logger.info("Starting PDF Processing Pipeline V4")
        
        self.stats['start_time'] = time.time()
        
        # Start PDF extraction workers
        for i in range(self.config.pdf_extraction_workers):
            worker = PDFExtractionWorker(
                pdf_queue=self.pdf_queue,
                text_queue=self.text_queue,
                db_queue=self.db_queue,
                config=self.config,
                stop_event=self.extraction_stop,
                worker_id=i,
                completion_tracker=self.completion_tracker
            )
            worker.start()
            self.pdf_workers.append(worker)
            
        # Start chunking workers
        for i in range(self.config.preprocessing_workers):
            worker = SmartDocumentChunker(
                text_queue=self.text_queue,
                gpu_queue=self.gpu_queue,
                config=self.config,
                stop_event=self.chunking_stop,
                worker_id=i,
                completion_tracker=self.completion_tracker,
                checkpoint_manager=self.checkpoint_manager
            )
            worker.start()
            self.chunking_workers.append(worker)
            
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
            
        # Start database writer
        self.db_writer = AtomicDatabaseWriter(
            config=self.config,
            write_queue=self.db_write_queue,
            stop_event=self.output_stop,
            checkpoint_manager=self.checkpoint_manager
        )
        self.db_writer.start()
        
        # Start extraction monitor
        self.extraction_monitor = PDFExtractionMonitor(
            pdf_queue=self.pdf_queue,
            text_queue=self.text_queue,
            config=self.config,
            stop_event=self.extraction_stop,
            extraction_times=self.extraction_times
        )
        self.extraction_monitor.start()
        
        # Start processing threads
        self.output_thread = Thread(target=self._process_outputs, daemon=True)
        self.output_thread.start()
        
        self.stats_thread = Thread(target=self._track_stats, daemon=True)
        self.stats_thread.start()
        
        self.db_relay_thread = Thread(target=self._relay_db_items, daemon=True)
        self.db_relay_thread.start()
        
        logger.info("PDF pipeline V4 started successfully")
        
    def process_pdf_files(self, pdf_files: List[Path], resume: bool = True):
        """Process PDF files through the pipeline"""
        # Filter already processed if resuming
        if resume:
            filtered_files = []
            for pdf_path in pdf_files:
                arxiv_id = pdf_path.stem
                if not self.checkpoint_manager.is_pdf_processed(arxiv_id):
                    filtered_files.append(pdf_path)
                    self.completion_tracker.add_pdf(arxiv_id)
            logger.info(f"Resuming: {len(pdf_files) - len(filtered_files)} PDFs already processed")
            pdf_files = filtered_files
        else:
            # Add all PDFs to tracker
            for pdf_path in pdf_files:
                self.completion_tracker.add_pdf(pdf_path.stem)
                
        self.stats['total_pdfs'] = len(pdf_files)
        logger.info(f"Processing {len(pdf_files)} PDF files")
        
        # Queue PDFs
        for pdf_path in tqdm(pdf_files, desc="Queueing PDFs"):
            self.pdf_queue.put(pdf_path)
            
        # Send poison pills
        for _ in self.pdf_workers:
            self.pdf_queue.put(None)
            
    def wait_for_completion(self):
        """Wait for all processing to complete"""
        logger.info("Waiting for PDF extraction to complete...")
        
        # Wait for PDF workers
        for worker in self.pdf_workers:
            worker.join()
            
        logger.info("PDF extraction complete")
        
        # Send poison pills to chunking workers
        for _ in self.chunking_workers:
            self.text_queue.put(None)
            
        # Wait for chunking
        for worker in self.chunking_workers:
            worker.join()
            
        logger.info("Document chunking complete")
        
        # Wait for GPU processing
        logger.info("Waiting for GPU processing...")
        
        while not self.completion_tracker.is_complete():
            stats = self.completion_tracker.get_stats()
            logger.info(
                f"PDFs: {stats['pdfs']['completed']}/{stats['pdfs']['queued']} "
                f"(failed: {stats['pdfs']['failed']}), "
                f"Chunks: {stats['chunks']['completed']}/{stats['chunks']['expected']} "
                f"(in flight: {stats['chunks']['in_flight']})"
            )
            time.sleep(5.0)
            
        # Stop GPU workers
        for _ in self.gpu_workers:
            work = ChunkWork(batch_id=None, texts=[], metadata=[], chunks=[])
            self.gpu_queue.put(work, priority=0)
            
        for worker in self.gpu_workers:
            worker.join()
            
        # Stop other components
        self.output_stop.set()
        self.stop_event.set()
        
        # Wait for queues to empty
        while not self.output_queue.empty() or not self.db_write_queue.empty() or not self.db_queue.empty():
            time.sleep(1.0)
            
        # Stop database writer
        self.db_write_queue.put(None)
        if self.db_writer:
            self.db_writer.join()
            
        # Join threads
        self.output_thread.join()
        self.stats_thread.join()
        self.db_relay_thread.join()
        
        logger.info("PDF pipeline V4 completion successful")
        
    def _process_outputs(self):
        """Process GPU outputs and send to database"""
        while not self.output_stop.is_set():
            try:
                output = self.output_queue.get(timeout=1.0)
                if output is None:
                    break
                    
                # Send to database as chunks
                output['type'] = 'chunks'
                self.db_write_queue.put(output)
                
                # Update stats
                self.stats['total_chunks'] += output['doc_count']
                gpu_id = output['gpu_id']
                self.stats['gpu_stats'][gpu_id]['batches'] += 1
                self.stats['gpu_stats'][gpu_id]['time'] += output['gpu_time']
                
                # Log progress
                if self.stats['total_chunks'] % 1000 == 0:
                    self._log_progress()
                    
            except Empty:
                continue
                
    def _relay_db_items(self):
        """Relay items from db_queue to db_write_queue"""
        while not self.output_stop.is_set():
            try:
                item = self.db_queue.get(timeout=1.0)
                if item is None:
                    break
                self.db_write_queue.put(item)
            except Empty:
                continue
                
    def _track_stats(self):
        """Track processing statistics"""
        while not self.stop_event.is_set():
            try:
                stat = self.stats_queue.get(timeout=1.0)
                if stat is None:
                    break
                    
                event_type, gpu_id, batch_id, doc_count = stat
                
                if event_type == 'processing':
                    self.completion_tracker.start_chunk_batch(gpu_id, batch_id, doc_count)
                elif event_type == 'completed':
                    self.completion_tracker.complete_chunk_batch(gpu_id, batch_id, doc_count)
                elif event_type == 'failed':
                    self.completion_tracker.fail_chunk_batch(gpu_id, batch_id, doc_count)
                    
            except Empty:
                continue
                
    def _log_progress(self):
        """Log processing progress"""
        elapsed = time.time() - self.stats['start_time']
        chunks_per_sec = self.stats['total_chunks'] / elapsed if elapsed > 0 else 0
        
        stats = self.completion_tracker.get_stats()
        
        logger.info(
            f"Progress: PDFs {stats['pdfs']['completed']}/{stats['pdfs']['queued']}, "
            f"Chunks {stats['chunks']['completed']}/{stats['chunks']['expected']} "
            f"({chunks_per_sec:.1f} chunks/s)"
        )
        
    def shutdown(self):
        """Clean shutdown"""
        logger.info("Shutting down PDF pipeline")
        
        # Stop all events
        self.extraction_stop.set()
        self.chunking_stop.set()
        self.output_stop.set()
        self.stop_event.set()
        
        # Cleanup
        self.checkpoint_manager.cleanup()
        if self.manager:
            self.manager.shutdown()
            
        logger.info("PDF pipeline shutdown complete")


def main():
    """Main entry point for PDF processing"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process PDF documents through continuous GPU pipeline V4"
    )
    
    parser.add_argument(
        '--pdf-dir',
        type=str,
        default='/mnt/data/arxiv_data/pdf',
        help='Directory containing PDF files'
    )
    parser.add_argument('--db-name', type=str, default='irec_three_collections')
    parser.add_argument('--db-host', type=str, default='localhost')
    parser.add_argument('--gpu-devices', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--pdf-workers', type=int, default=4)
    parser.add_argument('--chunk-workers', type=int, default=4)
    parser.add_argument('--max-pdfs', type=int, help='Maximum number of PDFs to process')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--clean-start', action='store_true')
    parser.add_argument('--no-cpu-extraction', action='store_true', 
                       help='Disable forcing CPU for PDF extraction (allow GPU usage)')
    
    args = parser.parse_args()
    
    # Get password
    db_password = os.environ.get('ARANGO_PASSWORD', '')
    if not db_password:
        print("ERROR: ARANGO_PASSWORD environment variable not set!")
        sys.exit(1)
        
    # Create config
    config = PipelineConfig(
        gpu_devices=args.gpu_devices,
        batch_size=args.batch_size,
        pdf_extraction_workers=args.pdf_workers,
        preprocessing_workers=args.chunk_workers,
        db_host=args.db_host,
        db_name=args.db_name,
        db_password=db_password,
        force_cpu_extraction=not args.no_cpu_extraction  # Default is True (use CPU)
    )
    
    # Get PDF files
    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        print(f"ERROR: PDF directory not found: {pdf_dir}")
        sys.exit(1)
        
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if args.max_pdfs:
        pdf_files = pdf_files[:args.max_pdfs]
        
    print(f"\nFound {len(pdf_files)} PDF files")
    
    if not pdf_files:
        print("No PDF files found!")
        sys.exit(1)
        
    # Run pipeline
    try:
        pipeline = PDFProcessingPipeline(config)
        
        # Setup database
        if not args.resume:
            pipeline.setup_database(clean_start=args.clean_start)
            
        # Start pipeline
        pipeline.start()
        
        # Process PDFs
        pipeline.process_pdf_files(pdf_files, resume=args.resume)
        
        # Wait for completion
        pipeline.wait_for_completion()
        
        # Print final stats
        elapsed = time.time() - pipeline.stats['start_time']
        stats = pipeline.completion_tracker.get_stats()
        
        logger.info(
            f"\nCompleted in {elapsed/60:.1f} minutes:\n"
            f"  PDFs: {stats['pdfs']['completed']} processed, {stats['pdfs']['failed']} failed\n"
            f"  Chunks: {stats['chunks']['completed']} processed\n"
            f"  Rate: {stats['chunks']['completed']/elapsed:.1f} chunks/s"
        )
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
    finally:
        if 'pipeline' in locals():
            pipeline.shutdown()
            
    print("\n✅ PDF processing pipeline V4 completed!")


if __name__ == "__main__":
    main()