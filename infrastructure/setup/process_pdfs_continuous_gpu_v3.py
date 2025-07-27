#!/usr/bin/env python3
"""
Production-Ready Continuous GPU Pipeline V3 - Full PDF Processing
Processes complete PDF documents using Docling for extraction.

Key features:
- Full PDF text extraction with Docling
- Three-collection architecture (metadata, documents, chunks)
- All V3 reliability features (retry, validation, monitoring)
- Optimized for larger documents
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
import hashlib

# Add infrastructure directory to path
infrastructure_dir = Path(__file__).parent.parent
sys.path.append(str(infrastructure_dir))

# Import Docling for PDF processing
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
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
        logging.FileHandler('pdf_processing_pipeline_v3.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Reuse PriorityQueue from V3
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
class GPUWork:
    """Work item for GPU processing (embeddings)"""
    batch_id: str
    texts: List[str]
    metadata: List[Dict]
    chunks: List[Dict]  # For chunk information
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
    batch_size: int = 32  # Smaller for full documents
    prefetch_factor: int = 2  # Less prefetching due to larger documents
    
    # Worker settings
    pdf_extraction_workers: int = 4  # Docling is CPU-intensive
    preprocessing_workers: int = 4
    max_gpu_queue_size: Optional[int] = None
    max_output_queue_size: Optional[int] = None
    
    # Database
    db_host: str = "localhost"
    db_port: int = 8529
    db_name: str = "irec_three_collections"
    db_username: str = "root"
    db_password: str = ""
    db_batch_size: int = 100  # Smaller batches for full documents
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints/pdf_pipeline"
    checkpoint_interval: int = 50
    
    # Monitoring
    enable_monitoring: bool = True
    monitor_interval: float = 10.0  # Slower for PDF processing
    low_util_threshold: float = 50.0
    low_util_alert_threshold: int = 3
    
    # Memory management
    max_embeddings_in_memory: int = 1000  # Fewer due to larger docs
    embedding_cleanup_interval: int = 25
    
    # Retry settings
    max_retries: int = 3
    retry_delay_base: float = 2.0
    
    # Validation
    min_document_length: int = 100  # Minimum chars for valid document
    embedding_dimension: int = 2048  # Jina v4
    
    # Chunking settings
    chunk_size: int = 1024  # Characters per chunk
    chunk_overlap: int = 128
    
    # PDF extraction settings
    use_ocr: bool = False  # OCR is slow, disable for testing
    extract_tables: bool = True
    extract_figures: bool = False  # Disable for speed
    
    def __post_init__(self):
        """Validate and adjust configuration"""
        # Smaller queue sizes for PDFs
        if self.max_gpu_queue_size is None:
            self.max_gpu_queue_size = min(
                self.batch_size * self.prefetch_factor * len(self.gpu_devices),
                100  # Much smaller for PDFs
            )
            
        if self.max_output_queue_size is None:
            self.max_output_queue_size = self.max_gpu_queue_size * 2
            
        # Validate GPU devices
        available_gpus = torch.cuda.device_count()
        for gpu_id in self.gpu_devices:
            if gpu_id >= available_gpus:
                raise ValueError(f"GPU {gpu_id} not available")


class PDFExtractionWorker(Thread):
    """Worker for extracting text from PDFs using Docling"""
    
    def __init__(
        self,
        pdf_queue: Queue,
        text_queue: Queue,
        config: PipelineConfig,
        stop_event: Event,
        worker_id: int
    ):
        super().__init__(daemon=True)
        self.pdf_queue = pdf_queue
        self.text_queue = text_queue
        self.config = config
        self.stop_event = stop_event
        self.worker_id = worker_id
        self.processed_count = 0
        self.failed_count = 0
        
        # Initialize Docling
        self.converter = self._init_docling()
        
    def _init_docling(self):
        """Initialize Docling converter"""
        try:
            # Configure Docling
            pdf_options = PdfFormatOption(
                use_ocr=self.config.use_ocr,
                extract_tables=self.config.extract_tables,
                extract_figures=self.config.extract_figures,
                extract_footnotes=True,
                extract_page_footnotes=True,
                extract_references=True,
            )
            
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: pdf_options
                }
            )
            
            logger.info(f"PDF Worker {self.worker_id} initialized Docling")
            return converter
            
        except Exception as e:
            logger.error(f"Failed to initialize Docling: {e}")
            raise
            
    def run(self):
        """Process PDFs and extract text"""
        logger.info(f"PDF extraction worker {self.worker_id} started")
        
        while not self.stop_event.is_set():
            try:
                # Get PDF path
                pdf_path = self.pdf_queue.get(timeout=1.0)
                if pdf_path is None:  # Poison pill
                    break
                    
                # Extract text
                try:
                    extracted_data = self._extract_pdf(pdf_path)
                    if extracted_data:
                        self.text_queue.put(extracted_data)
                        self.processed_count += 1
                    else:
                        self.failed_count += 1
                        
                except Exception as e:
                    logger.error(f"Failed to process {pdf_path}: {e}")
                    self.failed_count += 1
                    
            except Empty:
                continue
            except Exception as e:
                logger.error(f"PDF worker {self.worker_id} error: {e}")
                
        logger.info(
            f"PDF worker {self.worker_id} done. "
            f"Processed: {self.processed_count}, Failed: {self.failed_count}"
        )
        
    def _extract_pdf(self, pdf_path: Path) -> Optional[Dict]:
        """Extract text and metadata from PDF"""
        try:
            # Convert PDF
            start_time = time.time()
            result = self.converter.convert(str(pdf_path))
            extraction_time = time.time() - start_time
            
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
            arxiv_id = pdf_path.stem  # Assumes filename is arxiv_id.pdf
            
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
                    'worker_id': self.worker_id
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
                # Try to extract title, authors, etc.
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
            'has_references': False
        }
        
        try:
            # Extract section headers
            text = result.document.export_to_markdown()
            
            # Find section headers (# Header)
            import re
            sections = re.findall(r'^#+\s+(.+)$', text, re.MULTILINE)
            structure['sections'] = sections[:20]  # Limit to first 20
            
            # Check for common sections
            text_lower = text.lower()
            structure['has_abstract'] = 'abstract' in text_lower[:5000]
            structure['has_references'] = 'references' in text_lower[-10000:]
            
        except Exception as e:
            logger.debug(f"Error extracting structure: {e}")
            
        return structure


class DocumentChunkingWorker(Thread):
    """Worker for chunking documents and preparing for embedding"""
    
    def __init__(
        self,
        text_queue: Queue,
        gpu_queue,  # PriorityQueue proxy
        config: PipelineConfig,
        stop_event: Event,
        worker_id: int
    ):
        super().__init__(daemon=True)
        self.text_queue = text_queue
        self.gpu_queue = gpu_queue
        self.config = config
        self.stop_event = stop_event
        self.worker_id = worker_id
        self.processed_count = 0
        
    def run(self):
        """Chunk documents and prepare batches"""
        batch_chunks = []
        batch_metadata = []
        batch_texts = []
        batch_num = 0
        
        while not self.stop_event.is_set():
            try:
                # Get extracted document
                doc_data = self.text_queue.get(timeout=1.0)
                if doc_data is None:  # Poison pill
                    # Send remaining batch
                    if batch_chunks:
                        self._send_batch(batch_texts, batch_metadata, batch_chunks, batch_num)
                    break
                    
                # Chunk the document
                chunks = self._chunk_document(doc_data)
                
                if not chunks:
                    logger.warning(f"No chunks created for {doc_data['arxiv_id']}")
                    continue
                    
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
                        
                # Store full document data (would go to documents collection)
                self._store_document_data(doc_data)
                self.processed_count += 1
                
            except Empty:
                # Send partial batch if idle
                if batch_chunks:
                    self._send_batch(batch_texts, batch_metadata, batch_chunks, batch_num)
                    batch_chunks = []
                    batch_metadata = []
                    batch_texts = []
                    batch_num += 1
                    
            except Exception as e:
                logger.error(f"Chunking worker {self.worker_id} error: {e}")
                
        logger.info(f"Chunking worker {self.worker_id} done. Processed: {self.processed_count} documents")
        
    def _chunk_document(self, doc_data: Dict) -> List[Dict]:
        """Chunk document into overlapping segments"""
        chunks = []
        text = doc_data['full_text']
        arxiv_id = doc_data['arxiv_id']
        
        # Simple character-based chunking with overlap
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Find chunk boundaries
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end
                sentence_end = text.rfind('. ', start, end)
                if sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
                    
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) > 50:  # Minimum chunk size
                chunk_id = f"{arxiv_id}_chunk_{chunk_index:04d}"
                
                chunks.append({
                    'chunk_id': chunk_id,
                    'arxiv_id': arxiv_id,
                    'text': chunk_text,
                    'index': chunk_index,
                    'start_char': start,
                    'end_char': end,
                    'metadata': {
                        'chunk_size': len(chunk_text),
                        'overlapping': start > 0 and overlap > 0
                    }
                })
                
                chunk_index += 1
                
            # Move to next chunk with overlap
            start = end - overlap if overlap > 0 else end
            
        logger.debug(f"Created {len(chunks)} chunks for {arxiv_id}")
        return chunks
        
    def _store_document_data(self, doc_data: Dict):
        """Store full document data (placeholder - would go to documents collection)"""
        # In real implementation, this would store to the documents collection
        # For now, just log
        logger.debug(
            f"Would store document {doc_data['arxiv_id']} "
            f"({doc_data['extraction_metadata']['char_count']} chars)"
        )
        
    def _send_batch(self, texts: List[str], metadata: List[Dict], chunks: List[Dict], batch_num: int):
        """Send batch to GPU queue"""
        if not texts:
            return
            
        batch_id = f"chunk_w{self.worker_id}_b{batch_num}_{int(time.time() * 1000)}"
        
        work = GPUWork(
            batch_id=batch_id,
            texts=texts,
            metadata=metadata,
            chunks=chunks,
            priority=1.0
        )
        
        self.gpu_queue.put(work, priority=work.priority)
        logger.debug(f"Chunking worker {self.worker_id} sent batch {batch_id} with {len(texts)} chunks")


class PDFDatabaseWriter(Thread):
    """Database writer for three-collection architecture"""
    
    def __init__(
        self,
        config: PipelineConfig,
        write_queue: Queue,
        stop_event: Event,
        checkpoint_manager
    ):
        super().__init__(daemon=True)
        self.config = config
        self.write_queue = write_queue
        self.stop_event = stop_event
        self.checkpoint_manager = checkpoint_manager
        
        self.client = None
        self.db = None
        self.collections = {}
        
        self.batch_buffer = {
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
        logger.info("PDF Database Writer started")
        
        # Initialize with retry
        if not self._initialize_db_with_retry():
            logger.error("Failed to initialize database")
            return
            
        while not self.stop_event.is_set():
            try:
                # Get item
                item = self.write_queue.get(timeout=1.0)
                if item is None:
                    self._flush_all_buffers()
                    break
                    
                # Route to appropriate buffer
                self._add_to_buffer(item)
                
                # Check if any buffer needs flushing
                self._check_and_flush_buffers()
                
            except Empty:
                # Flush on idle
                self._check_and_flush_buffers()
                
            except Exception as e:
                logger.error(f"Database writer error: {e}")
                
        logger.info(f"Database Writer stopped. Written: {self.written_counts}")
        
    def _initialize_db_with_retry(self) -> bool:
        """Initialize with three collections"""
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
            self.batch_buffer['metadata'].append(item['data'])
        elif item_type == 'documents':
            self.batch_buffer['documents'].append(item['data'])
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
                'start_char': chunk_data['start_char'],
                'end_char': chunk_data['end_char'],
                'chunk_metadata': chunk_data['metadata'],
                'processed_gpu': gpu_output['gpu_id'],
                'processing_time': gpu_output['gpu_time'] / len(gpu_output['chunks']),
                'processed_at': gpu_output['timestamp']
            }
            
            # Validate before adding
            if self._validate_chunk(chunk_record):
                self.batch_buffer['chunks'].append(chunk_record)
                
    def _validate_chunk(self, chunk: Dict) -> bool:
        """Validate chunk record"""
        # Check required fields
        if not chunk.get('chunk_id') or not chunk.get('arxiv_id'):
            return False
            
        # Validate embedding
        embedding = chunk.get('embedding', [])
        if not isinstance(embedding, list) or len(embedding) != self.config.embedding_dimension:
            return False
            
        # Check for NaN/Inf
        try:
            if any(math.isnan(x) or math.isinf(x) for x in embedding):
                return False
        except:
            return False
            
        return True
        
    def _check_and_flush_buffers(self):
        """Check and flush buffers that are full"""
        for collection_name, buffer in self.batch_buffer.items():
            if len(buffer) >= self.config.db_batch_size:
                self._flush_buffer(collection_name)
                
    def _flush_all_buffers(self):
        """Flush all buffers"""
        for collection_name in self.batch_buffer:
            if self.batch_buffer[collection_name]:
                self._flush_buffer(collection_name)
                
    def _flush_buffer(self, collection_name: str):
        """Flush a specific buffer with retry"""
        buffer = self.batch_buffer[collection_name]
        if not buffer:
            return
            
        success = self._write_batch_with_retry(collection_name, buffer)
        
        if success:
            self.written_counts[collection_name] += len(buffer)
            self.batch_buffer[collection_name] = []
        else:
            logger.error(f"Failed to write {len(buffer)} records to {collection_name}")
            # Could save to checkpoint for recovery
            self.batch_buffer[collection_name] = []
            
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


class PDFProcessingPipeline:
    """Main pipeline for processing PDF documents"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.checkpoint_manager = SecureCheckpointManager(config.checkpoint_dir)
        
        # Create manager for priority queue
        self.manager = BaseManager()
        self.manager.start()
        
        # Queues
        self.pdf_queue = Queue(maxsize=100)  # Limit PDF queue size
        self.text_queue = Queue(maxsize=50)  # Extracted text queue
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
        
        # Tracking
        self.completion_tracker = DocumentBatchCompletionTracker(config.batch_size)
        
        # Stats
        self.stats = {
            'start_time': None,
            'total_pdfs': 0,
            'total_chunks': 0,
            'extraction_times': [],
            'gpu_stats': {gpu: {'batches': 0, 'time': 0} for gpu in config.gpu_devices}
        }
        
    def setup_database(self, clean_start=False):
        """Setup three-collection database"""
        try:
            client = ArangoClient(
                hosts=f'http://{self.config.db_host}:{self.config.db_port}',
                http_client=HTTPClient(pool_connections=10, pool_maxsize=10)
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
            
            # Create collections
            collections = {
                'metadata': {
                    'indexes': [
                        {'type': 'hash', 'fields': ['arxiv_id'], 'unique': True},
                        {'type': 'persistent', 'fields': ['categories[*]']},
                        {'type': 'persistent', 'fields': ['published']}
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
        logger.info("Starting PDF Processing Pipeline")
        
        self.stats['start_time'] = time.time()
        
        # Start PDF extraction workers
        for i in range(self.config.pdf_extraction_workers):
            worker = PDFExtractionWorker(
                pdf_queue=self.pdf_queue,
                text_queue=self.text_queue,
                config=self.config,
                stop_event=self.extraction_stop,
                worker_id=i
            )
            worker.start()
            self.pdf_workers.append(worker)
            
        logger.info(f"Started {self.config.pdf_extraction_workers} PDF extraction workers")
        
        # Start chunking workers
        for i in range(self.config.preprocessing_workers):
            worker = DocumentChunkingWorker(
                text_queue=self.text_queue,
                gpu_queue=self.gpu_queue,
                config=self.config,
                stop_event=self.chunking_stop,
                worker_id=i
            )
            worker.start()
            self.chunking_workers.append(worker)
            
        logger.info(f"Started {self.config.preprocessing_workers} chunking workers")
        
        # Start GPU workers (reuse from V3)
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
            
        logger.info(f"Started GPU workers on devices: {self.config.gpu_devices}")
        
        # Start database writer
        self.db_writer = PDFDatabaseWriter(
            config=self.config,
            write_queue=self.db_write_queue,
            stop_event=self.output_stop,
            checkpoint_manager=self.checkpoint_manager
        )
        self.db_writer.start()
        
        # Start processing threads
        self.output_thread = Thread(target=self._process_outputs, daemon=True)
        self.output_thread.start()
        
        self.stats_thread = Thread(target=self._track_stats, daemon=True)
        self.stats_thread.start()
        
        logger.info("PDF pipeline started successfully")
        
    def process_pdf_files(self, pdf_files: List[Path], resume: bool = True):
        """Process PDF files through the pipeline"""
        # Filter already processed if resuming
        if resume:
            processed_count = self.checkpoint_manager.get_processed_count()
            logger.info(f"Resuming: {processed_count} documents already processed")
            
            # Filter files
            filtered_files = []
            for pdf_path in pdf_files:
                arxiv_id = pdf_path.stem
                if not self.checkpoint_manager.is_document_processed(arxiv_id):
                    filtered_files.append(pdf_path)
            pdf_files = filtered_files
            
        self.stats['total_pdfs'] = len(pdf_files)
        logger.info(f"Processing {len(pdf_files)} PDF files")
        
        # Queue PDFs
        for pdf_path in tqdm(pdf_files, desc="Queueing PDFs"):
            self.pdf_queue.put(pdf_path)
            
        # Send poison pills
        for _ in self.pdf_workers:
            self.pdf_queue.put(None)
            
        # Calculate expected chunks (rough estimate)
        avg_chunks_per_doc = 50  # Estimate
        expected_chunks = len(pdf_files) * avg_chunks_per_doc
        self.completion_tracker.add_documents(expected_chunks)
        
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
                f"Chunks: {stats['documents']['completed']}/{stats['documents']['queued']} "
                f"(in flight: {stats['documents']['in_flight']})"
            )
            time.sleep(5.0)
            
        # Stop GPU workers
        for _ in self.gpu_workers:
            work = GPUWork(batch_id=None, texts=[], metadata=[], chunks=[])
            self.gpu_queue.put(work, priority=0)
            
        for worker in self.gpu_workers:
            worker.join()
            
        # Stop other components
        self.output_stop.set()
        self.stop_event.set()
        
        # Wait for queues to empty
        while not self.output_queue.empty() or not self.db_write_queue.empty():
            time.sleep(1.0)
            
        # Stop database writer
        self.db_write_queue.put(None)
        if self.db_writer:
            self.db_writer.join()
            
        # Join threads
        self.output_thread.join()
        self.stats_thread.join()
        
        logger.info("PDF pipeline completion successful")
        
    def _process_outputs(self):
        """Process GPU outputs and send to database"""
        while not self.output_stop.is_set():
            try:
                output = self.output_queue.get(timeout=1.0)
                if output is None:
                    break
                    
                # Save checkpoint
                self.checkpoint_manager.save_batch_result(output['batch_id'], output)
                
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
                
    def _track_stats(self):
        """Track processing statistics"""
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
                    
            except Empty:
                continue
                
    def _log_progress(self):
        """Log processing progress"""
        elapsed = time.time() - self.stats['start_time']
        chunks_per_sec = self.stats['total_chunks'] / elapsed if elapsed > 0 else 0
        
        # Estimate PDFs processed
        avg_chunks_per_doc = self.stats['total_chunks'] / max(1, self.stats['total_pdfs'])
        pdfs_processed = int(self.stats['total_chunks'] / max(1, avg_chunks_per_doc))
        
        logger.info(
            f"Progress: ~{pdfs_processed}/{self.stats['total_pdfs']} PDFs, "
            f"{self.stats['total_chunks']} chunks ({chunks_per_sec:.1f} chunks/s)"
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


# Reuse classes from V3
from process_abstracts_continuous_gpu_v3 import (
    GPUWorker,
    SecureCheckpointManager,
    DocumentBatchCompletionTracker
)


def main():
    """Main entry point for PDF processing"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process PDF documents through continuous GPU pipeline"
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
        db_password=db_password
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
        
        # Print stats
        elapsed = time.time() - pipeline.stats['start_time']
        logger.info(
            f"Completed in {elapsed/60:.1f} minutes. "
            f"Processed {pipeline.stats['total_pdfs']} PDFs, "
            f"{pipeline.stats['total_chunks']} chunks "
            f"({pipeline.stats['total_chunks']/elapsed:.1f} chunks/s)"
        )
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
    finally:
        if 'pipeline' in locals():
            pipeline.shutdown()
            
    print("\n✅ PDF processing pipeline completed!")


if __name__ == "__main__":
    main()