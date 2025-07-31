#!/usr/bin/env python3
"""
Dual GPU PDF Processing Pipeline
Phase 2: PDF to searchable content using Docling and Jina
GPU 0: Docling PDF → Markdown conversion  
GPU 1: Jina embedding of markdown chunks
Target: 2 PDFs/second (limited by Docling)
"""

import os
import sys
import json
import time
import queue
import logging
import warnings
import threading
import multiprocessing as mp
import tarfile
import tempfile
import shutil
import fcntl
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set

import numpy as np
from tqdm import tqdm
from arango import ArangoClient
from arango.exceptions import DocumentUpdateError

# Import torch and transformers after setting environment
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import torch
from transformers import AutoTokenizer, AutoModel

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_processing_dual_gpu.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PDFConfig:
    """Configuration for PDF processing pipeline"""
    # Data sources
    pdf_tar_dir: str = "/mnt/data-cold/arxiv_data"
    max_pdfs: Optional[int] = None
    
    # Database
    db_name: str = "arxiv_single_collection"
    collection_name: str = "arxiv_documents"
    db_host: str = "192.168.1.69"
    db_port: int = 8529
    db_username: str = "root"
    db_password: str = os.environ.get('ARANGO_PASSWORD', '')
    
    # Processing
    batch_size: int = 50  # PDFs per batch
    extraction_workers: int = 2
    docling_workers: int = 1  # GPU 0
    embedding_workers: int = 1  # GPU 1
    update_workers: int = 2
    
    # Queues
    extraction_queue_size: int = 3
    docling_queue_size: int = 2  # GPU memory limited
    chunk_queue_size: int = 10
    embedding_queue_size: int = 7  # Proven in Phase 1
    update_queue_size: int = 5
    
    # PDF Processing
    working_dir: str = "/tmp/arxiv_pdf_processing"
    chunk_size: int = 2048
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    
    # GPUs
    docling_gpu: int = 0
    embedding_gpu: int = 1
    
    # Retry settings
    max_retries: int = 3
    extraction_timeout: int = 60
    conversion_timeout: int = 300
    embedding_timeout: int = 120
    
    def __post_init__(self):
        """Validate configuration"""
        # Database configuration validation
        if not self.db_password:
            raise ValueError("ARANGO_PASSWORD environment variable not set")
        if not self.db_host:
            raise ValueError("db_host cannot be empty")
        if not self.db_user:
            raise ValueError("db_user cannot be empty")
        if not self.db_name:
            raise ValueError("db_name cannot be empty")
        if not self.collection_name:
            raise ValueError("collection_name cannot be empty")
            
        # Path validation
        if not self.pdf_tar_dir:
            raise ValueError("pdf_tar_dir cannot be empty")
        if not self.working_dir:
            raise ValueError("working_dir cannot be empty")
            
        # Numerical parameter validation
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.chunk_size <= self.chunk_overlap:
            raise ValueError("chunk_size must be larger than chunk_overlap")
        if self.min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if self.extraction_timeout <= 0:
            raise ValueError("extraction_timeout must be positive")
        if self.conversion_timeout <= 0:
            raise ValueError("conversion_timeout must be positive")
        if self.embedding_timeout <= 0:
            raise ValueError("embedding_timeout must be positive")
            
        # Worker configuration validation
        if self.extraction_workers <= 0:
            raise ValueError("extraction_workers must be positive")
        if self.docling_workers <= 0:
            raise ValueError("docling_workers must be positive")
        if self.embedding_workers <= 0:
            raise ValueError("embedding_workers must be positive")
        if self.update_workers <= 0:
            raise ValueError("update_workers must be positive")
            
        # GPU validation
        if self.docling_gpu < 0:
            raise ValueError("docling_gpu must be non-negative")
        if self.embedding_gpu < 0:
            raise ValueError("embedding_gpu must be non-negative")

class PDFPipelineMetrics:
    """Metrics tracking for PDF pipeline"""
    
    def __init__(self):
        self.start_time = time.time()
        self.pdfs_extracted = 0
        self.pdfs_converted = 0
        self.chunks_created = 0
        self.chunks_embedded = 0
        self.documents_updated = 0
        self.extraction_errors = {}
        self.conversion_errors = {}
        self.gpu0_utilization = []
        self.gpu1_utilization = []
        self._lock = threading.Lock()
        
    def record_extraction(self, success: bool, tar_file: str = None, error: str = None):
        with self._lock:
            if success:
                self.pdfs_extracted += 1
            elif tar_file and error:
                if tar_file not in self.extraction_errors:
                    self.extraction_errors[tar_file] = []
                self.extraction_errors[tar_file].append(error)
                
    def record_conversion(self, success: bool, pdf_id: str = None, error: str = None):
        with self._lock:
            if success:
                self.pdfs_converted += 1
            elif pdf_id and error:
                self.conversion_errors[pdf_id] = error
                
    def record_chunks(self, count: int):
        with self._lock:
            self.chunks_created += count
            
    def record_embeddings(self, count: int):
        with self._lock:
            self.chunks_embedded += count
            
    def record_update(self, success: bool):
        with self._lock:
            if success:
                self.documents_updated += 1
                
    def get_stats(self):
        with self._lock:
            elapsed = time.time() - self.start_time
            return {
                'elapsed_seconds': elapsed,
                'pdfs_extracted': self.pdfs_extracted,
                'pdfs_converted': self.pdfs_converted,
                'chunks_created': self.chunks_created,
                'chunks_embedded': self.chunks_embedded,
                'documents_updated': self.documents_updated,
                'extraction_rate': self.pdfs_extracted / elapsed if elapsed > 0 else 0,
                'conversion_rate': self.pdfs_converted / elapsed if elapsed > 0 else 0,
                'update_rate': self.documents_updated / elapsed if elapsed > 0 else 0,
                'extraction_errors': len(self.extraction_errors),
                'conversion_errors': len(self.conversion_errors)
            }

class PDFExtractor:
    """Extract PDFs from tar archives"""
    
    def __init__(self, working_dir: str):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_pdfs_from_tar(self, tar_path: Path, pdf_ids: List[str]) -> Dict[str, Path]:
        """Extract specific PDFs from tar archive with file locking"""
        extracted = {}
        temp_dir = self.working_dir / f"extract_{int(time.time() * 1000)}"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Open tar file with file locking
            with open(tar_path, 'rb') as tar_file:
                # Acquire shared lock to prevent concurrent access issues
                fcntl.flock(tar_file.fileno(), fcntl.LOCK_SH)
                try:
                    with tarfile.open(fileobj=tar_file, mode='r') as tar:
                        # Get all members
                        members = {m.name: m for m in tar.getmembers()}
                
                        # Extract requested PDFs
                        for pdf_id in pdf_ids:
                            # Try different naming patterns
                            patterns = [
                                f"{pdf_id}.pdf",
                                f"{pdf_id.replace('.', '')}.pdf",
                                f"pdf/{pdf_id}.pdf"
                            ]
                            
                            for pattern in patterns:
                                if pattern in members:
                                    member = members[pattern]
                                    tar.extract(member, path=temp_dir)
                                    extracted_path = temp_dir / member.name
                                    if extracted_path.exists():
                                        extracted[pdf_id] = extracted_path
                                        break
                finally:
                    # Release the file lock
                    fcntl.flock(tar_file.fileno(), fcntl.LOCK_UN)
                                
            return extracted
            
        except Exception as e:
            logger.error(f"Error extracting from {tar_path}: {e}")
            # Clean up on error
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise
            
    def cleanup_extracted(self, paths: List[Path]):
        """Clean up extracted PDF files"""
        for path in paths:
            try:
                if path.exists():
                    path.unlink()
                # Also try to clean up parent directories if empty
                parent = path.parent
                if parent.exists() and not any(parent.iterdir()):
                    parent.rmdir()
            except Exception as e:
                logger.warning(f"Failed to cleanup {path}: {e}")

class DoclingConverter:
    """Convert PDFs to markdown using Docling on GPU 0"""
    
    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self._initialize_docling()
        
    def _initialize_docling(self):
        """Initialize Docling with GPU support"""
        self.docling_available = False
        try:
            # Import Docling (assuming it's installed)
            from docling import DocumentConverter
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            
            # Configure for GPU usage
            pipeline_options = PdfPipelineOptions(
                do_ocr=False,  # Disable OCR for speed
                do_table_structure=True,
                table_structure_options={
                    "device": f"cuda:{self.gpu_id}"
                }
            )
            
            self.converter = DocumentConverter(
                pipeline_options=pipeline_options,
                pdf_backend="pypdfium2"  # Faster than pdfplumber
            )
            
            self.docling_available = True
            logger.info(f"Docling initialized on GPU {self.gpu_id}")
            
        except ImportError as e:
            logger.error("Docling not installed. Please install with: pip install docling")
            logger.error(f"Import error details: {e}")
            # Set flag to prevent usage
            self.docling_available = False
            raise RuntimeError("Docling is required but not installed. Cannot proceed with PDF processing.")
            
    def convert_pdf(self, pdf_path: Path) -> Optional[str]:
        """Convert PDF to markdown"""
        if not self.docling_available:
            raise RuntimeError("Docling is not available. Cannot convert PDF.")
            
        try:
            # Use device context instead of global setting
            with torch.cuda.device(self.gpu_id):
                # Convert PDF
                result = self.converter.convert(str(pdf_path))
            
            # Extract markdown
            if result and hasattr(result, 'render_as_markdown'):
                markdown = result.render_as_markdown()
                return markdown
            else:
                logger.warning(f"No markdown output for {pdf_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error converting {pdf_path}: {e}")
            return None
        finally:
            # Clear GPU memory
            torch.cuda.empty_cache()

class TextChunker:
    """Chunk markdown text semantically"""
    
    def __init__(self, chunk_size: int = 2048, overlap: int = 200, min_size: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_size = min_size
        
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Chunk text while preserving semantic boundaries"""
        if not text or len(text) < self.min_size:
            return []
            
        chunks = []
        lines = text.split('\n')
        current_chunk = []
        current_size = 0
        current_section = "Unknown"
        
        for line in lines:
            # Detect section headers (common markdown patterns)
            if line.startswith('#'):
                # New section
                if current_chunk and current_size >= self.min_size:
                    # Save current chunk
                    chunk_text = '\n'.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'metadata': {
                            'section': current_section,
                            'char_start': len(text) - len(chunk_text),
                            'char_end': len(text)
                        }
                    })
                    
                    # Start new chunk with overlap
                    if self.overlap > 0 and len(current_chunk) > 5:
                        overlap_lines = current_chunk[-5:]
                        current_chunk = overlap_lines
                        current_size = sum(len(line) for line in overlap_lines)
                    else:
                        current_chunk = []
                        current_size = 0
                        
                current_section = line.strip('#').strip()
                
            current_chunk.append(line)
            current_size += len(line)
            
            # Check if chunk is full
            if current_size >= self.chunk_size:
                chunk_text = '\n'.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        'section': current_section,
                        'char_start': len(text) - len(chunk_text),
                        'char_end': len(text)
                    }
                })
                
                # Start new chunk with overlap
                if self.overlap > 0 and len(current_chunk) > 5:
                    overlap_lines = current_chunk[-5:]
                    current_chunk = overlap_lines
                    current_size = sum(len(line) for line in overlap_lines)
                else:
                    current_chunk = []
                    current_size = 0
                    
        # Add final chunk
        if current_chunk and current_size >= self.min_size:
            chunk_text = '\n'.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'metadata': {
                    'section': current_section,
                    'char_start': len(text) - len(chunk_text),
                    'char_end': len(text)
                }
            })
            
        # Add chunk IDs
        for i, chunk in enumerate(chunks):
            chunk['chunk_id'] = i
            
        return chunks

class JinaEmbedder:
    """Jina embeddings on GPU 1 (reused from Phase 1)"""
    
    def __init__(self, device='cuda:1'):
        self.device = device
        self.model_name = "jinaai/jina-embeddings-v3"
        
        logger.info(f"Loading Jina model on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_name, 
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info("Jina model loaded successfully")
        
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts"""
        try:
            inputs = self.tokenizer(
                texts, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=8192
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings_np = embeddings.cpu().numpy()
            
            # Clean up GPU memory
            del outputs, embeddings, inputs
            torch.cuda.empty_cache()
            
            return embeddings_np
            
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.error(f"GPU OOM error on batch of {len(texts)} chunks")
            
            # Process in smaller batches
            if len(texts) > 1:
                mid = len(texts) // 2
                emb1 = self.embed_batch(texts[:mid])
                emb2 = self.embed_batch(texts[mid:])
                return np.vstack([emb1, emb2])
            else:
                raise

def extraction_worker(
    worker_id: int,
    tar_queue: mp.Queue,
    extraction_queue: mp.Queue,
    metrics: PDFPipelineMetrics,
    config: PDFConfig
):
    """Worker process for extracting PDFs from tar files"""
    logger = logging.getLogger(f"ExtractionWorker-{worker_id}")
    extractor = PDFExtractor(config.working_dir)
    
    while True:
        try:
            item = tar_queue.get(timeout=1.0)
            if item is None:  # Poison pill
                break
                
            tar_path, pdf_batch = item
            
            try:
                # Extract PDFs
                extracted = extractor.extract_pdfs_from_tar(Path(tar_path), pdf_batch)
                
                # Send extracted PDFs to next stage
                for pdf_id, pdf_path in extracted.items():
                    extraction_queue.put((pdf_id, pdf_path))
                    metrics.record_extraction(True)
                    
                # Record failed extractions
                failed = set(pdf_batch) - set(extracted.keys())
                for pdf_id in failed:
                    metrics.record_extraction(False, tar_path, f"PDF {pdf_id} not found")
                    
            except Exception as e:
                logger.error(f"Extraction error for {tar_path}: {e}")
                for pdf_id in pdf_batch:
                    metrics.record_extraction(False, tar_path, str(e))
                    
        except queue.Empty:
            continue

def docling_worker(
    gpu_id: int,
    extraction_queue: mp.Queue,
    chunk_queue: mp.Queue,
    stop_event: mp.Event,
    metrics: PDFPipelineMetrics,
    config: PDFConfig
):
    """GPU 0 worker for PDF to markdown conversion"""
    # Set device in local context instead of globally
    device = torch.device(f'cuda:{gpu_id}')
    logger = logging.getLogger(f"DoclingWorker-GPU{gpu_id}")
    
    converter = DoclingConverter(gpu_id)
    chunker = TextChunker(config.chunk_size, config.chunk_overlap, config.min_chunk_size)
    extractor = PDFExtractor(config.working_dir)
    
    while not stop_event.is_set():
        try:
            item = extraction_queue.get(timeout=1.0)
            if item is None:
                break
                
            pdf_id, pdf_path = item
            
            try:
                # Convert PDF to markdown
                markdown = converter.convert_pdf(pdf_path)
                
                if markdown:
                    # Chunk the markdown
                    chunks = chunker.chunk_text(markdown)
                    
                    if chunks:
                        # Send to embedding queue
                        chunk_queue.put({
                            'pdf_id': pdf_id,
                            'markdown': markdown,
                            'chunks': chunks
                        })
                        
                        metrics.record_conversion(True)
                        metrics.record_chunks(len(chunks))
                    else:
                        logger.warning(f"No chunks generated for {pdf_id}")
                        metrics.record_conversion(False, pdf_id, "No chunks generated")
                else:
                    metrics.record_conversion(False, pdf_id, "No markdown generated")
                    
            except Exception as e:
                logger.error(f"Conversion error for {pdf_id}: {e}")
                metrics.record_conversion(False, pdf_id, str(e))
                
            finally:
                # Clean up the PDF file
                extractor.cleanup_extracted([pdf_path])
                
        except queue.Empty:
            continue

def embedding_worker(
    gpu_id: int,
    chunk_queue: mp.Queue,
    update_queue: mp.Queue,
    stop_event: mp.Event,
    metrics: PDFPipelineMetrics,
    config: PDFConfig
):
    """GPU 1 worker for chunk embeddings"""
    # Set device in local context instead of globally
    device = torch.device(f'cuda:{gpu_id}')
    logger = logging.getLogger(f"EmbeddingWorker-GPU{gpu_id}")
    
    embedder = JinaEmbedder(device=f'cuda:{gpu_id}')
    
    while not stop_event.is_set():
        try:
            item = chunk_queue.get(timeout=1.0)
            if item is None:
                break
                
            pdf_id = item['pdf_id']
            markdown = item['markdown']
            chunks = item['chunks']
            
            try:
                # Extract texts for batch embedding
                texts = [chunk['text'] for chunk in chunks]
                
                # Generate embeddings
                embeddings = embedder.embed_batch(texts)
                
                # Add embeddings to chunks
                for i, chunk in enumerate(chunks):
                    chunk['embedding'] = embeddings[i].tolist()
                    
                # Send to update queue
                update_queue.put({
                    'pdf_id': pdf_id,
                    'pdf_content': {
                        'markdown': markdown,
                        'chunks': chunks,
                        'extraction_metadata': {
                            'docling_version': '1.0',  # Update with actual version
                            'extraction_time': datetime.utcnow().isoformat() + 'Z',
                            'chunk_count': len(chunks),
                            'chunking_strategy': f'semantic_{config.chunk_size}'
                        }
                    }
                })
                
                metrics.record_embeddings(len(chunks))
                
            except Exception as e:
                logger.error(f"Embedding error for {pdf_id}: {e}")
                
        except queue.Empty:
            continue

class DocumentUpdater(threading.Thread):
    """Update existing documents with PDF content"""
    
    def __init__(self, update_queue: queue.Queue, config: PDFConfig, metrics: PDFPipelineMetrics):
        super().__init__()
        self.update_queue = update_queue
        self.config = config
        self.metrics = metrics
        self.stop_event = threading.Event()
        self._initialize_db()
        
    def _initialize_db(self):
        """Initialize database connection"""
        client = ArangoClient(hosts=f'http://{self.config.db_host}:{self.config.db_port}')
        self.db = client.db(
            self.config.db_name,
            username=self.config.db_username,
            password=self.config.db_password
        )
        self.collection = self.db.collection(self.config.collection_name)
        
    def run(self):
        """Main update loop"""
        while not self.stop_event.is_set() or not self.update_queue.empty():
            try:
                item = self.update_queue.get(timeout=1.0)
                pdf_id = item['pdf_id']
                pdf_content = item['pdf_content']
                
                # Update document with retry logic
                max_retries = 3
                retry_delay = 1.0
                
                for attempt in range(max_retries):
                    try:
                        # Update document
                        self.collection.update({
                            '_key': pdf_id,
                            'pdf_content': pdf_content,
                            'pdf_status': {
                                'state': 'completed',
                                'last_updated': datetime.utcnow().isoformat() + 'Z',
                                'processing_time_seconds': time.time() - self.metrics.start_time
                            }
                        })
                        
                        self.metrics.record_update(True)
                        break  # Success, exit retry loop
                        
                    except DocumentUpdateError as e:
                        if attempt < max_retries - 1:
                            logger.warning(f"Update attempt {attempt + 1} failed for {pdf_id}: {e}. Retrying...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                        else:
                            logger.error(f"Failed to update {pdf_id} after {max_retries} attempts: {e}")
                            self.metrics.record_update(False)
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Update error: {e}")
                
    def stop(self):
        self.stop_event.set()

class PDFProcessingPipeline:
    """Main orchestrator for PDF processing"""
    
    def __init__(self, config: PDFConfig):
        self.config = config
        self.metrics = PDFPipelineMetrics()
        
    def get_unprocessed_pdfs(self) -> List[Tuple[str, str]]:
        """Get list of unprocessed PDFs from database"""
        client = ArangoClient(hosts=f'http://{self.config.db_host}:{self.config.db_port}')
        db = client.db(
            self.config.db_name,
            username=self.config.db_username,
            password=self.config.db_password
        )
        
        # Query for unprocessed PDFs
        query = """
        FOR doc IN @@collection
            FILTER doc.pdf_status.state == "unprocessed"
            LIMIT @limit
            RETURN {
                pdf_id: doc._key,
                tar_source: doc.pdf_status.tar_source
            }
        """
        
        cursor = db.aql.execute(
            query,
            bind_vars={
                '@collection': self.config.collection_name,
                'limit': self.config.max_pdfs or 1000000
            }
        )
        
        return [(doc['pdf_id'], doc['tar_source']) for doc in cursor if doc['tar_source']]
        
    def group_by_tar(self, pdf_list: List[Tuple[str, str]]) -> Dict[str, List[str]]:
        """Group PDFs by their tar source"""
        tar_groups = {}
        for pdf_id, tar_source in pdf_list:
            if tar_source not in tar_groups:
                tar_groups[tar_source] = []
            tar_groups[tar_source].append(pdf_id)
        return tar_groups
        
    def run(self):
        """Run the PDF processing pipeline"""
        logger.info("Starting Dual GPU PDF Processing Pipeline")
        logger.info(f"Target: 2 PDFs/second")
        logger.info(f"Configuration: {self.config}")
        
        # Get unprocessed PDFs
        logger.info("Querying for unprocessed PDFs...")
        pdf_list = self.get_unprocessed_pdfs()
        logger.info(f"Found {len(pdf_list)} unprocessed PDFs")
        
        if not pdf_list:
            logger.info("No unprocessed PDFs found")
            return
            
        # Group by tar file
        tar_groups = self.group_by_tar(pdf_list)
        logger.info(f"PDFs grouped into {len(tar_groups)} tar files")
        
        # Create queues
        tar_queue = mp.Queue(maxsize=self.config.extraction_queue_size)
        extraction_queue = mp.Queue(maxsize=self.config.docling_queue_size)
        chunk_queue = mp.Queue(maxsize=self.config.chunk_queue_size)
        update_queue = mp.Queue(maxsize=self.config.update_queue_size)
        
        # Start update thread
        updater = DocumentUpdater(update_queue, self.config, self.metrics)
        updater.start()
        
        # Start worker processes
        processes = []
        stop_event = mp.Event()
        
        # Extraction workers
        for i in range(self.config.extraction_workers):
            p = mp.Process(
                target=extraction_worker,
                args=(i, tar_queue, extraction_queue, self.metrics, self.config)
            )
            p.start()
            processes.append(p)
            
        # Docling worker (GPU 0)
        p = mp.Process(
            target=docling_worker,
            args=(self.config.docling_gpu, extraction_queue, chunk_queue, stop_event, self.metrics, self.config)
        )
        p.start()
        processes.append(p)
        
        # Embedding worker (GPU 1)
        p = mp.Process(
            target=embedding_worker,
            args=(self.config.embedding_gpu, chunk_queue, update_queue, stop_event, self.metrics, self.config)
        )
        p.start()
        processes.append(p)
        
        # Feed tar queue
        def feed_tar_queue():
            for tar_path, pdf_ids in tar_groups.items():
                # Process in batches
                for i in range(0, len(pdf_ids), self.config.batch_size):
                    batch = pdf_ids[i:i+self.config.batch_size]
                    full_tar_path = Path(self.config.pdf_tar_dir) / tar_path
                    if full_tar_path.exists():
                        tar_queue.put((str(full_tar_path), batch))
                    else:
                        logger.warning(f"Tar file not found: {full_tar_path}")
                        
            # Send poison pills to all worker types
            for _ in range(self.config.extraction_workers):
                tar_queue.put(None)
            for _ in range(self.config.docling_workers):
                extraction_queue.put(None)
            for _ in range(self.config.embedding_workers):
                chunk_queue.put(None)
                
        feeder_thread = threading.Thread(target=feed_tar_queue)
        feeder_thread.start()
        
        # Monitor progress
        try:
            while any(p.is_alive() for p in processes):
                time.sleep(30)
                stats = self.metrics.get_stats()
                logger.info(
                    f"Progress: Extracted={stats['pdfs_extracted']}, "
                    f"Converted={stats['pdfs_converted']}, "
                    f"Updated={stats['documents_updated']}, "
                    f"Rate={stats['conversion_rate']:.2f} PDFs/sec"
                )
                
        except KeyboardInterrupt:
            logger.warning("Shutting down...")
            stop_event.set()
            
        # Wait for completion
        feeder_thread.join()
        for p in processes:
            p.join()
            
        # Stop updater
        updater.stop()
        updater.join()
        
        # Final report
        stats = self.metrics.get_stats()
        logger.info("=" * 80)
        logger.info("PDF Processing Complete")
        logger.info(f"Total time: {stats['elapsed_seconds']/3600:.2f} hours")
        logger.info(f"PDFs processed: {stats['documents_updated']}")
        logger.info(f"Average rate: {stats['conversion_rate']:.2f} PDFs/second")
        logger.info(f"Chunks created: {stats['chunks_created']}")
        logger.info(f"Errors: {stats['extraction_errors']} extraction, {stats['conversion_errors']} conversion")
        logger.info("=" * 80)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Dual GPU PDF Processing Pipeline")
    parser.add_argument('--pdf-tar-dir', type=str, default='/mnt/data-cold/arxiv_data')
    parser.add_argument('--max-pdfs', type=int, help='Maximum PDFs to process')
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--db-name', type=str, default='arxiv_single_collection')
    parser.add_argument('--db-host', type=str, default='192.168.1.69')
    parser.add_argument('--docling-gpu', type=int, default=0)
    parser.add_argument('--embedding-gpu', type=int, default=1)
    parser.add_argument('--working-dir', type=str, default='/tmp/arxiv_pdf_processing')
    
    args = parser.parse_args()
    
    config = PDFConfig(
        pdf_tar_dir=args.pdf_tar_dir,
        max_pdfs=args.max_pdfs,
        batch_size=args.batch_size,
        db_name=args.db_name,
        db_host=args.db_host,
        docling_gpu=args.docling_gpu,
        embedding_gpu=args.embedding_gpu,
        working_dir=args.working_dir
    )
    
    pipeline = PDFProcessingPipeline(config)
    pipeline.run()

if __name__ == '__main__':
    main()