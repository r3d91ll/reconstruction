#!/usr/bin/env python3
"""
Multi-Worker PDF Processing Pipeline with Late Chunking
- Multiple Docling workers on GPU 0 for parallel PDF conversion
- Single Jina worker on GPU 1 for late chunking
- Queue-based architecture for maximum throughput
"""

import os
import sys
import json
import time
import queue
import logging
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
import threading

import numpy as np
from tqdm import tqdm
from arango import ArangoClient

# Set environment variable before imports
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import torch
from transformers import AutoTokenizer, AutoModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(processName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_multi_worker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class MultiWorkerConfig:
    """Configuration for multi-worker PDF processing"""
    # Directories
    pdf_directory: str = "/mnt/data-cold/arxiv_data/pdf"
    
    # Database
    db_name: str = "base"
    collection_name: str = "arxiv_documents"
    db_host: str = "192.168.1.69"
    db_port: int = 8529
    db_username: str = "root"
    db_password: str = os.environ.get('ARANGO_PASSWORD', '')
    
    # Workers
    docling_workers: int = 4  # Multiple Docling instances on GPU 0
    embedding_workers: int = 1  # Single Jina worker on GPU 1
    
    # Processing
    pdf_batch_size: int = 10  # PDFs per batch check
    
    # Queues
    pdf_queue_size: int = 20  # PDFs waiting for Docling
    markdown_queue_size: int = 10  # Markdowns waiting for embedding
    db_queue_size: int = 10  # Documents waiting for database
    
    # Late chunking
    max_context_length: int = 32768
    chunk_size_tokens: int = 512
    chunk_stride_tokens: int = 256
    
    # GPUs
    docling_gpu: int = 0
    embedding_gpu: int = 1
    
    # GPU memory fractions for each Docling worker
    docling_memory_fraction: float = 0.22  # ~10GB per worker (4 workers = 40GB/48GB)
    
    # Options
    dry_run: bool = False
    max_pdfs: Optional[int] = None

@dataclass
class PDFWork:
    """PDF ready for processing"""
    arxiv_id: str
    pdf_path: Path
    
@dataclass
class MarkdownWork:
    """Markdown ready for chunking"""
    arxiv_id: str
    markdown: str
    pdf_path: Path  # For deletion after processing
    start_time: float

@dataclass
class ChunkedWork:
    """Chunked document ready for database"""
    arxiv_id: str
    markdown: str
    chunks: List[Dict]
    pdf_path: Path
    total_time: float

class ProcessingStats:
    """Thread-safe processing statistics"""
    
    def __init__(self):
        self.start_time = time.time()
        self._lock = threading.Lock()
        self.pdfs_found = 0
        self.pdfs_with_metadata = 0
        self.pdfs_processed = 0
        self.pdfs_failed = 0
        self.pdfs_orphaned = 0
        self.pdfs_deleted = 0
        self.total_chunks = 0
        self.docling_times = []
        self.embedding_times = []
        
    def add_docling_time(self, duration: float):
        with self._lock:
            self.docling_times.append(duration)
            
    def add_embedding_time(self, duration: float):
        with self._lock:
            self.embedding_times.append(duration)
            
    def increment_processed(self):
        with self._lock:
            self.pdfs_processed += 1
            
    def add_chunks(self, count: int):
        with self._lock:
            self.total_chunks += count
            
    def get_summary(self):
        with self._lock:
            elapsed = time.time() - self.start_time
            avg_docling = np.mean(self.docling_times) if self.docling_times else 0
            avg_embedding = np.mean(self.embedding_times) if self.embedding_times else 0
            
            return {
                'elapsed_time': elapsed,
                'pdfs_processed': self.pdfs_processed,
                'total_chunks': self.total_chunks,
                'processing_rate': self.pdfs_processed / elapsed if elapsed > 0 else 0,
                'avg_docling_time': avg_docling,
                'avg_embedding_time': avg_embedding,
                'docling_samples': len(self.docling_times),
                'embedding_samples': len(self.embedding_times)
            }

def docling_worker_process(
    worker_id: int,
    gpu_id: int,
    memory_fraction: float,
    pdf_queue: mp.Queue,
    markdown_queue: mp.Queue,
    stop_event: mp.Event,
    stats: ProcessingStats
):
    """Docling worker process - runs on GPU 0 with memory fraction"""
    # Set GPU before any imports
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Now import and configure PyTorch
    import torch
    torch.cuda.set_per_process_memory_fraction(memory_fraction)
    
    # Import Docling after GPU setup
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    
    logger = logging.getLogger(f"DoclingWorker-{worker_id}")
    logger.info(f"Starting on GPU {gpu_id} with {memory_fraction*100}% memory")
    
    # Initialize Docling
    pipeline_options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=True  # Keep for charts/graphs
    )
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )
    
    logger.info("Docling initialized")
    
    # Process PDFs
    while not stop_event.is_set():
        try:
            pdf_work = pdf_queue.get(timeout=1.0)
            if pdf_work is None:  # Poison pill
                break
                
            start_time = time.time()
            logger.info(f"Converting {pdf_work.arxiv_id}")
            
            try:
                # Convert PDF
                result = converter.convert(str(pdf_work.pdf_path))
                
                if result and hasattr(result, 'document'):
                    markdown = result.document.export_to_markdown()
                    
                    if markdown and len(markdown) > 100:
                        duration = time.time() - start_time
                        stats.add_docling_time(duration)
                        
                        # Send to embedding queue
                        markdown_work = MarkdownWork(
                            arxiv_id=pdf_work.arxiv_id,
                            markdown=markdown,
                            pdf_path=pdf_work.pdf_path,
                            start_time=start_time
                        )
                        markdown_queue.put(markdown_work)
                        
                        logger.info(f"Converted {pdf_work.arxiv_id} in {duration:.1f}s ({len(markdown)} chars)")
                    else:
                        logger.warning(f"No content for {pdf_work.arxiv_id}")
                else:
                    logger.error(f"Conversion failed for {pdf_work.arxiv_id}")
                    
            except Exception as e:
                logger.error(f"Error converting {pdf_work.arxiv_id}: {e}")
                
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Worker error: {e}")
            
    logger.info("Worker stopped")

def embedding_worker_process(
    gpu_id: int,
    markdown_queue: mp.Queue,
    db_queue: mp.Queue,
    stop_event: mp.Event,
    config: MultiWorkerConfig,
    stats: ProcessingStats
):
    """Embedding worker with late chunking - runs on GPU 1"""
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    import torch
    device = f'cuda:0'  # Since we set CUDA_VISIBLE_DEVICES, it's always 0
    
    logger = logging.getLogger(f"EmbeddingWorker-GPU{gpu_id}")
    logger.info(f"Starting on GPU {gpu_id}")
    
    # Initialize Jina
    model_name = "jinaai/jina-embeddings-v3"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    model = model.to(device)
    model.eval()
    
    logger.info("Jina model loaded")
    
    # Process markdowns
    while not stop_event.is_set():
        try:
            markdown_work = markdown_queue.get(timeout=1.0)
            if markdown_work is None:
                break
                
            start_time = time.time()
            logger.info(f"Late chunking {markdown_work.arxiv_id}")
            
            try:
                # Tokenize
                inputs = tokenizer(
                    markdown_work.markdown,
                    return_tensors='pt',
                    max_length=config.max_context_length,
                    truncation=True,
                    return_offsets_mapping=True,
                    padding=True
                ).to(device)
                
                seq_len = inputs['input_ids'].shape[1]
                
                # Get embeddings
                with torch.amp.autocast('cuda'):
                    with torch.no_grad():
                        outputs = model(
                            **{k: v for k, v in inputs.items() if k != 'offset_mapping'},
                            task='retrieval.passage'
                        )
                        all_token_embeddings = outputs.last_hidden_state[0]
                
                # Extract chunks
                chunks = []
                offset_mapping = inputs['offset_mapping'][0].cpu().numpy()
                chunk_size = config.chunk_size_tokens
                stride = config.chunk_stride_tokens
                
                for start_idx in range(0, seq_len - chunk_size + 1, stride):
                    end_idx = min(start_idx + chunk_size, seq_len)
                    
                    # Get chunk embedding
                    chunk_embedding = all_token_embeddings[start_idx:end_idx].mean(dim=0)
                    
                    # Get text boundaries
                    start_char = offset_mapping[start_idx][0]
                    end_char = offset_mapping[end_idx - 1][1]
                    chunk_text = markdown_work.markdown[start_char:end_char]
                    
                    chunks.append({
                        'chunk_id': len(chunks),
                        'text': chunk_text,
                        'embedding': chunk_embedding.cpu().numpy().tolist(),
                        'metadata': {
                            'token_start': start_idx,
                            'token_end': end_idx,
                            'char_start': int(start_char),
                            'char_end': int(end_char)
                        }
                    })
                
                # Clean up GPU memory
                del inputs, outputs, all_token_embeddings
                torch.cuda.empty_cache()
                
                duration = time.time() - start_time
                stats.add_embedding_time(duration)
                
                # Send to database queue
                chunked_work = ChunkedWork(
                    arxiv_id=markdown_work.arxiv_id,
                    markdown=markdown_work.markdown,
                    chunks=chunks,
                    pdf_path=markdown_work.pdf_path,
                    total_time=time.time() - markdown_work.start_time
                )
                db_queue.put(chunked_work)
                
                logger.info(f"Created {len(chunks)} chunks for {markdown_work.arxiv_id} in {duration:.1f}s")
                
            except Exception as e:
                logger.error(f"Error chunking {markdown_work.arxiv_id}: {e}")
                torch.cuda.empty_cache()
                
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Worker error: {e}")
            
    logger.info("Worker stopped")

class DatabaseWriter(threading.Thread):
    """Database writer thread"""
    
    def __init__(self, db_queue: queue.Queue, config: MultiWorkerConfig, stats: ProcessingStats):
        super().__init__()
        self.db_queue = db_queue
        self.config = config
        self.stats = stats
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
        """Write documents to database"""
        while not self.stop_event.is_set() or not self.db_queue.empty():
            try:
                chunked_work = self.db_queue.get(timeout=1.0)
                
                # Prepare document update
                pdf_content = {
                    'markdown': chunked_work.markdown,
                    'chunks': chunked_work.chunks,
                    'extraction_metadata': {
                        'docling_version': '1.0',
                        'extraction_time': datetime.utcnow().isoformat() + 'Z',
                        'chunk_count': len(chunked_work.chunks),
                        'chunking_strategy': 'late_chunking',
                        'chunk_size_tokens': self.config.chunk_size_tokens,
                        'chunk_stride_tokens': self.config.chunk_stride_tokens,
                        'markdown_length': len(chunked_work.markdown),
                        'processing_time': chunked_work.total_time
                    }
                }
                
                # Update document
                self.collection.update({
                    '_key': chunked_work.arxiv_id,
                    'pdf_content': pdf_content,
                    'pdf_status': {
                        'state': 'completed',
                        'last_updated': datetime.utcnow().isoformat() + 'Z',
                        'processing_time_seconds': chunked_work.total_time
                    }
                })
                
                # Delete PDF file
                try:
                    chunked_work.pdf_path.unlink()
                    logger.info(f"Deleted {chunked_work.pdf_path}")
                except:
                    pass
                    
                self.stats.increment_processed()
                self.stats.add_chunks(len(chunked_work.chunks))
                
                logger.info(f"Saved {chunked_work.arxiv_id} with {len(chunked_work.chunks)} chunks")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Database error: {e}")
                
    def stop(self):
        self.stop_event.set()

class MultiWorkerPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config: MultiWorkerConfig):
        self.config = config
        self.stats = ProcessingStats()
        
    def get_pdfs_to_process(self):
        """Get PDFs that need processing"""
        pdf_dir = Path(self.config.pdf_directory)
        all_pdfs = sorted(pdf_dir.glob("*.pdf"))
        
        if self.config.max_pdfs:
            all_pdfs = all_pdfs[:self.config.max_pdfs]
            
        # Check database for metadata
        client = ArangoClient(hosts=f'http://{self.config.db_host}:{self.config.db_port}')
        db = client.db(
            self.config.db_name,
            username=self.config.db_username,
            password=self.config.db_password
        )
        collection = db.collection(self.config.collection_name)
        
        pdfs_to_process = []
        
        for pdf_path in all_pdfs:
            arxiv_id = pdf_path.stem
            if '_' in arxiv_id and '.' not in arxiv_id:
                arxiv_id = arxiv_id.replace('_', '.', 1)
                
            # Check if document exists and needs processing
            doc = collection.get(arxiv_id)
            if doc and not doc.get('pdf_content'):
                pdfs_to_process.append(PDFWork(arxiv_id=arxiv_id, pdf_path=pdf_path))
                
        return pdfs_to_process
        
    def run(self):
        """Run the multi-worker pipeline"""
        logger.info("Starting Multi-Worker PDF Processing Pipeline")
        logger.info(f"Docling workers: {self.config.docling_workers} on GPU {self.config.docling_gpu}")
        logger.info(f"Embedding workers: {self.config.embedding_workers} on GPU {self.config.embedding_gpu}")
        
        # Get PDFs to process
        pdfs_to_process = self.get_pdfs_to_process()
        logger.info(f"Found {len(pdfs_to_process)} PDFs to process")
        
        if not pdfs_to_process:
            logger.info("No PDFs to process")
            return
            
        # Create queues
        pdf_queue = mp.Queue(maxsize=self.config.pdf_queue_size)
        markdown_queue = mp.Queue(maxsize=self.config.markdown_queue_size)
        db_queue = mp.Queue(maxsize=self.config.db_queue_size)
        
        # Start database writer
        db_writer = DatabaseWriter(db_queue, self.config, self.stats)
        db_writer.start()
        
        # Start workers
        stop_event = mp.Event()
        processes = []
        
        # Start Docling workers
        for i in range(self.config.docling_workers):
            p = mp.Process(
                target=docling_worker_process,
                args=(
                    i,
                    self.config.docling_gpu,
                    self.config.docling_memory_fraction,
                    pdf_queue,
                    markdown_queue,
                    stop_event,
                    self.stats
                )
            )
            p.start()
            processes.append(p)
            
        # Start embedding worker
        p = mp.Process(
            target=embedding_worker_process,
            args=(
                self.config.embedding_gpu,
                markdown_queue,
                db_queue,
                stop_event,
                self.config,
                self.stats
            )
        )
        p.start()
        processes.append(p)
        
        # Feed PDF queue
        def feed_queue():
            for pdf_work in pdfs_to_process:
                pdf_queue.put(pdf_work)
            # Send poison pills
            for _ in range(self.config.docling_workers):
                pdf_queue.put(None)
                
        feeder = threading.Thread(target=feed_queue)
        feeder.start()
        
        # Monitor progress
        last_report = time.time()
        
        try:
            with tqdm(total=len(pdfs_to_process), desc="Processing PDFs") as pbar:
                last_processed = 0
                
                while any(p.is_alive() for p in processes) or not db_queue.empty():
                    time.sleep(1)
                    
                    # Update progress
                    current_processed = self.stats.pdfs_processed
                    if current_processed > last_processed:
                        pbar.update(current_processed - last_processed)
                        last_processed = current_processed
                        
                    # Periodic report
                    if time.time() - last_report > 30:
                        stats = self.stats.get_summary()
                        logger.info(
                            f"Progress: {stats['pdfs_processed']}/{len(pdfs_to_process)} | "
                            f"Rate: {stats['processing_rate']:.2f} PDFs/s | "
                            f"Avg Docling: {stats['avg_docling_time']:.1f}s | "
                            f"Avg Embedding: {stats['avg_embedding_time']:.1f}s"
                        )
                        last_report = time.time()
                        
        except KeyboardInterrupt:
            logger.warning("Shutting down...")
            stop_event.set()
            
        # Wait for completion
        feeder.join()
        for p in processes:
            p.join()
            
        # Stop database writer
        db_writer.stop()
        db_writer.join()
        
        # Final report
        self.print_final_report()
        
    def print_final_report(self):
        """Print final processing report"""
        stats = self.stats.get_summary()
        
        print("\n" + "=" * 80)
        print("Multi-Worker PDF Processing Complete")
        print("=" * 80)
        print(f"Total time: {stats['elapsed_time']/60:.1f} minutes")
        print(f"PDFs processed: {stats['pdfs_processed']}")
        print(f"Total chunks created: {stats['total_chunks']}")
        print(f"Processing rate: {stats['processing_rate']:.2f} PDFs/sec")
        print(f"\nPerformance:")
        print(f"  - Avg Docling time: {stats['avg_docling_time']:.1f}s")
        print(f"  - Avg Embedding time: {stats['avg_embedding_time']:.1f}s")
        print(f"  - Speedup: {self.config.docling_workers}x parallel Docling")
        print("=" * 80)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-worker PDF processing")
    parser.add_argument('--pdf-directory', type=str, default='/mnt/data-cold/arxiv_data/pdf')
    parser.add_argument('--max-pdfs', type=int, help='Maximum PDFs to process')
    parser.add_argument('--docling-workers', type=int, default=4, help='Number of Docling workers')
    parser.add_argument('--db-name', type=str, default='base')
    parser.add_argument('--db-host', type=str, default='192.168.1.69')
    parser.add_argument('--dry-run', action='store_true')
    
    args = parser.parse_args()
    
    if not os.environ.get('ARANGO_PASSWORD'):
        logger.error("ARANGO_PASSWORD environment variable not set")
        sys.exit(1)
        
    config = MultiWorkerConfig(
        pdf_directory=args.pdf_directory,
        max_pdfs=args.max_pdfs,
        docling_workers=args.docling_workers,
        db_name=args.db_name,
        db_host=args.db_host,
        dry_run=args.dry_run
    )
    
    pipeline = MultiWorkerPipeline(config)
    pipeline.run()

if __name__ == '__main__':
    main()