#!/usr/bin/env python3
"""
Safe Mode PDF Processing Pipeline
- Single worker per GPU with resource limits
- I/O throttling to prevent storage overload  
- Health checks and graceful failure handling
- Automatic pause on system stress
"""

import os
import sys
import json
import time
import signal
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import psutil

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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_safe_mode.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SafeModeConfig:
    """Configuration for safe mode PDF processing"""
    # Directories
    pdf_directory: str = "/mnt/data-cold/arxiv_data/pdf"
    
    # Database
    db_name: str = "base"
    collection_name: str = "arxiv_documents"
    db_host: str = "192.168.1.69"
    db_port: int = 8529
    db_username: str = "root"
    db_password: str = os.environ.get('ARANGO_PASSWORD', '')
    
    # Processing (CONSERVATIVE)
    batch_size: int = 1  # Process one PDF at a time
    max_pdfs_per_run: Optional[int] = 100  # Limit per run
    
    # Resource limits
    max_cpu_percent: float = 70.0  # Pause if CPU > 70%
    max_memory_percent: float = 70.0  # Pause if RAM > 70%
    max_gpu_memory_gb: float = 20.0  # Max GPU memory per process
    io_delay_seconds: float = 0.5  # Delay between I/O operations
    
    # Late chunking
    max_context_length: int = 32768
    chunk_size_tokens: int = 512
    chunk_stride_tokens: int = 256
    
    # Health check intervals
    health_check_interval: int = 10  # Check every N PDFs
    
    # Options
    dry_run: bool = False
    delete_after_processing: bool = False  # Safer to keep PDFs initially

class SystemHealthMonitor:
    """Monitor system health and throttle processing"""
    
    def __init__(self, config: SafeModeConfig):
        self.config = config
        self.check_count = 0
        
    def check_raid_health(self) -> bool:
        """Check if RAID arrays are healthy"""
        try:
            result = subprocess.run(['cat', '/proc/mdstat'], 
                                  capture_output=True, text=True)
            mdstat = result.stdout
            
            # Check for degraded arrays
            if 'degraded' in mdstat or '[_' in mdstat:
                logger.error("RAID array degraded!")
                return False
                
            # Check for recovery in progress
            if 'recovery' in mdstat or 'resync' in mdstat:
                logger.warning("RAID recovery in progress")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Failed to check RAID status: {e}")
            return False
    
    def check_resources(self) -> Tuple[bool, str]:
        """Check CPU, memory, and disk usage"""
        # CPU check
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > self.config.max_cpu_percent:
            return False, f"CPU usage too high: {cpu_percent:.1f}%"
        
        # Memory check
        memory = psutil.virtual_memory()
        if memory.percent > self.config.max_memory_percent:
            return False, f"Memory usage too high: {memory.percent:.1f}%"
        
        # Disk I/O check
        disk_io = psutil.disk_io_counters()
        if disk_io:
            # Simple check - could be more sophisticated
            io_wait = psutil.cpu_times_percent(interval=0.1).iowait
            if io_wait > 30:  # High I/O wait
                return False, f"High I/O wait: {io_wait:.1f}%"
        
        return True, "System resources OK"
    
    def check_gpu_memory(self) -> Tuple[bool, str]:
        """Check GPU memory usage"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                                   '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True)
            for line in result.stdout.strip().split('\n'):
                used, total = map(float, line.split(', '))
                used_gb = used / 1024
                if used_gb > self.config.max_gpu_memory_gb:
                    return False, f"GPU memory too high: {used_gb:.1f}GB"
            return True, "GPU memory OK"
        except Exception as e:
            logger.error(f"Failed to check GPU memory: {e}")
            return True, "GPU check skipped"
    
    def should_process(self) -> Tuple[bool, str]:
        """Check if it's safe to process next PDF"""
        self.check_count += 1
        
        # Always check RAID health
        if not self.check_raid_health():
            return False, "RAID unhealthy"
        
        # Periodic resource checks
        if self.check_count % self.config.health_check_interval == 0:
            ok, msg = self.check_resources()
            if not ok:
                return False, msg
                
            ok, msg = self.check_gpu_memory()
            if not ok:
                return False, msg
        
        return True, "OK"
    
    def wait_for_resources(self):
        """Wait for system resources to become available"""
        while True:
            ok, msg = self.should_process()
            if ok:
                break
            logger.warning(f"Waiting for resources: {msg}")
            time.sleep(30)  # Wait 30 seconds before retry

class SafeModePipeline:
    """Safe mode PDF processing pipeline"""
    
    def __init__(self, config: SafeModeConfig):
        self.config = config
        self.health_monitor = SystemHealthMonitor(config)
        self.processed_count = 0
        self.failed_count = 0
        self.start_time = time.time()
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self.shutdown_requested = False
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info("Shutdown requested, finishing current PDF...")
        self.shutdown_requested = True
        
    def initialize_models(self):
        """Initialize Docling and Jina models"""
        logger.info("Initializing models...")
        
        # Initialize Docling on GPU 0
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat
        
        pipeline_options = PdfPipelineOptions(
            do_ocr=False,
            do_table_structure=True
        )
        
        self.docling_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )
        
        # Initialize Jina on GPU 1  
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        model_name = "jinaai/jina-embeddings-v3"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.embedding_model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        self.embedding_model = self.embedding_model.cuda()
        self.embedding_model.eval()
        
        # Initialize database
        client = ArangoClient(hosts=f'http://{self.config.db_host}:{self.config.db_port}')
        self.db = client.db(
            self.config.db_name,
            username=self.config.db_username,
            password=self.config.db_password
        )
        self.collection = self.db.collection(self.config.collection_name)
        
        logger.info("Models initialized")
        
    def get_pdfs_to_process(self) -> List[Tuple[str, Path]]:
        """Get list of PDFs that need processing"""
        pdf_dir = Path(self.config.pdf_directory)
        all_pdfs = sorted(pdf_dir.glob("*.pdf"))
        
        if self.config.max_pdfs_per_run:
            all_pdfs = all_pdfs[:self.config.max_pdfs_per_run]
            
        pdfs_to_process = []
        
        for pdf_path in all_pdfs:
            arxiv_id = pdf_path.stem
            if '_' in arxiv_id and '.' not in arxiv_id:
                arxiv_id = arxiv_id.replace('_', '.', 1)
                
            # Check if already processed
            doc = self.collection.get(arxiv_id)
            if doc and not doc.get('pdf_content'):
                pdfs_to_process.append((arxiv_id, pdf_path))
                
        return pdfs_to_process
        
    def process_single_pdf(self, arxiv_id: str, pdf_path: Path) -> bool:
        """Process a single PDF with all safety checks"""
        try:
            # Check system health
            self.health_monitor.wait_for_resources()
            
            # I/O throttling
            time.sleep(self.config.io_delay_seconds)
            
            logger.info(f"Processing {arxiv_id}")
            start_time = time.time()
            
            # Step 1: Convert PDF to markdown (GPU 0)
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            result = self.docling_converter.convert(str(pdf_path))
            
            if not result or not hasattr(result, 'document'):
                logger.error(f"Failed to convert {arxiv_id}")
                return False
                
            markdown = result.document.export_to_markdown()
            if not markdown or len(markdown) < 100:
                logger.warning(f"No content extracted from {arxiv_id}")
                return False
                
            # Clear GPU 0 memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Step 2: Late chunking with embeddings (GPU 1)
            os.environ['CUDA_VISIBLE_DEVICES'] = '1'
            chunks = self.create_chunks_with_embeddings(markdown)
            
            if not chunks:
                logger.error(f"Failed to create chunks for {arxiv_id}")
                return False
                
            # Step 3: Save to database
            processing_time = time.time() - start_time
            
            pdf_content = {
                'markdown': markdown,
                'chunks': chunks,
                'extraction_metadata': {
                    'docling_version': '1.0',
                    'extraction_time': datetime.utcnow().isoformat() + 'Z',
                    'chunk_count': len(chunks),
                    'chunking_strategy': 'late_chunking_safe_mode',
                    'chunk_size_tokens': self.config.chunk_size_tokens,
                    'chunk_stride_tokens': self.config.chunk_stride_tokens,
                    'markdown_length': len(markdown),
                    'processing_time': processing_time
                }
            }
            
            self.collection.update({
                '_key': arxiv_id,
                'pdf_content': pdf_content,
                'pdf_status': {
                    'state': 'completed',
                    'last_updated': datetime.utcnow().isoformat() + 'Z',
                    'processing_time_seconds': processing_time
                }
            })
            
            # Delete PDF if configured
            if self.config.delete_after_processing:
                try:
                    pdf_path.unlink()
                    logger.info(f"Deleted {pdf_path}")
                except Exception as e:
                    logger.error(f"Failed to delete {pdf_path}: {e}")
                    
            logger.info(f"Processed {arxiv_id} in {processing_time:.1f}s with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {arxiv_id}: {e}")
            return False
            
    def create_chunks_with_embeddings(self, markdown: str) -> List[Dict]:
        """Create chunks with embeddings using late chunking"""
        try:
            # Tokenize
            inputs = self.tokenizer(
                markdown,
                return_tensors='pt',
                max_length=self.config.max_context_length,
                truncation=True,
                return_offsets_mapping=True,
                padding=True
            ).to('cuda')
            
            seq_len = inputs['input_ids'].shape[1]
            
            # Get embeddings
            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    outputs = self.embedding_model(
                        **{k: v for k, v in inputs.items() if k != 'offset_mapping'},
                        task='retrieval.passage'
                    )
                    all_token_embeddings = outputs.last_hidden_state[0]
            
            # Extract chunks
            chunks = []
            offset_mapping = inputs['offset_mapping'][0].cpu().numpy()
            
            for start_idx in range(0, seq_len - self.config.chunk_size_tokens + 1, 
                                 self.config.chunk_stride_tokens):
                end_idx = min(start_idx + self.config.chunk_size_tokens, seq_len)
                
                # Get chunk embedding
                chunk_embedding = all_token_embeddings[start_idx:end_idx].mean(dim=0)
                
                # Get text boundaries
                start_char = offset_mapping[start_idx][0]
                end_char = offset_mapping[end_idx - 1][1]
                chunk_text = markdown[start_char:end_char]
                
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
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating chunks: {e}")
            torch.cuda.empty_cache()
            return []
            
    def run(self):
        """Run the safe mode pipeline"""
        logger.info("Starting Safe Mode PDF Processing Pipeline")
        logger.info(f"Max PDFs per run: {self.config.max_pdfs_per_run}")
        logger.info(f"Resource limits - CPU: {self.config.max_cpu_percent}%, "
                   f"Memory: {self.config.max_memory_percent}%, "
                   f"GPU: {self.config.max_gpu_memory_gb}GB")
        
        # Initialize models
        self.initialize_models()
        
        # Get PDFs to process
        pdfs_to_process = self.get_pdfs_to_process()
        logger.info(f"Found {len(pdfs_to_process)} PDFs to process")
        
        if not pdfs_to_process:
            logger.info("No PDFs to process")
            return
            
        # Process PDFs one by one
        with tqdm(total=len(pdfs_to_process), desc="Processing PDFs") as pbar:
            for arxiv_id, pdf_path in pdfs_to_process:
                if self.shutdown_requested:
                    logger.info("Shutdown requested, stopping...")
                    break
                    
                success = self.process_single_pdf(arxiv_id, pdf_path)
                
                if success:
                    self.processed_count += 1
                else:
                    self.failed_count += 1
                    
                pbar.update(1)
                pbar.set_postfix({
                    'processed': self.processed_count,
                    'failed': self.failed_count,
                    'rate': f"{self.processed_count / (time.time() - self.start_time):.2f}/s"
                })
                
        # Final report
        self.print_final_report()
        
    def print_final_report(self):
        """Print processing summary"""
        elapsed = time.time() - self.start_time
        
        print("\n" + "=" * 80)
        print("Safe Mode PDF Processing Complete")
        print("=" * 80)
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"PDFs processed: {self.processed_count}")
        print(f"PDFs failed: {self.failed_count}")
        print(f"Processing rate: {self.processed_count/elapsed:.2f} PDFs/sec")
        print("=" * 80)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Safe mode PDF processing")
    parser.add_argument('--pdf-directory', type=str, default='/mnt/data-cold/arxiv_data/pdf')
    parser.add_argument('--max-pdfs', type=int, default=100, help='Maximum PDFs per run')
    parser.add_argument('--db-name', type=str, default='base')
    parser.add_argument('--db-host', type=str, default='192.168.1.69')
    parser.add_argument('--delete-pdfs', action='store_true', help='Delete PDFs after processing')
    parser.add_argument('--dry-run', action='store_true')
    
    args = parser.parse_args()
    
    if not os.environ.get('ARANGO_PASSWORD'):
        logger.error("ARANGO_PASSWORD environment variable not set")
        sys.exit(1)
        
    config = SafeModeConfig(
        pdf_directory=args.pdf_directory,
        max_pdfs_per_run=args.max_pdfs,
        db_name=args.db_name,
        db_host=args.db_host,
        delete_after_processing=args.delete_pdfs,
        dry_run=args.dry_run
    )
    
    pipeline = SafeModePipeline(config)
    pipeline.run()

if __name__ == '__main__':
    main()