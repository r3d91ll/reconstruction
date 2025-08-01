#!/usr/bin/env python3
"""
Directory-based PDF Processing Pipeline with Late Chunking
- Processes PDFs from /mnt/data-cold/arxiv_data/pdf directory
- Uses Docling on GPU 0 for PDF → Markdown
- Uses Jina late chunking on GPU 1 for embeddings
- Everything stays in memory until database write
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        logging.FileHandler('pdf_late_chunking.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class LateChunkingConfig:
    """Configuration for PDF processing with late chunking"""
    # Directories
    pdf_directory: str = "/mnt/data-cold/arxiv_data/pdf"
    
    # Database
    db_name: str = "base"
    collection_name: str = "arxiv_documents"
    db_host: str = "192.168.1.69"
    db_port: int = 8529
    db_username: str = "root"
    db_password: str = os.environ.get('ARANGO_PASSWORD', '')
    
    # Processing
    batch_size: int = 5  # PDFs per batch (smaller for late chunking)
    max_workers: int = 2  # Parallel processing threads
    
    # Late chunking parameters
    max_context_length: int = 32768  # Max tokens for Jina
    chunk_size_tokens: int = 512
    chunk_stride_tokens: int = 256
    
    # GPUs
    docling_gpu: int = 0
    embedding_gpu: int = 1
    
    # Options
    dry_run: bool = False
    max_pdfs: Optional[int] = None
    
    # Memory optimization
    use_nvlink: bool = True  # Try to use NVLink for GPU transfer

@dataclass
class LateChunkOutput:
    """Output from late chunking"""
    arxiv_id: str
    chunk_embeddings: List[np.ndarray]
    chunk_texts: List[str]
    chunk_metadata: List[Dict]
    total_tokens: int
    processing_time: float

class ProcessingStats:
    """Track processing statistics"""
    
    def __init__(self):
        self.start_time = time.time()
        self.pdfs_found = 0
        self.pdfs_with_metadata = 0
        self.pdfs_already_processed = 0
        self.pdfs_newly_processed = 0
        self.pdfs_failed = 0
        self.pdfs_orphaned = 0
        self.pdfs_deleted = 0
        self.total_chunks_created = 0
        self.orphaned_list = []
        self.failed_list = []
        
    def get_summary(self):
        elapsed = time.time() - self.start_time
        return {
            'elapsed_time': elapsed,
            'pdfs_found': self.pdfs_found,
            'pdfs_with_metadata': self.pdfs_with_metadata,
            'pdfs_already_processed': self.pdfs_already_processed,
            'pdfs_newly_processed': self.pdfs_newly_processed,
            'pdfs_failed': self.pdfs_failed,
            'pdfs_orphaned': self.pdfs_orphaned,
            'pdfs_deleted': self.pdfs_deleted,
            'total_chunks': self.total_chunks_created,
            'processing_rate': self.pdfs_newly_processed / elapsed if elapsed > 0 else 0,
            'orphaned_files': self.orphaned_list[:10],
            'failed_files': self.failed_list[:10]
        }

class DatabaseChecker:
    """Check database for metadata and processing status"""
    
    def __init__(self, config: LateChunkingConfig):
        self.config = config
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
        
    def batch_check_status(self, arxiv_ids: List[str]) -> Dict[str, Tuple[bool, bool]]:
        """Batch check multiple documents"""
        results = {}
        
        query = """
        FOR id IN @ids
            LET doc = DOCUMENT(@@collection, id)
            RETURN {
                id: id,
                exists: doc != null,
                has_pdf: doc != null AND doc.pdf_content != null AND doc.pdf_content.chunks != null
            }
        """
        
        try:
            cursor = self.db.aql.execute(
                query,
                bind_vars={
                    'ids': arxiv_ids,
                    '@collection': self.config.collection_name
                }
            )
            
            for result in cursor:
                results[result['id']] = (result['exists'], result['has_pdf'])
                
        except Exception as e:
            logger.error(f"Batch check error: {e}")
            
        return results

class DoclingConverter:
    """Convert PDFs to markdown using Docling on GPU 0"""
    
    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self._initialize_docling()
        
    def _initialize_docling(self):
        """Initialize Docling with GPU support"""
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.datamodel.base_models import InputFormat
            
            # Optimize for speed - no OCR, minimal processing
            pipeline_options = PdfPipelineOptions(
                do_ocr=False,
                do_table_structure=False  # Skip tables for speed
            )
            
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options
                    )
                }
            )
            
            logger.info(f"Docling initialized on GPU {self.gpu_id}")
            
        except ImportError:
            logger.error("Docling not installed. Please install with: pip install docling")
            raise
            
    def convert_pdf(self, pdf_path: Path) -> Optional[str]:
        """Convert PDF to markdown"""
        try:
            with torch.cuda.device(self.gpu_id):
                result = self.converter.convert(str(pdf_path))
                
            if result and hasattr(result, 'document'):
                markdown = result.document.export_to_markdown()
                return markdown
            else:
                logger.warning(f"No markdown output for {pdf_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error converting {pdf_path}: {e}")
            return None
        finally:
            torch.cuda.empty_cache()

class JinaLateChunkingEmbedder:
    """Jina embeddings with late chunking on GPU 1"""
    
    def __init__(self, device='cuda:1', config: LateChunkingConfig = None):
        self.device = device
        self.config = config or LateChunkingConfig()
        self.model_name = "jinaai/jina-embeddings-v3"
        
        logger.info(f"Loading Jina model on {device} for late chunking...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_name, 
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info("Jina model loaded for late chunking")
        
    def late_chunk_document(self, text: str, arxiv_id: str) -> LateChunkOutput:
        """Process document with late chunking"""
        start_time = time.time()
        
        try:
            # Tokenize with full context
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                max_length=self.config.max_context_length,
                truncation=True,
                return_offsets_mapping=True,
                padding=True
            ).to(self.device)
            
            seq_len = inputs['input_ids'].shape[1]
            
            # Get embeddings with mixed precision
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    outputs = self.model(
                        **{k: v for k, v in inputs.items() if k != 'offset_mapping'},
                        task='retrieval.passage'  # Use passage task for documents
                    )
                    
                    # Check if model returns multi-vector embeddings
                    if hasattr(outputs, 'last_hidden_state'):
                        # Use last hidden state for token embeddings
                        all_token_embeddings = outputs.last_hidden_state[0]
                    else:
                        # Fall back to regular embeddings
                        logger.warning(f"Model doesn't support multi-vector embeddings, using mean pooling")
                        embeddings = outputs.last_hidden_state.mean(dim=1)
                        return self._fallback_chunking(text, embeddings[0], arxiv_id, time.time() - start_time)
            
            # Extract chunks using sliding window
            chunks = self._extract_chunks(
                all_token_embeddings,
                inputs['offset_mapping'][0].cpu().numpy(),
                text,
                seq_len
            )
            
            # Clean up GPU memory
            del inputs, outputs, all_token_embeddings
            torch.cuda.empty_cache()
            
            return LateChunkOutput(
                arxiv_id=arxiv_id,
                chunk_embeddings=[c['embedding'] for c in chunks],
                chunk_texts=[c['text'] for c in chunks],
                chunk_metadata=[c['metadata'] for c in chunks],
                total_tokens=seq_len,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Late chunking failed for {arxiv_id}: {e}")
            raise
            
    def _extract_chunks(self, token_embeddings, offset_mapping, text, seq_len):
        """Extract overlapping chunks from token embeddings"""
        chunks = []
        chunk_size = self.config.chunk_size_tokens
        stride = self.config.chunk_stride_tokens
        
        for start_idx in range(0, seq_len - chunk_size + 1, stride):
            end_idx = min(start_idx + chunk_size, seq_len)
            
            # Get chunk embedding by averaging token embeddings
            chunk_embedding = token_embeddings[start_idx:end_idx].mean(dim=0)
            
            # Get text boundaries
            start_char = offset_mapping[start_idx][0]
            end_char = offset_mapping[end_idx - 1][1]
            chunk_text = text[start_char:end_char]
            
            chunks.append({
                'embedding': chunk_embedding.cpu().numpy(),
                'text': chunk_text,
                'metadata': {
                    'token_start': int(start_idx),
                    'token_end': int(end_idx),
                    'char_start': int(start_char),
                    'char_end': int(end_char),
                    'tokens': int(end_idx - start_idx)
                }
            })
            
        return chunks
        
    def _fallback_chunking(self, text: str, single_embedding, arxiv_id: str, elapsed: float):
        """Fallback to simple chunking if late chunking not supported"""
        # Simple character-based chunking
        chunk_size_chars = self.config.chunk_size_tokens * 4  # Rough approximation
        chunks = []
        
        for i in range(0, len(text), chunk_size_chars):
            chunk_text = text[i:i + chunk_size_chars]
            chunks.append({
                'embedding': single_embedding.cpu().numpy(),
                'text': chunk_text,
                'metadata': {
                    'char_start': i,
                    'char_end': i + len(chunk_text),
                    'tokens': len(chunk_text) // 4
                }
            })
            
        return LateChunkOutput(
            arxiv_id=arxiv_id,
            chunk_embeddings=[c['embedding'] for c in chunks],
            chunk_texts=[c['text'] for c in chunks],
            chunk_metadata=[c['metadata'] for c in chunks],
            total_tokens=len(text) // 4,
            processing_time=elapsed
        )

class PDFProcessor:
    """Process PDFs with late chunking"""
    
    def __init__(self, config: LateChunkingConfig):
        self.config = config
        self.converter = DoclingConverter(config.docling_gpu)
        self.embedder = JinaLateChunkingEmbedder(f'cuda:{config.embedding_gpu}', config)
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
        
    def process_pdf(self, pdf_path: Path, arxiv_id: str) -> Tuple[bool, int]:
        """
        Process a single PDF and update database
        Returns: (success, num_chunks)
        """
        try:
            # Convert to markdown on GPU 0
            logger.info(f"Converting {arxiv_id} to markdown...")
            markdown = self.converter.convert_pdf(pdf_path)
            
            if not markdown:
                logger.error(f"Failed to convert {arxiv_id} to markdown")
                return False, 0
                
            # Late chunking on GPU 1
            logger.info(f"Late chunking {arxiv_id} ({len(markdown)} chars)...")
            chunk_output = self.embedder.late_chunk_document(markdown, arxiv_id)
            
            logger.info(f"Created {len(chunk_output.chunk_embeddings)} chunks for {arxiv_id}")
            
            # Prepare chunks for database
            chunks = []
            for i in range(len(chunk_output.chunk_embeddings)):
                chunks.append({
                    'chunk_id': i,
                    'text': chunk_output.chunk_texts[i],
                    'embedding': chunk_output.chunk_embeddings[i].tolist(),
                    'metadata': chunk_output.chunk_metadata[i]
                })
                
            # Update database
            pdf_content = {
                'markdown': markdown,
                'chunks': chunks,
                'extraction_metadata': {
                    'docling_version': '1.0',
                    'extraction_time': datetime.utcnow().isoformat() + 'Z',
                    'chunk_count': len(chunks),
                    'total_tokens': chunk_output.total_tokens,
                    'chunking_strategy': 'late_chunking',
                    'chunk_size_tokens': self.config.chunk_size_tokens,
                    'chunk_stride_tokens': self.config.chunk_stride_tokens,
                    'pdf_file_size': pdf_path.stat().st_size,
                    'markdown_length': len(markdown),
                    'processing_time': chunk_output.processing_time
                }
            }
            
            self.collection.update({
                '_key': arxiv_id,
                'pdf_content': pdf_content,
                'pdf_status': {
                    'state': 'completed',
                    'last_updated': datetime.utcnow().isoformat() + 'Z',
                    'processing_time_seconds': chunk_output.processing_time
                }
            })
            
            logger.info(f"Successfully processed {arxiv_id} with {len(chunks)} chunks")
            return True, len(chunks)
            
        except Exception as e:
            logger.error(f"Error processing {arxiv_id}: {e}")
            
            # Update with error status
            try:
                self.collection.update({
                    '_key': arxiv_id,
                    'pdf_status': {
                        'state': 'failed',
                        'last_updated': datetime.utcnow().isoformat() + 'Z',
                        'error_message': str(e)
                    }
                })
            except:
                pass
                
            return False, 0

class DirectoryPDFPipeline:
    """Main pipeline for processing PDFs from directory with late chunking"""
    
    def __init__(self, config: LateChunkingConfig):
        self.config = config
        self.stats = ProcessingStats()
        self.db_checker = DatabaseChecker(config)
        self.processor = PDFProcessor(config)
        
    def get_pdf_files(self) -> List[Path]:
        """Get all PDF files from directory"""
        pdf_dir = Path(self.config.pdf_directory)
        if not pdf_dir.exists():
            logger.error(f"PDF directory not found: {pdf_dir}")
            return []
            
        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        
        if self.config.max_pdfs:
            pdf_files = pdf_files[:self.config.max_pdfs]
            
        return pdf_files
        
    def extract_arxiv_id(self, pdf_path: Path) -> Optional[str]:
        """Extract arXiv ID from filename"""
        filename = pdf_path.stem
        
        # Replace underscore with dot if present
        if '_' in filename and '.' not in filename:
            filename = filename.replace('_', '.', 1)
            
        return filename
        
    def process_batch(self, pdf_batch: List[Path]) -> Dict[str, str]:
        """Process a batch of PDFs"""
        results = {}
        files_to_delete = []
        
        # Extract arXiv IDs
        id_to_path = {}
        for pdf_path in pdf_batch:
            arxiv_id = self.extract_arxiv_id(pdf_path)
            if arxiv_id:
                id_to_path[arxiv_id] = pdf_path
                
        # Batch check database status
        db_status = self.db_checker.batch_check_status(list(id_to_path.keys()))
        
        # Process each PDF based on status
        for arxiv_id, pdf_path in id_to_path.items():
            has_metadata, is_processed = db_status.get(arxiv_id, (False, False))
            
            if not has_metadata:
                # No metadata - leave file in place
                self.stats.pdfs_orphaned += 1
                self.stats.orphaned_list.append(str(pdf_path))
                results[str(pdf_path)] = "orphaned"
                logger.warning(f"No metadata for {arxiv_id}, leaving {pdf_path}")
                
            elif is_processed:
                # Already processed - mark for deletion
                if not self.config.dry_run:
                    files_to_delete.append(pdf_path)
                self.stats.pdfs_already_processed += 1
                results[str(pdf_path)] = "already_processed"
                logger.info(f"Already processed {arxiv_id}, will delete {pdf_path}")
                
            else:
                # Has metadata but not processed - process it
                logger.info(f"Processing {arxiv_id}...")
                
                if not self.config.dry_run:
                    success, num_chunks = self.processor.process_pdf(pdf_path, arxiv_id)
                    
                    if success:
                        # Mark for deletion after successful processing
                        files_to_delete.append(pdf_path)
                        self.stats.pdfs_newly_processed += 1
                        self.stats.total_chunks_created += num_chunks
                        results[str(pdf_path)] = "processed"
                    else:
                        self.stats.pdfs_failed += 1
                        self.stats.failed_list.append(str(pdf_path))
                        results[str(pdf_path)] = "failed"
                else:
                    results[str(pdf_path)] = "would_process"
        
        # Delete files after all processing is complete
        for pdf_path in files_to_delete:
            try:
                pdf_path.unlink()
                self.stats.pdfs_deleted += 1
                logger.info(f"Deleted {pdf_path}")
            except Exception as e:
                logger.error(f"Failed to delete {pdf_path}: {e}")
                    
        return results
        
    def run(self):
        """Run the pipeline"""
        logger.info("Starting Directory-based PDF Processing with Late Chunking")
        logger.info(f"PDF Directory: {self.config.pdf_directory}")
        logger.info(f"Late chunking: {self.config.chunk_size_tokens} tokens, stride {self.config.chunk_stride_tokens}")
        logger.info(f"Dry run: {self.config.dry_run}")
        
        # Get all PDFs
        pdf_files = self.get_pdf_files()
        self.stats.pdfs_found = len(pdf_files)
        
        logger.info(f"Found {len(pdf_files)} PDF files to check")
        
        if not pdf_files:
            logger.info("No PDF files found in directory")
            return
            
        # Process in batches
        batch_results = {}
        
        with tqdm(total=len(pdf_files), desc="Processing PDFs") as pbar:
            for i in range(0, len(pdf_files), self.config.batch_size):
                batch = pdf_files[i:i+self.config.batch_size]
                results = self.process_batch(batch)
                batch_results.update(results)
                pbar.update(len(batch))
                
                # Update stats
                self.stats.pdfs_with_metadata = sum(
                    1 for r in batch_results.values() 
                    if r != "orphaned"
                )
                
        # Final report
        self.print_final_report()
        
    def print_final_report(self):
        """Print final processing report"""
        stats = self.stats.get_summary()
        
        print("\n" + "=" * 80)
        print("PDF Processing with Late Chunking Complete")
        print("=" * 80)
        print(f"Total PDFs found: {stats['pdfs_found']}")
        print(f"PDFs with metadata: {stats['pdfs_with_metadata']}")
        print(f"  - Already processed: {stats['pdfs_already_processed']}")
        print(f"  - Newly processed: {stats['pdfs_newly_processed']}")
        print(f"  - Failed: {stats['pdfs_failed']}")
        print(f"PDFs without metadata (orphaned): {stats['pdfs_orphaned']}")
        print(f"PDFs deleted: {stats['pdfs_deleted']}")
        print(f"Total chunks created: {stats['total_chunks']}")
        print(f"Processing rate: {stats['processing_rate']:.2f} PDFs/sec")
        print(f"Total time: {stats['elapsed_time']/60:.1f} minutes")
        
        if stats['orphaned_files']:
            print(f"\nOrphaned PDFs (first 10 of {stats['pdfs_orphaned']}):")
            for pdf in stats['orphaned_files']:
                print(f"  - {pdf}")
                
        if stats['failed_files']:
            print(f"\nFailed PDFs (first 10 of {stats['pdfs_failed']}):")
            for pdf in stats['failed_files']:
                print(f"  - {pdf}")
                
        print("=" * 80)
        
        if stats['pdfs_orphaned'] > 0:
            print(f"\n⚠️  {stats['pdfs_orphaned']} PDFs remain in directory (no metadata)")
            print("These files were NOT deleted and need metadata before processing")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Process PDFs with late chunking")
    parser.add_argument('--pdf-directory', type=str, default='/mnt/data-cold/arxiv_data/pdf')
    parser.add_argument('--max-pdfs', type=int, help='Maximum PDFs to process')
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--db-name', type=str, default='base')
    parser.add_argument('--db-host', type=str, default='192.168.1.69')
    parser.add_argument('--docling-gpu', type=int, default=0)
    parser.add_argument('--embedding-gpu', type=int, default=1)
    parser.add_argument('--chunk-size', type=int, default=512, help='Chunk size in tokens')
    parser.add_argument('--chunk-stride', type=int, default=256, help='Chunk stride in tokens')
    parser.add_argument('--dry-run', action='store_true', help='Check files without processing')
    
    args = parser.parse_args()
    
    if not os.environ.get('ARANGO_PASSWORD'):
        logger.error("ARANGO_PASSWORD environment variable not set")
        sys.exit(1)
        
    config = LateChunkingConfig(
        pdf_directory=args.pdf_directory,
        max_pdfs=args.max_pdfs,
        batch_size=args.batch_size,
        db_name=args.db_name,
        db_host=args.db_host,
        docling_gpu=args.docling_gpu,
        embedding_gpu=args.embedding_gpu,
        chunk_size_tokens=args.chunk_size,
        chunk_stride_tokens=args.chunk_stride,
        dry_run=args.dry_run
    )
    
    pipeline = DirectoryPDFPipeline(config)
    pipeline.run()

if __name__ == '__main__':
    main()