#!/usr/bin/env python3
"""
Directory-based PDF Processing Pipeline
Processes PDFs from /mnt/data-cold/arxiv_data/pdf directory
- Checks for corresponding metadata in database
- Processes PDFs with metadata that aren't already processed
- Deletes processed PDFs
- Leaves orphaned PDFs (no metadata) in place
"""

import os
import sys
import json
import time
import shutil
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
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
        logging.FileHandler('pdf_directory_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DirectoryPDFConfig:
    """Configuration for directory-based PDF processing"""
    # Directories
    pdf_directory: str = "/mnt/data-cold/arxiv_data/pdf"
    working_dir: str = "/tmp/arxiv_pdf_processing"
    
    # Database
    db_name: str = "arxiv_single_collection"
    collection_name: str = "arxiv_documents"
    db_host: str = "192.168.1.69"
    db_port: int = 8529
    db_username: str = "root"
    db_password: str = os.environ.get('ARANGO_PASSWORD', '')
    
    # Processing
    batch_size: int = 10  # PDFs to check at once
    max_workers: int = 4  # Parallel workers
    
    # Chunking
    chunk_size: int = 2048
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    
    # GPUs
    docling_gpu: int = 0
    embedding_gpu: int = 1
    
    # Options
    dry_run: bool = False
    max_pdfs: Optional[int] = None

class ProcessingStats:
    """Track processing statistics"""
    
    def __init__(self):
        self.start_time = time.time()
        self.pdfs_found = 0
        self.pdfs_with_metadata = 0
        self.pdfs_already_processed = 0
        self.pdfs_newly_processed = 0
        self.pdfs_failed = 0
        self.pdfs_orphaned = 0  # No metadata
        self.pdfs_deleted = 0
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
            'processing_rate': self.pdfs_newly_processed / elapsed if elapsed > 0 else 0,
            'orphaned_files': self.orphaned_list[:10],  # First 10
            'failed_files': self.failed_list[:10]
        }

class DatabaseChecker:
    """Check database for metadata and processing status"""
    
    def __init__(self, config: DirectoryPDFConfig):
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
        
    def check_document_status(self, arxiv_id: str) -> Tuple[bool, bool]:
        """
        Check if document exists and if PDF is already processed
        Returns: (has_metadata, is_processed)
        """
        try:
            doc = self.collection.get(arxiv_id)
            if not doc:
                return False, False
                
            # Check if PDF content exists
            has_pdf_content = (
                'pdf_content' in doc and 
                doc['pdf_content'] and 
                'markdown' in doc['pdf_content'] and
                doc['pdf_content']['markdown']
            )
            
            return True, has_pdf_content
            
        except Exception as e:
            logger.error(f"Error checking document {arxiv_id}: {e}")
            return False, False
            
    def batch_check_status(self, arxiv_ids: List[str]) -> Dict[str, Tuple[bool, bool]]:
        """Batch check multiple documents"""
        results = {}
        
        # Use AQL for efficient batch lookup
        query = """
        FOR id IN @ids
            LET doc = DOCUMENT(@@collection, id)
            RETURN {
                id: id,
                exists: doc != null,
                has_pdf: doc != null AND doc.pdf_content != null AND doc.pdf_content.markdown != null
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
            # Fallback to individual checks
            for arxiv_id in arxiv_ids:
                results[arxiv_id] = self.check_document_status(arxiv_id)
                
        return results

class DoclingConverter:
    """Convert PDFs to markdown using Docling"""
    
    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self._initialize_docling()
        
    def _initialize_docling(self):
        """Initialize Docling with GPU support"""
        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            
            pipeline_options = PdfPipelineOptions(
                do_ocr=False,
                do_table_structure=True,
                table_structure_options={
                    "device": f"cuda:{self.gpu_id}"
                }
            )
            
            self.converter = DocumentConverter(
                pipeline_options=pipeline_options,
                pdf_backend="pypdfium2"
            )
            
            logger.info(f"Docling initialized on GPU {self.gpu_id}")
            
        except ImportError:
            logger.error("Docling not installed. Please install with: pip install docling")
            raise
            
    def convert_pdf(self, pdf_path: Path) -> Optional[str]:
        """Convert PDF to markdown"""
        try:
            torch.cuda.set_device(self.gpu_id)
            result = self.converter.convert(str(pdf_path))
            
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
            torch.cuda.empty_cache()

class TextChunker:
    """Chunk text semantically"""
    
    def __init__(self, chunk_size: int = 2048, overlap: int = 200, min_size: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_size = min_size
        
    def chunk_text(self, text: str) -> List[Dict]:
        """Chunk text while preserving semantic boundaries"""
        if not text or len(text) < self.min_size:
            return []
            
        chunks = []
        lines = text.split('\n')
        current_chunk = []
        current_size = 0
        current_section = "Unknown"
        char_offset = 0  # Track running character position
        
        for line_idx, line in enumerate(lines):
            if line.startswith('#'):
                if current_chunk and current_size >= self.min_size:
                    chunk_text = '\n'.join(current_chunk)
                    # Calculate actual positions based on running offset
                    char_start = char_offset
                    char_end = char_offset + len(chunk_text)
                    chunks.append({
                        'text': chunk_text,
                        'metadata': {
                            'section': current_section,
                            'char_start': char_start,
                            'char_end': char_end
                        }
                    })
                    
                    if self.overlap > 0 and len(current_chunk) > 5:
                        overlap_lines = current_chunk[-5:]
                        # Update offset to account for overlap
                        overlap_text = '\n'.join(current_chunk[:-5])
                        if overlap_text:
                            char_offset += len(overlap_text) + 1  # +1 for newline
                        current_chunk = overlap_lines
                        current_size = sum(len(line) + 1 for line in overlap_lines)  # +1 for newlines
                    else:
                        # Update offset to next position
                        char_offset += len(chunk_text) + (1 if line_idx < len(lines) - 1 else 0)
                        current_chunk = []
                        current_size = 0
                        
                current_section = line.strip('#').strip()
                
            current_chunk.append(line)
            current_size += len(line) + 1  # +1 for newline
            
            if current_size >= self.chunk_size:
                chunk_text = '\n'.join(current_chunk)
                # Calculate actual positions based on running offset
                char_start = char_offset
                char_end = char_offset + len(chunk_text)
                chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        'section': current_section,
                        'char_start': char_start,
                        'char_end': char_end
                    }
                })
                
                if self.overlap > 0 and len(current_chunk) > 5:
                    overlap_lines = current_chunk[-5:]
                    # Update offset to account for overlap
                    overlap_text = '\n'.join(current_chunk[:-5])
                    if overlap_text:
                        char_offset += len(overlap_text) + 1  # +1 for newline
                    current_chunk = overlap_lines
                    current_size = sum(len(line) + 1 for line in overlap_lines)  # +1 for newlines
                else:
                    # Update offset to next position
                    char_offset += len(chunk_text) + (1 if line_idx < len(lines) - 1 else 0)
                    current_chunk = []
                    current_size = 0
                    
        if current_chunk and current_size >= self.min_size:
            chunk_text = '\n'.join(current_chunk)
            # Calculate actual positions based on running offset
            char_start = char_offset
            char_end = char_offset + len(chunk_text)
            chunks.append({
                'text': chunk_text,
                'metadata': {
                    'section': current_section,
                    'char_start': char_start,
                    'char_end': char_end
                }
            })
            
        for i, chunk in enumerate(chunks):
            chunk['chunk_id'] = i
            
        return chunks

class JinaEmbedder:
    """Generate embeddings using Jina"""
    
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
        """Generate embeddings for texts"""
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
            
            del outputs, embeddings, inputs
            torch.cuda.empty_cache()
            
            return embeddings_np
            
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.error(f"GPU OOM error on batch of {len(texts)} texts")
            
            if len(texts) > 1:
                mid = len(texts) // 2
                emb1 = self.embed_batch(texts[:mid])
                emb2 = self.embed_batch(texts[mid:])
                return np.vstack([emb1, emb2])
            else:
                raise

class PDFProcessor:
    """Process individual PDFs"""
    
    def __init__(self, config: DirectoryPDFConfig):
        self.config = config
        self.converter = DoclingConverter(config.docling_gpu)
        self.chunker = TextChunker(config.chunk_size, config.chunk_overlap, config.min_chunk_size)
        self.embedder = JinaEmbedder(f'cuda:{config.embedding_gpu}')
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
        
    def process_pdf(self, pdf_path: Path, arxiv_id: str) -> bool:
        """Process a single PDF and update database"""
        try:
            # Convert to markdown
            logger.info(f"Converting {arxiv_id} to markdown...")
            markdown = self.converter.convert_pdf(pdf_path)
            
            if not markdown:
                logger.error(f"Failed to convert {arxiv_id} to markdown")
                return False
                
            # Chunk the text
            chunks = self.chunker.chunk_text(markdown)
            logger.info(f"Created {len(chunks)} chunks for {arxiv_id}")
            
            if not chunks:
                logger.warning(f"No chunks created for {arxiv_id}")
                return False
                
            # Generate embeddings
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.embedder.embed_batch(texts)
            
            # Add embeddings to chunks
            for i, chunk in enumerate(chunks):
                chunk['embedding'] = embeddings[i].tolist()
                
            # Update database
            pdf_content = {
                'markdown': markdown,
                'chunks': chunks,
                'extraction_metadata': {
                    'docling_version': '1.0',
                    'extraction_time': datetime.utcnow().isoformat() + 'Z',
                    'chunk_count': len(chunks),
                    'chunking_strategy': f'semantic_{self.config.chunk_size}',
                    'pdf_file_size': pdf_path.stat().st_size,
                    'markdown_length': len(markdown)
                }
            }
            
            self.collection.update({
                '_key': arxiv_id,
                'pdf_content': pdf_content,
                'pdf_status': {
                    'state': 'completed',
                    'last_updated': datetime.utcnow().isoformat() + 'Z',
                    'processing_time_seconds': None
                }
            })
            
            logger.info(f"Successfully processed and updated {arxiv_id}")
            return True
            
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
                
            return False

class DirectoryPDFPipeline:
    """Main pipeline for processing PDFs from directory"""
    
    def __init__(self, config: DirectoryPDFConfig):
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
        # Handle different naming conventions
        # e.g., "2301.00001.pdf" or "2301_00001.pdf"
        filename = pdf_path.stem
        
        # Replace underscore with dot if present
        if '_' in filename and '.' not in filename:
            filename = filename.replace('_', '.', 1)
            
        return filename
        
    def process_batch(self, pdf_batch: List[Path]) -> Dict[str, str]:
        """Process a batch of PDFs"""
        results = {}
        files_to_delete = []  # Track files for deletion after processing
        
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
                    success = self.processor.process_pdf(pdf_path, arxiv_id)
                    
                    if success:
                        # Mark for deletion after successful processing
                        files_to_delete.append(pdf_path)
                        self.stats.pdfs_newly_processed += 1
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
                # Don't affect processing state - just log the error
                    
        return results
        
    def run(self):
        """Run the pipeline"""
        logger.info("Starting Directory-based PDF Processing Pipeline")
        logger.info(f"PDF Directory: {self.config.pdf_directory}")
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
        print("PDF Directory Processing Complete")
        print("=" * 80)
        print(f"Total PDFs found: {stats['pdfs_found']}")
        print(f"PDFs with metadata: {stats['pdfs_with_metadata']}")
        print(f"  - Already processed: {stats['pdfs_already_processed']}")
        print(f"  - Newly processed: {stats['pdfs_newly_processed']}")
        print(f"  - Failed: {stats['pdfs_failed']}")
        print(f"PDFs without metadata (orphaned): {stats['pdfs_orphaned']}")
        print(f"PDFs deleted: {stats['pdfs_deleted']}")
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
    
    parser = argparse.ArgumentParser(description="Process PDFs from directory")
    parser.add_argument('--pdf-directory', type=str, default='/mnt/data-cold/arxiv_data/pdf')
    parser.add_argument('--max-pdfs', type=int, help='Maximum PDFs to process')
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--db-name', type=str, default='arxiv_single_collection')
    parser.add_argument('--db-host', type=str, default='192.168.1.69')
    parser.add_argument('--docling-gpu', type=int, default=0)
    parser.add_argument('--embedding-gpu', type=int, default=1)
    parser.add_argument('--dry-run', action='store_true', help='Check files without processing')
    parser.add_argument('--workers', type=int, default=4)
    
    args = parser.parse_args()
    
    if not os.environ.get('ARANGO_PASSWORD'):
        logger.error("ARANGO_PASSWORD environment variable not set")
        sys.exit(1)
        
    config = DirectoryPDFConfig(
        pdf_directory=args.pdf_directory,
        max_pdfs=args.max_pdfs,
        batch_size=args.batch_size,
        db_name=args.db_name,
        db_host=args.db_host,
        docling_gpu=args.docling_gpu,
        embedding_gpu=args.embedding_gpu,
        dry_run=args.dry_run,
        max_workers=args.workers
    )
    
    pipeline = DirectoryPDFPipeline(config)
    pipeline.run()

if __name__ == '__main__':
    main()