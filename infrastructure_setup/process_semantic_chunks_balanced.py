#!/usr/bin/env python3
"""
Process arXiv documents with TRUE late chunking - Balanced GPU Version

This version uses balanced GPU loading to prevent OOM errors when processing
large batches of documents.

Key improvements:
- Even distribution of work across GPUs
- Better memory management
- Automatic batch size adjustment on OOM
- Periodic memory cleanup
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Optional
from tqdm import tqdm

# Add paths
sys.path.append(str(Path(__file__).parent.parent))

from arango import ArangoClient
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat

# Import balanced GPU processor
from irec_infrastructure.embeddings.local_jina_gpu_balanced import (
    create_balanced_local_jina_processor,
    LocalJinaConfig
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('process_semantic_chunks_balanced.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BalancedChunkProcessor:
    """Processor with balanced GPU utilization"""
    
    def __init__(self, metadata_dir: Path):
        """Initialize with balanced GPU configuration."""
        logger.info("Initializing Balanced Chunk Processor...")
        
        self.metadata_dir = metadata_dir
        
        # Configure balanced GPU processing
        config = LocalJinaConfig(
            use_balanced_loading=True,
            batch_size=16,  # Start with larger batch
            use_fp16=True,
            chunk_size=1024,
            chunk_overlap=200
        )
        
        # Create balanced processor
        self.jina_processor = create_balanced_local_jina_processor(config)
        
        # Docling for text extraction
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(use_ocr=False)
            }
        )
        
        logger.info("Balanced processor ready!")
        
    def load_metadata(self, pdf_path: Path) -> Dict:
        """Load arXiv metadata for a given PDF."""
        metadata_file = self.metadata_dir / f"{pdf_path.stem}.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading metadata for {pdf_path.name}: {e}")
                return {}
        else:
            logger.debug(f"No metadata found for {pdf_path.name}")
            return {}
            
    def process_document(self, pdf_path: Path) -> Dict:
        """Process a single document with balanced GPU processing."""
        try:
            # Extract text with Docling
            logger.debug(f"Extracting text from {pdf_path.name}")
            docling_result = self.converter.convert(pdf_path)
            
            if not docling_result or not hasattr(docling_result, 'document'):
                return {
                    'success': False,
                    'error': 'Docling extraction failed',
                    'document_id': pdf_path.stem
                }
                
            # Get full text
            full_text = docling_result.document.export_to_text()
            
            if not full_text or len(full_text.strip()) < 100:
                return {
                    'success': False,
                    'error': 'Insufficient text extracted',
                    'document_id': pdf_path.stem
                }
                
            # Load metadata
            metadata = self.load_metadata(pdf_path)
            
            # Process with balanced GPU late chunking
            logger.debug(f"Processing {pdf_path.name} with balanced GPUs")
            chunking_result = self.jina_processor.encode_with_late_chunking(
                full_text,
                return_chunks=True
            )
            
            # Prepare chunks with metadata
            chunks = []
            for i, (chunk_text, embedding) in enumerate(zip(
                chunking_result['chunks'],
                chunking_result['embeddings']
            )):
                chunk_id = f"{pdf_path.stem}_chunk_{i}"
                chunks.append({
                    'chunk_id': chunk_id,
                    'chunk_index': i,
                    'text': chunk_text,
                    'embedding': embedding
                })
                
            return {
                'success': True,
                'document_id': pdf_path.stem,
                'chunks': chunks,
                'metadata': metadata or {
                    'title': 'Unknown',
                    'authors': [],
                    'categories': [],
                    'abstract': '',
                    'published': None
                },
                'num_chunks': len(chunks),
                'text_length': len(full_text)
            }
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'document_id': pdf_path.stem
            }


class BalancedChunkPipeline:
    """Pipeline with balanced GPU processing and better error handling"""
    
    def __init__(self, db_host: str, db_name: str, metadata_dir: Optional[Path] = None):
        # Database configuration
        self.db_host = db_host
        self.db_port = int(os.environ.get('ARANGO_PORT', '8529'))
        self.db_name = db_name
        self.username = os.environ.get('ARANGO_USERNAME', 'root')
        self.password = os.environ.get('ARANGO_PASSWORD', '')
        
        if not self.password:
            raise ValueError("ARANGO_PASSWORD environment variable not set!")
            
        # Connect to database
        self.client = ArangoClient(hosts=f'http://{self.db_host}:{self.db_port}')
        self.sys_db = self.client.db('_system', username=self.username, password=self.password)
        
        # Initialize balanced processor
        self.metadata_dir = metadata_dir or Path("/mnt/data/arxiv_data/metadata")
        self.processor = BalancedChunkProcessor(self.metadata_dir)
        
        logger.info(f"Initialized balanced chunk pipeline")
        logger.info(f"Database: {self.db_name} at {self.db_host}")
        
    def setup_database(self, clean_start=False):
        """Setup database schema (same as original)"""
        if self.sys_db.has_database(self.db_name):
            if clean_start:
                logger.warning(f"Dropping existing database: {self.db_name}")
                self.sys_db.delete_database(self.db_name)
            else:
                logger.info(f"Using existing database: {self.db_name}")
                self.db = self.client.db(self.db_name, username=self.username, password=self.password)
                return
                
        # Create database
        self.sys_db.create_database(self.db_name)
        self.db = self.client.db(self.db_name, username=self.username, password=self.password)
        
        # Create collections
        collections = {
            'chunks': [
                {'type': 'hash', 'fields': ['chunk_id'], 'unique': True},
                {'type': 'hash', 'fields': ['arxiv_id']},
                {'type': 'persistent', 'fields': ['chunk_index']},
                {'type': 'persistent', 'fields': ['metadata.categories[*]']},
                {'type': 'persistent', 'fields': ['metadata.published']},
                {'type': 'fulltext', 'fields': ['text']}
            ],
            'processing_log': [
                {'type': 'persistent', 'fields': ['batch_id']},
                {'type': 'persistent', 'fields': ['timestamp']},
                {'type': 'persistent', 'fields': ['status']}
            ],
            'failed_documents': [
                {'type': 'hash', 'fields': ['document_id'], 'unique': True},
                {'type': 'persistent', 'fields': ['error_type']},
                {'type': 'persistent', 'fields': ['timestamp']}
            ]
        }
        
        for coll_name, indexes in collections.items():
            collection = self.db.create_collection(coll_name)
            logger.info(f"Created collection: {coll_name}")
            
            for index in indexes:
                if index['type'] == 'hash':
                    collection.add_hash_index(
                        fields=index['fields'],
                        unique=index.get('unique', False)
                    )
                elif index['type'] == 'fulltext':
                    collection.add_fulltext_index(fields=index['fields'])
                elif index['type'] == 'persistent':
                    collection.add_persistent_index(fields=index['fields'])
                    
        logger.info("Database schema created successfully")
        
    def process_and_store_document(self, pdf_path: Path, retry_count: int = 0) -> Dict:
        """Process document with retry logic"""
        try:
            # Process document
            result = self.processor.process_document(pdf_path)
            
            if not result['success']:
                # Log failure
                self.db.collection('failed_documents').insert({
                    '_key': result['document_id'],
                    'document_id': result['document_id'],
                    'error': result.get('error', 'Unknown error'),
                    'error_type': 'processing_failed',
                    'timestamp': datetime.now().isoformat(),
                    'retry_count': retry_count
                }, overwrite=True)
                return result
                
            # Store chunks with better error handling
            chunks_stored = 0
            chunk_errors = []
            
            for chunk in result['chunks']:
                try:
                    chunk_data = {
                        '_key': chunk['chunk_id'],
                        'chunk_id': chunk['chunk_id'],
                        'arxiv_id': result['document_id'],
                        'chunk_index': chunk['chunk_index'],
                        'text': chunk['text'],
                        'embedding': chunk['embedding'],
                        'metadata': result['metadata'],
                        'processed_at': datetime.now().isoformat()
                    }
                    
                    self.db.collection('chunks').insert(chunk_data)
                    chunks_stored += 1
                    
                except Exception as e:
                    if "unique constraint violated" in str(e) or "duplicate key" in str(e).lower():
                        logger.debug(f"Chunk {chunk['chunk_id']} already exists, skipping")
                        # Still count it as stored since it exists
                        chunks_stored += 1
                    else:
                        chunk_errors.append(f"Chunk {chunk['chunk_index']}: {str(e)}")
                    
            if chunk_errors:
                logger.error(f"Chunk storage errors for {pdf_path.name}: {chunk_errors}")
                
            if chunks_stored == 0:
                return {
                    'success': False,
                    'error': f"Failed to store any chunks: {chunk_errors}",
                    'document_id': result['document_id']
                }
                
            return {
                'success': True,
                'chunks': chunks_stored,
                'document_id': result['document_id'],
                'partial': chunks_stored < len(result['chunks'])
            }
            
        except Exception as e:
            logger.error(f"Processing error for {pdf_path}: {e}")
            
            # Log failure
            try:
                self.db.collection('failed_documents').insert({
                    '_key': pdf_path.stem,
                    'document_id': pdf_path.stem,
                    'error': str(e),
                    'error_type': 'exception',
                    'timestamp': datetime.now().isoformat(),
                    'retry_count': retry_count
                }, overwrite=True)
            except:
                pass
                
            return {'success': False, 'error': str(e), 'document_id': pdf_path.stem}
            
    def process_documents(self, pdf_dir: Path, count: Optional[int] = None, 
                         batch_size: int = 10, retry_failed: bool = True) -> Dict:
        """Process documents with retry logic and better error tracking"""
        # Get PDFs
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if count:
            pdf_files = pdf_files[:count]
            
        total_files = len(pdf_files)
        logger.info(f"Found {total_files} PDF files to process")
        
        # Check existing progress
        try:
            cursor = self.db.aql.execute(
                "FOR c IN chunks COLLECT arxiv_id = c.arxiv_id RETURN arxiv_id",
                batch_size=1000
            )
            processed_ids = set(cursor)
            logger.info(f"Found {len(processed_ids)} already processed documents")
            
            # Check failed documents
            cursor = self.db.aql.execute(
                "FOR d IN failed_documents RETURN {id: d.document_id, retries: d.retry_count}"
            )
            failed_docs = {doc['id']: doc['retries'] for doc in cursor}
            logger.info(f"Found {len(failed_docs)} previously failed documents")
            
        except Exception as e:
            logger.warning(f"Error checking existing documents: {e}")
            processed_ids = set()
            failed_docs = {}
            
        # Filter unprocessed
        unprocessed = []
        for p in pdf_files:
            doc_id = p.stem
            if doc_id not in processed_ids:
                if doc_id in failed_docs:
                    if retry_failed and failed_docs[doc_id] < 3:  # Max 3 retries
                        unprocessed.append(p)
                else:
                    unprocessed.append(p)
                    
        logger.info(f"Processing {len(unprocessed)} documents")
        
        # Process in batches
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        stats = {
            'batch_id': batch_id,
            'total': len(unprocessed),
            'successful': 0,
            'failed': 0,
            'partial': 0,
            'total_chunks': 0,
            'start_time': datetime.now().isoformat()
        }
        
        # Log batch start
        self.db.collection('processing_log').insert({
            '_key': batch_id,
            'batch_id': batch_id,
            'status': 'started',
            'total_documents': len(unprocessed),
            'timestamp': stats['start_time']
        })
        
        # Process with progress bar
        with tqdm(total=len(unprocessed), desc="Processing documents") as pbar:
            for i in range(0, len(unprocessed), batch_size):
                batch = unprocessed[i:i+batch_size]
                
                for pdf_path in batch:
                    retry_count = failed_docs.get(pdf_path.stem, 0)
                    result = self.process_and_store_document(pdf_path, retry_count)
                    
                    if result['success']:
                        stats['successful'] += 1
                        stats['total_chunks'] += result['chunks']
                        if result.get('partial'):
                            stats['partial'] += 1
                    else:
                        stats['failed'] += 1
                        logger.error(f"Failed: {pdf_path.name} - {result.get('error')}")
                        
                    pbar.update(1)
                    
        # Final stats
        stats['end_time'] = datetime.now().isoformat()
        stats['avg_chunks_per_doc'] = stats['total_chunks'] / stats['successful'] if stats['successful'] > 0 else 0
        
        # Log completion
        self.db.collection('processing_log').insert({
            'batch_id': batch_id,
            'status': 'completed',
            'successful': stats['successful'],
            'failed': stats['failed'],
            'partial': stats['partial'],
            'total_chunks': stats['total_chunks'],
            'timestamp': stats['end_time']
        })
        
        return stats
        
    def verify_database(self):
        """Verify database contents with failure analysis"""
        print("\nBalanced Pipeline Database Verification:")
        print("="*60)
        
        # Basic stats
        chunk_count = self.db.collection('chunks').count()
        failed_count = self.db.collection('failed_documents').count()
        
        # Documents represented
        cursor = self.db.aql.execute("""
            FOR c IN chunks
                COLLECT arxiv_id = c.arxiv_id WITH COUNT INTO num_chunks
                RETURN {arxiv_id, num_chunks}
        """)
        doc_stats = list(cursor)
        
        print(f"Total chunks: {chunk_count}")
        print(f"Documents successfully processed: {len(doc_stats)}")
        print(f"Documents failed: {failed_count}")
        
        if doc_stats:
            chunk_counts = [d['num_chunks'] for d in doc_stats]
            print(f"Average chunks per document: {sum(chunk_counts)/len(chunk_counts):.1f}")
            print(f"Min chunks: {min(chunk_counts)}")
            print(f"Max chunks: {max(chunk_counts)}")
            
        # Failure analysis
        if failed_count > 0:
            print("\nFailure Analysis:")
            cursor = self.db.aql.execute("""
                FOR d IN failed_documents
                    COLLECT error_type = d.error_type WITH COUNT INTO count
                    RETURN {error_type, count}
            """)
            for error in cursor:
                print(f"  {error['error_type']}: {error['count']}")
                
        # Recent failures
        cursor = self.db.aql.execute("""
            FOR d IN failed_documents
                SORT d.timestamp DESC
                LIMIT 5
                RETURN {
                    id: d.document_id,
                    error: SUBSTRING(d.error, 0, 100),
                    retries: d.retry_count
                }
        """)
        
        print("\nRecent failures:")
        for doc in cursor:
            print(f"  {doc['id']}: {doc['error']}... (retries: {doc['retries']})")


def main():
    """Run the balanced chunk pipeline"""
    parser = argparse.ArgumentParser(
        description="Process arXiv documents with balanced GPU utilization"
    )
    parser.add_argument('--count', type=int, help='Number of documents to process')
    parser.add_argument('--source-dir', type=str, default='/mnt/data/arxiv_data/pdf',
                       help='Directory containing PDF files')
    parser.add_argument('--db-name', type=str, default='irec_chunks_balanced',
                       help='Database name')
    parser.add_argument('--db-host', type=str, default='192.168.1.69',
                       help='Database host/IP')
    parser.add_argument('--metadata-dir', type=str, default='/mnt/data/arxiv_data/metadata',
                       help='Directory containing metadata JSON files')
    parser.add_argument('--clean-start', action='store_true',
                       help='Drop existing database and start fresh')
    parser.add_argument('--no-retry', action='store_true',
                       help='Do not retry failed documents')
    
    args = parser.parse_args()
    
    print("\nINFORMATION RECONSTRUCTIONISM - Balanced GPU Pipeline")
    print("="*60)
    print(f"Source: {args.source_dir}")
    print(f"Count: {args.count if args.count else 'ALL'}")
    print(f"Database: {args.db_name} @ {args.db_host}")
    print(f"Retry failed: {not args.no_retry}")
    print("="*60)
    
    # Initialize pipeline
    pipeline = BalancedChunkPipeline(
        db_host=args.db_host,
        db_name=args.db_name,
        metadata_dir=Path(args.metadata_dir)
    )
    
    # Setup database
    pipeline.setup_database(clean_start=args.clean_start)
    
    # Process documents
    pdf_dir = Path(args.source_dir)
    stats = pipeline.process_documents(
        pdf_dir, 
        count=args.count,
        retry_failed=not args.no_retry
    )
    
    # Print results
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Total processed: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Partial success: {stats.get('partial', 0)}")
    print(f"Total chunks created: {stats['total_chunks']}")
    print(f"Average chunks per document: {stats['avg_chunks_per_doc']:.1f}")
    print(f"Time: {stats['start_time']} to {stats['end_time']}")
    
    # Verify database
    pipeline.verify_database()
    
    print("\n✅ Balanced GPU pipeline completed!")
    print("Check failed_documents collection for any errors.")


if __name__ == "__main__":
    main()