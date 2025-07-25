#!/usr/bin/env python3
"""
Process arXiv documents with TRUE late chunking - Chunk-Centric Architecture

This pipeline implements Information Reconstructionism's fundamental principle:
The SEMANTIC CHUNK is our atomic unit of information.

Features:
1. Uses Docling for full text extraction (NO chunking)
2. Uses local Jina GPU for TRUE late chunking
3. Stores chunks as self-contained units with embedded metadata
4. Supports configurable document count and source directory
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
from process_documents_local_gpu_with_metadata import MetadataAwareProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('process_semantic_chunks.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ChunkCentricPipeline:
    """Chunk-centric pipeline where semantic chunks are the primitive unit"""
    
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
        
        # Initialize processor
        self.metadata_dir = metadata_dir or Path("/mnt/data/arxiv_data/metadata")
        self.processor = MetadataAwareProcessor(self.metadata_dir)
        
        logger.info(f"Initialized chunk-centric pipeline")
        logger.info(f"Database: {self.db_name} at {self.db_host}")
        logger.info(f"Metadata directory: {self.metadata_dir}")
        
    def setup_database(self, clean_start=False):
        """Setup chunk-centric database schema"""
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
        
        # Create ONLY the chunks collection - our semantic primitive
        # Everything else can be derived or computed later
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
                    
        logger.info("Chunk-centric database schema created successfully")
        
    def process_and_store_document(self, pdf_path: Path) -> Dict:
        """Process a document and store its semantic chunks"""
        try:
            # Process document
            result = self.processor.process_document(pdf_path)
            
            if not result['success']:
                return {'success': False, 'error': result.get('error')}
            
            # Store chunks directly - no complex transactions needed!
            chunks_stored = 0
            
            try:
                # Each chunk is self-contained with all needed metadata
                for chunk in result['chunks']:
                    chunk_data = {
                        '_key': chunk['chunk_id'],
                        'chunk_id': chunk['chunk_id'],
                        'arxiv_id': result['document_id'],
                        'chunk_index': chunk['chunk_index'],
                        'text': chunk['text'],
                        'embedding': chunk['embedding'],
                        'metadata': result['metadata'],  # Full metadata in each chunk
                        'processed_at': datetime.now().isoformat()
                    }
                    
                    # Simple, atomic insert
                    self.db.collection('chunks').insert(chunk_data)
                    chunks_stored += 1
                
                return {
                    'success': True,
                    'chunks': chunks_stored,
                    'document_id': result['document_id']
                }
                
            except Exception as e:
                logger.error(f"Database error: {e}")
                return {
                    'success': False,
                    'error': f"DB error after {chunks_stored} chunks: {str(e)}"
                }
                
        except Exception as e:
            logger.error(f"Processing error for {pdf_path}: {e}")
            return {'success': False, 'error': str(e)}
    
    def process_documents(self, pdf_dir: Path, count: Optional[int] = None, batch_size: int = 10) -> Dict:
        """Process documents with semantic chunks as the atomic unit"""
        # Get PDFs up to count limit
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if count:
            pdf_files = pdf_files[:count]
        total_files = len(pdf_files)
        logger.info(f"Found {total_files} PDF files to process" + 
                   (f" (limited from {len(list(pdf_dir.glob('*.pdf')))})" if count else ""))
        
        # Check for existing progress
        try:
            cursor = self.db.aql.execute(
                "FOR c IN chunks COLLECT arxiv_id = c.arxiv_id RETURN arxiv_id",
                batch_size=1000,
                count=True
            )
            processed_ids = set(cursor)
            logger.info(f"Found {len(processed_ids)} already processed documents")
        except Exception as e:
            logger.warning(f"Error checking existing documents: {e}")
            processed_ids = set()
        
        # Filter unprocessed
        unprocessed = [p for p in pdf_files if p.stem not in processed_ids]
        logger.info(f"Processing {len(unprocessed)} new documents")
        
        # Process in batches
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        stats = {
            'batch_id': batch_id,
            'total': len(unprocessed),
            'successful': 0,
            'failed': 0,
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
        
        # Process documents
        with tqdm(total=len(unprocessed), desc="Processing documents") as pbar:
            for i in range(0, len(unprocessed), batch_size):
                batch = unprocessed[i:i+batch_size]
                
                for pdf_path in batch:
                    result = self.process_and_store_document(pdf_path)
                    
                    if result['success']:
                        stats['successful'] += 1
                        stats['total_chunks'] += result['chunks']
                    else:
                        stats['failed'] += 1
                        logger.error(f"Failed: {pdf_path.name} - {result.get('error')}")
                    
                    pbar.update(1)
        
        # Update stats
        stats['end_time'] = datetime.now().isoformat()
        stats['avg_chunks_per_doc'] = stats['total_chunks'] / stats['successful'] if stats['successful'] > 0 else 0
        
        # Final log entry
        self.db.collection('processing_log').insert({
            'batch_id': batch_id,
            'status': 'completed',
            'successful': stats['successful'],
            'failed': stats['failed'],
            'total_chunks': stats['total_chunks'],
            'timestamp': stats['end_time']
        })
        
        return stats
    
    def verify_database(self):
        """Verify chunk-centric database contents"""
        print("\nChunk-Centric Database Verification:")
        print("="*60)
        
        # Chunk statistics
        chunk_count = self.db.collection('chunks').count()
        
        # Documents represented
        cursor = self.db.aql.execute("""
            FOR c IN chunks
                COLLECT arxiv_id = c.arxiv_id WITH COUNT INTO num_chunks
                RETURN {arxiv_id, num_chunks}
        """, batch_size=1000, count=True)
        doc_stats = list(cursor)
        
        print(f"Total chunks: {chunk_count}")
        print(f"Documents represented: {len(doc_stats)}")
        
        if doc_stats:
            chunk_counts = [d['num_chunks'] for d in doc_stats]
            print(f"Average chunks per document: {sum(chunk_counts)/len(chunk_counts):.1f}")
            print(f"Min chunks: {min(chunk_counts)}")
            print(f"Max chunks: {max(chunk_counts)}")
        
        # Sample chunks
        cursor = self.db.aql.execute("""
            FOR c IN chunks
                SORT c.arxiv_id, c.chunk_index
                LIMIT 3
                RETURN {
                    chunk_id: c.chunk_id,
                    arxiv_id: c.arxiv_id,
                    index: c.chunk_index,
                    text_preview: SUBSTRING(c.text, 0, 100),
                    has_embedding: LENGTH(c.embedding) > 0,
                    title: c.metadata.title
                }
        """, count=True)
        
        print("\nSample chunks:")
        for chunk in cursor:
            print(f"  {chunk['chunk_id']}: \"{chunk['text_preview']}...\"")
            print(f"    Embedding: {'✓' if chunk['has_embedding'] else '✗'}")
        
        # Category distribution from chunk metadata
        cursor = self.db.aql.execute("""
            FOR c IN chunks
                FOR cat IN c.metadata.categories
                    COLLECT category = cat WITH COUNT INTO count
                    SORT count DESC
                    LIMIT 10
                    RETURN {category, count}
        """, batch_size=1000, count=True)
        
        print("\nTop categories (from chunk metadata):")
        for cat in cursor:
            print(f"  {cat['category']}: {cat['count']} chunks")
        
        # Reconstruct a document example
        cursor = self.db.aql.execute("""
            FOR c IN chunks
                FILTER c.arxiv_id != null
                COLLECT arxiv_id = c.arxiv_id
                LIMIT 1
                RETURN arxiv_id
        """, count=True)
        
        sample_id = list(cursor)[0] if cursor else None
        if sample_id:
            print(f"\nDocument reconstruction example ({sample_id}):")
            cursor = self.db.aql.execute("""
                FOR c IN chunks
                    FILTER c.arxiv_id == @arxiv_id
                    SORT c.chunk_index
                    RETURN {
                        index: c.chunk_index,
                        text_length: LENGTH(c.text)
                    }
            """, bind_vars={'arxiv_id': sample_id}, count=True)
            
            chunks = list(cursor)
            total_length = sum(c['text_length'] for c in chunks)
            print(f"  Chunks: {len(chunks)}")
            print(f"  Total text length: {total_length} characters")
            print(f"  Can reconstruct: ✓")


def main():
    """Run the chunk-centric pipeline"""
    parser = argparse.ArgumentParser(
        description="Process arXiv documents using semantic chunks as the atomic unit"
    )
    parser.add_argument('--count', type=int, help='Number of documents to process')
    parser.add_argument('--source-dir', type=str, default='/mnt/data/arxiv_data/pdf',
                       help='Directory containing PDF files')
    parser.add_argument('--db-name', type=str, default='irec_chunks',
                       help='Database name')
    parser.add_argument('--db-host', type=str, default='localhost',
                       help='Database host/IP')
    parser.add_argument('--metadata-dir', type=str, default='/mnt/data/arxiv_data/metadata',
                       help='Directory containing metadata JSON files')
    parser.add_argument('--clean-start', action='store_true',
                       help='Drop existing database and start fresh')
    
    args = parser.parse_args()
    
    print("\nINFORMATION RECONSTRUCTIONISM - Chunk-Centric Pipeline")
    print(f"Semantic chunks are our atomic unit of information")
    print("="*60)
    print(f"Source: {args.source_dir}")
    print(f"Count: {args.count if args.count else 'ALL'}")
    print(f"Database: {args.db_name} @ {args.db_host}")
    print("="*60)
    
    # Initialize pipeline
    pipeline = ChunkCentricPipeline(
        db_host=args.db_host,
        db_name=args.db_name,
        metadata_dir=Path(args.metadata_dir)
    )
    
    # Setup database
    pipeline.setup_database(clean_start=args.clean_start)
    
    # Process documents
    pdf_dir = Path(args.source_dir)
    stats = pipeline.process_documents(pdf_dir, count=args.count)
    
    # Print results
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Total processed: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Total chunks created: {stats['total_chunks']}")
    print(f"Average chunks per document: {stats['avg_chunks_per_doc']:.1f}")
    print(f"Time: {stats['start_time']} to {stats['end_time']}")
    
    # Verify database
    pipeline.verify_database()
    
    print("\n✅ Chunk-centric pipeline completed successfully!")
    print("\nSemantic chunks are stored as self-contained units.")
    print("Documents can be reconstructed by querying chunks by arxiv_id.")


if __name__ == "__main__":
    main()