#!/usr/bin/env python3
"""
Process all 1960 arXiv documents with TRUE late chunking and database storage

This is the final production pipeline that:
1. Uses Docling for full text extraction (NO chunking)
2. Uses local Jina GPU for TRUE late chunking
3. Integrates full arXiv metadata
4. Stores everything in ArangoDB
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List
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
        logging.FileHandler('process_1960_documents.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProductionPipeline:
    """Production pipeline for processing all documents"""
    
    def __init__(self):
        # Database configuration from environment
        self.db_host = os.environ.get('ARANGO_HOST', '192.168.1.69')
        self.db_port = int(os.environ.get('ARANGO_PORT', '8529'))
        self.db_name = os.environ.get('ARANGO_DB_NAME', 'irec_production')
        self.username = os.environ.get('ARANGO_USERNAME', 'root')
        self.password = os.environ.get('ARANGO_PASSWORD', '')
        
        if not self.password:
            raise ValueError("ARANGO_PASSWORD environment variable not set!")
        
        # Connect to database
        self.client = ArangoClient(hosts=f'http://{self.db_host}:{self.db_port}')
        self.sys_db = self.client.db('_system', username=self.username, password=self.password)
        
        # Initialize processor
        self.metadata_dir = Path("/mnt/data/arxiv_data/metadata")
        self.processor = MetadataAwareProcessor(self.metadata_dir)
        
        logger.info(f"Initialized production pipeline with database {self.db_name}")
        
    def setup_database(self, clean_start=False):
        """Setup production database with proper schema"""
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
            'documents': [
                {'type': 'hash', 'fields': ['arxiv_id'], 'unique': True},
                {'type': 'fulltext', 'fields': ['title']},
                {'type': 'fulltext', 'fields': ['abstract']},
                {'type': 'persistent', 'fields': ['categories[*]']},
                {'type': 'persistent', 'fields': ['published']}
            ],
            'chunks': [
                {'type': 'hash', 'fields': ['chunk_id'], 'unique': True},
                {'type': 'hash', 'fields': ['arxiv_id']},
                {'type': 'persistent', 'fields': ['chunk_index']}
            ],
            'metadata': [
                {'type': 'hash', 'fields': ['arxiv_id'], 'unique': True}
            ],
            'processing_log': [
                {'type': 'persistent', 'fields': ['batch_id']},
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
                    
        logger.info("Database schema created successfully")
        
    def process_and_store_document(self, pdf_path: Path) -> Dict:
        """Process a document and store in database"""
        try:
            # Process document
            result = self.processor.process_document(pdf_path)
            
            if not result['success']:
                return {'success': False, 'error': result.get('error')}
            
            # Store in database with transaction support
            # Begin transaction with write access to all required collections
            transaction = self.db.begin_transaction(
                write=['documents', 'chunks', 'metadata']
            )
            
            try:
                # 1. Document
                doc_data = {
                    '_key': result['document_id'],
                    'arxiv_id': result['document_id'],
                    'title': result['metadata']['title'],
                    'authors': result['metadata']['authors'],
                    'abstract': result['metadata']['abstract'],
                    'categories': result['metadata']['categories'],
                    'published': result['metadata']['published'],
                    'updated': result['metadata']['updated'],
                    'full_text': result['full_text'],
                    'text_length': result['processing']['text_length'],
                    'num_chunks': result['num_chunks'],
                    'processing_timestamp': result['processing']['extraction_timestamp']
                }
                transaction.collection('documents').insert(doc_data, overwrite=True)
                
                # 2. Chunks
                chunks_collection = transaction.collection('chunks')
                for chunk in result['chunks']:
                    chunk_data = {
                        '_key': chunk['chunk_id'],
                        'chunk_id': chunk['chunk_id'],
                        'arxiv_id': chunk['arxiv_id'],
                        'chunk_index': chunk['chunk_index'],
                        'text': chunk['text'],
                        'tokens': chunk['tokens'],
                        'embedding': chunk['embedding']
                    }
                    chunks_collection.insert(chunk_data, overwrite=True)
                
                # 3. Metadata
                meta_data = {
                    '_key': result['document_id'],
                    'arxiv_id': result['document_id'],
                    'pdf_url': result['metadata']['pdf_url'],
                    'abs_url': result['metadata']['abs_url'],
                    'pdf_path': result['pdf_path'],
                    'processing_mode': result['processing']['processing_mode']
                }
                transaction.collection('metadata').insert(meta_data, overwrite=True)
                
                # Commit the transaction
                transaction.commit()
                
            except Exception as e:
                # Rollback on any error
                transaction.abort()
                raise e
            
            return {
                'success': True,
                'document_id': result['document_id'],
                'chunks': result['num_chunks']
            }
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return {'success': False, 'error': str(e)}
    
    def process_all_documents(self, pdf_dir: Path, batch_size: int = 100):
        """Process all documents in batches"""
        # Get all PDFs
        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        total_files = len(pdf_files)
        
        logger.info(f"Found {total_files} PDF files to process")
        
        # Check for existing progress
        try:
            cursor = self.db.aql.execute(
                "FOR doc IN documents RETURN doc.arxiv_id"
            )
            processed_ids = set(cursor)
            logger.info(f"Found {len(processed_ids)} already processed documents")
        except:
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
            'start_time': stats['start_time']
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
                    pbar.set_postfix({
                        'success': stats['successful'],
                        'failed': stats['failed'],
                        'chunks': stats['total_chunks']
                    })
                
                # Periodic commit
                logger.info(f"Processed batch {i//batch_size + 1}, committing...")
        
        # Final statistics
        stats['end_time'] = datetime.now().isoformat()
        stats['avg_chunks_per_doc'] = stats['total_chunks'] / stats['successful'] if stats['successful'] > 0 else 0
        
        # Update log
        self.db.collection('processing_log').update({
            '_key': batch_id,
            'status': 'completed',
            'successful': stats['successful'],
            'failed': stats['failed'],
            'total_chunks': stats['total_chunks'],
            'end_time': stats['end_time']
        })
        
        return stats
    
    def verify_database(self):
        """Verify database contents"""
        print("\nDatabase Verification:")
        print("="*60)
        
        # Document count
        doc_count = self.db.collection('documents').count()
        chunk_count = self.db.collection('chunks').count()
        meta_count = self.db.collection('metadata').count()
        
        print(f"Documents: {doc_count}")
        print(f"Chunks: {chunk_count}")
        print(f"Metadata: {meta_count}")
        
        # Sample data
        cursor = self.db.aql.execute("""
            FOR doc IN documents
                LIMIT 5
                LET chunk_count = LENGTH(
                    FOR c IN chunks
                        FILTER c.arxiv_id == doc.arxiv_id
                        RETURN c
                )
                RETURN {
                    id: doc.arxiv_id,
                    title: SUBSTRING(doc.title, 0, 60),
                    chunks: chunk_count,
                    expected: doc.num_chunks
                }
        """)
        
        print("\nSample documents:")
        for doc in cursor:
            match = "✓" if doc['chunks'] == doc['expected'] else "✗"
            print(f"  {match} {doc['id']}: {doc['chunks']} chunks")
        
        # Category distribution
        cursor = self.db.aql.execute("""
            FOR doc IN documents
                FOR cat IN doc.categories
                    COLLECT category = cat WITH COUNT INTO count
                    SORT count DESC
                    LIMIT 10
                    RETURN {category, count}
        """)
        
        print("\nTop categories:")
        for cat in cursor:
            print(f"  {cat['category']}: {cat['count']} documents")


def main():
    """Run the production pipeline"""
    print("\nINFORMATION RECONSTRUCTIONISM - Production Pipeline")
    print("Processing 1960 arXiv documents with TRUE late chunking")
    print("="*60)
    
    # Initialize pipeline
    pipeline = ProductionPipeline()
    
    # Setup database (use --clean-start to reset)
    clean_start = '--clean-start' in sys.argv
    pipeline.setup_database(clean_start=clean_start)
    
    # Process all documents
    pdf_dir = Path("/mnt/data/arxiv_data/pdf")
    stats = pipeline.process_all_documents(pdf_dir)
    
    # Print results
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Total processed: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Total chunks created: {stats['total_chunks']}")
    print(f"Average chunks per document: {stats['avg_chunks_per_doc']:.1f}")
    print(f"Time taken: {stats['start_time']} to {stats['end_time']}")
    
    # Verify database
    pipeline.verify_database()
    
    print("\n✅ Production pipeline completed successfully!")
    print("\nDatabase ready for Information Reconstructionism experiments!")


if __name__ == "__main__":
    main()