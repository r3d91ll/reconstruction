#!/usr/bin/env python3
"""
Unified Document Processing Pipeline with Jina v4

This implements the new architecture where:
1. Docling extracts markdown from PDFs
2. arXiv metadata is loaded and combined with content
3. Enriched document is sent to Jina v4 for processing
4. Results are stored with proper separation of document/chunk metadata
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from arango import ArangoClient

# Import our infrastructure components
from irec_infrastructure.embeddings.local_jina_gpu import create_local_jina_processor
from irec_infrastructure.models.metadata import (
    ArxivMetadata,
    DocumentRecord,
    ChunkMetadata,
    ChunkRecord,
    EnrichedDocument
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('process_documents_unified_atomic.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class UnifiedDocumentProcessor:
    """
    Unified processor that creates enriched documents for Jina v4.
    
    Key features:
    - Combines metadata with content before embedding
    - Uses Jina v4 for better context understanding
    - Separates document and chunk metadata properly
    """
    
    def __init__(self, metadata_dir: Path):
        """Initialize with metadata directory and processors."""
        self.metadata_dir = metadata_dir
        
        logger.info("Initializing Unified Document Processor with Jina v4...")
        
        # Local GPU Jina v4 processor
        self.jina_processor = create_local_jina_processor()
        
        # Docling for text extraction
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(use_ocr=False)
            }
        )
        
        logger.info("Unified processor ready with Jina v4!")
        
    def load_metadata(self, pdf_path: Path) -> Optional[ArxivMetadata]:
        """Load and parse arXiv metadata for a document."""
        metadata_file = self.metadata_dir / f"{pdf_path.stem}.json"
        
        if not metadata_file.exists():
            logger.warning(f"No metadata found for {pdf_path.name}")
            return None
            
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
                
            # Parse dates properly
            if 'published' in data and data['published']:
                try:
                    data['published'] = datetime.fromisoformat(
                        data['published'].replace('Z', '+00:00')
                    )
                except ValueError as e:
                    logger.warning(f"Failed to parse published date: {e}")
                    data['published'] = None
            if 'updated' in data and data['updated']:
                try:
                    data['updated'] = datetime.fromisoformat(
                        data['updated'].replace('Z', '+00:00')
                    )
                except ValueError as e:
                    logger.warning(f"Failed to parse updated date: {e}")
                    data['updated'] = None
                
            return ArxivMetadata(**data)
            
        except Exception as e:
            logger.error(f"Error loading metadata for {pdf_path.name}: {e}")
            return None
            
    def create_enriched_document(self, pdf_path: Path) -> Optional[EnrichedDocument]:
        """Create enriched document by combining metadata and content."""
        try:
            # Load metadata
            metadata = self.load_metadata(pdf_path)
            if not metadata:
                # Create minimal metadata if not found
                metadata = ArxivMetadata(
                    arxiv_id=pdf_path.stem,
                    title="Unknown Title",
                    authors=["Unknown Author"],
                    abstract="No abstract available",
                    categories=[]
                )
                
            # Extract content with Docling
            logger.info(f"Extracting content from {pdf_path.name}")
            doc_result = self.converter.convert(pdf_path)
            
            if not doc_result or not hasattr(doc_result, 'document'):
                logger.error(f"Failed to extract content from {pdf_path.name}")
                return None
                
            # Get markdown content
            content = doc_result.document.export_to_markdown()
            
            if not content or len(content.strip()) < 100:
                logger.error(f"Insufficient content extracted from {pdf_path.name}")
                return None
                
            # Create enriched document
            enriched_doc = EnrichedDocument(
                arxiv_id=metadata.arxiv_id,
                metadata=metadata,
                content=content
            )
            
            # Generate enriched text
            enriched_doc.create_enriched_text()
            
            return enriched_doc
            
        except Exception as e:
            logger.error(f"Error creating enriched document for {pdf_path}: {e}")
            return None
            
    def process_enriched_document(self, enriched_doc: EnrichedDocument) -> Dict:
        """
        Process enriched document with Jina v4.
        
        Returns structured result with chunks and metadata.
        """
        try:
            # Process with Jina v4 using enriched text
            logger.info(f"Processing {enriched_doc.arxiv_id} with Jina v4 "
                       f"({len(enriched_doc.enriched_text)} chars)")
            
            result = self.jina_processor.encode_with_late_chunking(
                enriched_doc.enriched_text,
                return_chunks=True
            )
            
            # Format chunks with proper structure
            chunks = []
            for i, (chunk_text, embedding) in enumerate(zip(
                result['chunks'], 
                result['embeddings']
            )):
                chunk_id = f"{enriched_doc.arxiv_id}_chunk_{i}"
                
                # Extract chunk metadata (this could be enhanced with better detection)
                chunk_metadata = self._extract_chunk_metadata(chunk_text, i)
                
                chunks.append({
                    'chunk_id': chunk_id,
                    'chunk_index': i,
                    'text': chunk_text,
                    'embedding': embedding,
                    'tokens': len(chunk_text.split()),  # Simple estimate
                    'chunk_metadata': chunk_metadata
                })
                
            return {
                'success': True,
                'arxiv_id': enriched_doc.arxiv_id,
                'document_metadata': enriched_doc.metadata.model_dump(),
                'chunks': chunks,
                'processing_info': {
                    'num_chunks': len(chunks),
                    'total_tokens': sum(c['tokens'] for c in chunks),
                    'model_version': 'jina-embeddings-v4',
                    'processed_at': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing enriched document: {e}")
            return {
                'success': False,
                'error': str(e),
                'arxiv_id': enriched_doc.arxiv_id
            }
            
    def _extract_chunk_metadata(self, chunk_text: str, chunk_index: int) -> Dict:
        """
        Extract chunk-specific metadata.
        
        This is a simple implementation - could be enhanced with better detection.
        """
        metadata = {
            'section': None,
            'has_equations': False,
            'has_code': False,
            'has_figures': False,
            'has_tables': False,
            'chunk_type': 'text'
        }
        
        # Simple heuristics - can be improved
        if '\\begin{equation}' in chunk_text or '$' in chunk_text:
            metadata['has_equations'] = True
            
        if '```' in chunk_text or 'def ' in chunk_text or 'class ' in chunk_text:
            metadata['has_code'] = True
            metadata['chunk_type'] = 'code' if '```' in chunk_text else 'mixed'
            
        if 'Figure' in chunk_text or 'figure' in chunk_text:
            metadata['has_figures'] = True
            
        if 'Table' in chunk_text or 'table' in chunk_text:
            metadata['has_tables'] = True
            
        # Try to detect section headers
        lines = chunk_text.split('\n')
        for line in lines[:5]:  # Check first few lines
            if line.strip() and len(line) < 100:
                if line.startswith('#') or line.isupper():
                    metadata['section'] = line.strip('#').strip()
                    break
                    
        return metadata


class UnifiedPipeline:
    """Pipeline using the unified document processing approach."""
    
    def __init__(self, db_host: str, db_name: str, metadata_dir: Optional[Path] = None):
        """Initialize pipeline with database connection."""
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
        self.processor = UnifiedDocumentProcessor(self.metadata_dir)
        
        logger.info(f"Initialized unified pipeline")
        logger.info(f"Database: {self.db_name} at {self.db_host}")
        
    def setup_database(self, clean_start=False):
        """Setup database with separate documents and chunks collections."""
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
        
        # Create collections with proper schema
        collections = {
            'metadata': [  # Document metadata and chunk inventory
                {'type': 'hash', 'fields': ['arxiv_id'], 'unique': True},
                {'type': 'persistent', 'fields': ['categories[*]']},
                {'type': 'persistent', 'fields': ['published']},
                {'type': 'fulltext', 'fields': ['title']},
                {'type': 'fulltext', 'fields': ['abstract']}
            ],
            'chunks': [  # Chunks with only chunk-specific metadata
                {'type': 'hash', 'fields': ['chunk_id'], 'unique': True},
                {'type': 'hash', 'fields': ['arxiv_id']},  # Foreign key
                {'type': 'persistent', 'fields': ['chunk_index']},
                {'type': 'persistent', 'fields': ['chunk_metadata.section']},
                {'type': 'persistent', 'fields': ['chunk_metadata.chunk_type']},
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
                    
        logger.info("Database schema created with separated collections")
        
    def process_and_store_document(self, pdf_path: Path) -> Dict:
        """Process document and store with proper separation."""
        try:
            # Create enriched document
            enriched_doc = self.processor.create_enriched_document(pdf_path)
            if not enriched_doc:
                return {'success': False, 'error': 'Failed to create enriched document'}
                
            # Process with Jina
            result = self.processor.process_enriched_document(enriched_doc)
            if not result['success']:
                return result
                
            # Prepare metadata record (no chunk_ids list, just num_chunks)
            metadata_record = {
                '_key': result['arxiv_id'],
                'arxiv_id': result['arxiv_id'],
                **result['document_metadata'],  # Spread all metadata fields
                'num_chunks': len(result['chunks']),
                'processing_info': result['processing_info']
            }
            
            # Convert datetime fields to ISO format
            metadata_dict = json.loads(json.dumps(metadata_record, default=str))
            
            # Prepare chunk records
            chunk_records = []
            for chunk in result['chunks']:
                chunk_record = {
                    '_key': chunk['chunk_id'],
                    'chunk_id': chunk['chunk_id'],
                    'arxiv_id': result['arxiv_id'],
                    'chunk_index': chunk['chunk_index'],
                    'text': chunk['text'],
                    'embedding': chunk['embedding'],
                    'chunk_metadata': chunk['chunk_metadata'],
                    'tokens': chunk['tokens']
                }
                chunk_records.append(chunk_record)
            
            # ATOMIC TRANSACTION - All or nothing
            transaction = self.db.begin_transaction(
                write_collections=['metadata', 'chunks'],
                read_collections=[]
            )
            
            try:
                # Check if document exists and delete old chunks if reprocessing
                existing = transaction.collection('metadata').get(result['arxiv_id'])
                if existing:
                    logger.info(f"Reprocessing {result['arxiv_id']}, removing old chunks")
                    # Delete all old chunks for this document
                    transaction.aql.execute(
                        'FOR c IN chunks FILTER c.arxiv_id == @arxiv_id REMOVE c IN chunks',
                        bind_vars={'arxiv_id': result['arxiv_id']}
                    )
                
                # Insert metadata
                transaction.collection('metadata').insert(metadata_dict, overwrite=True)
                
                # Insert all chunks
                for chunk_dict in chunk_records:
                    transaction.collection('chunks').insert(chunk_dict)
                
                # Commit atomically
                transaction.commit()
                
                logger.info(f"Atomically stored {result['arxiv_id']}: "
                          f"1 metadata record, {len(chunk_records)} chunks")
                
                return {
                    'success': True,
                    'document_id': result['arxiv_id'],
                    'chunks_stored': len(chunk_records),
                    'total_chunks': len(result['chunks'])
                }
                
            except Exception as e:
                # Rollback - nothing was stored
                transaction.abort()
                logger.error(f"Transaction failed for {result['arxiv_id']}: {e}")
                return {
                    'success': False,
                    'error': f'Transaction failed: {str(e)}',
                    'document_id': result['arxiv_id']
                }
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return {'success': False, 'error': str(e)}
            
    def reconstruct_document(self, arxiv_id: str) -> Optional[Dict]:
        """
        Reconstruct a document from metadata and chunks.
        
        Uses the efficient chunk ID generation (no need to store all IDs).
        """
        try:
            # Get metadata
            metadata = self.db.collection('metadata').get(arxiv_id)
            if not metadata:
                logger.warning(f"No metadata found for {arxiv_id}")
                return None
                
            # Get all chunks using predictable naming
            chunks = []
            missing_chunks = []
            
            for i in range(metadata['num_chunks']):
                chunk_id = f"{arxiv_id}_chunk_{i}"
                chunk = self.db.collection('chunks').get(chunk_id)
                
                if chunk:
                    chunks.append(chunk)
                else:
                    missing_chunks.append(i)
                    
            # Verify integrity
            if missing_chunks:
                logger.warning(
                    f"Missing chunks for {arxiv_id}: "
                    f"expected {metadata['num_chunks']}, "
                    f"missing indices: {missing_chunks}"
                )
                
            # Sort chunks by index to ensure correct order
            chunks.sort(key=lambda x: x['chunk_index'])
            
            # Reconstruct full text
            full_text = '\n'.join(chunk['text'] for chunk in chunks)
            
            return {
                'arxiv_id': arxiv_id,
                'metadata': metadata,
                'chunks': chunks,
                'num_chunks': len(chunks),
                'expected_chunks': metadata['num_chunks'],
                'full_text': full_text,
                'complete': len(chunks) == metadata['num_chunks']
            }
            
        except Exception as e:
            logger.error(f"Error reconstructing document {arxiv_id}: {e}")
            return None
            
    def verify_document_integrity(self, arxiv_id: str) -> Dict[str, Any]:
        """Verify that a document has all expected chunks."""
        metadata = self.db.collection('metadata').get(arxiv_id)
        if not metadata:
            return {'exists': False, 'error': 'No metadata found'}
            
        # Count actual chunks
        cursor = self.db.aql.execute(
            'FOR c IN chunks FILTER c.arxiv_id == @arxiv_id RETURN c.chunk_id',
            bind_vars={'arxiv_id': arxiv_id}
        )
        actual_chunks = set(cursor)
        
        # Generate expected chunk IDs
        expected_chunks = set(
            f"{arxiv_id}_chunk_{i}" for i in range(metadata['num_chunks'])
        )
        
        # Find missing chunks
        missing = expected_chunks - actual_chunks
        extra = actual_chunks - expected_chunks
        
        return {
            'exists': True,
            'complete': len(missing) == 0 and len(extra) == 0,
            'expected': metadata['num_chunks'],
            'actual': len(actual_chunks),
            'missing': list(missing),
            'extra': list(extra)
        }
            
    def verify_database(self):
        """Verify the unified database structure."""
        print("\nUnified Pipeline Database Verification:")
        print("="*60)
        
        # Collection statistics
        metadata_count = self.db.collection('metadata').count()
        chunk_count = self.db.collection('chunks').count()
        
        print(f"Metadata records: {metadata_count}")
        print(f"Chunks: {chunk_count}")
        
        if metadata_count > 0:
            # Calculate average chunks per document
            cursor = self.db.aql.execute(
                'FOR m IN metadata RETURN m.num_chunks'
            )
            chunk_counts = list(cursor)
            avg_chunks = sum(chunk_counts) / len(chunk_counts)
            expected_total = sum(chunk_counts)
            
            print(f"Average chunks per document: {avg_chunks:.1f}")
            print(f"Expected total chunks: {expected_total}")
            print(f"Actual total chunks: {chunk_count}")
            print(f"Integrity: {'✓' if expected_total == chunk_count else '✗'}")
            
            # Sample metadata record
            cursor = self.db.aql.execute("""
                FOR m IN metadata
                    LIMIT 1
                    RETURN {
                        arxiv_id: m.arxiv_id,
                        title: m.title,
                        authors_count: LENGTH(m.authors),
                        categories: m.categories,
                        num_chunks: m.num_chunks
                    }
            """)
            
            sample = list(cursor)[0]
            print(f"\nSample metadata record:")
            print(f"  ID: {sample['arxiv_id']}")
            print(f"  Title: {sample['title'][:80]}...")
            print(f"  Authors: {sample['authors_count']}")
            print(f"  Categories: {', '.join(sample['categories'])}")
            print(f"  Chunks: {sample['num_chunks']}")
            
            # Verify sample document integrity
            integrity = self.verify_document_integrity(sample['arxiv_id'])
            print(f"  Integrity check: {'✓' if integrity['complete'] else '✗'}")
            
        # Chunk type distribution
        cursor = self.db.aql.execute("""
            FOR c IN chunks
                COLLECT type = c.chunk_metadata.chunk_type WITH COUNT INTO count
                RETURN {type: type, count: count}
        """)
        
        print("\nChunk type distribution:")
        for item in cursor:
            print(f"  {item['type']}: {item['count']}")


def main():
    """Run the unified pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process arXiv documents with unified metadata + content approach"
    )
    parser.add_argument('--count', type=int, help='Number of documents to process')
    parser.add_argument('--source-dir', type=str, default='/mnt/data/arxiv_data/pdf',
                       help='Directory containing PDF files')
    parser.add_argument('--db-name', type=str, default='irec_unified',
                       help='Database name')
    parser.add_argument('--db-host', type=str, default='192.168.1.69',
                       help='Database host/IP')
    parser.add_argument('--clean-start', action='store_true',
                       help='Drop existing database and start fresh')
    
    args = parser.parse_args()
    
    print("\nINFORMATION RECONSTRUCTIONISM - Unified Pipeline")
    print("="*60)
    print(f"Using Jina v4 with enriched documents")
    print(f"Source: {args.source_dir}")
    print(f"Count: {args.count if args.count else 'ALL'}")
    print(f"Database: {args.db_name} @ {args.db_host}")
    print("="*60)
    
    # Initialize pipeline
    pipeline = UnifiedPipeline(
        db_host=args.db_host,
        db_name=args.db_name,
        metadata_dir=Path("/mnt/data/arxiv_data/metadata")
    )
    
    # Setup database
    pipeline.setup_database(clean_start=args.clean_start)
    
    # Get PDFs to process
    pdf_dir = Path(args.source_dir)
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if args.count:
        pdf_files = pdf_files[:args.count]
        
    print(f"\nProcessing {len(pdf_files)} documents...")
    
    # Process documents
    stats = {
        'total': len(pdf_files),
        'successful': 0,
        'failed': 0,
        'total_chunks': 0
    }
    
    for pdf_path in tqdm(pdf_files, desc="Processing documents"):
        result = pipeline.process_and_store_document(pdf_path)
        
        if result['success']:
            stats['successful'] += 1
            stats['total_chunks'] += result['chunks_stored']
        else:
            stats['failed'] += 1
            logger.error(f"Failed: {pdf_path.name} - {result.get('error')}")
            
    # Print results
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Total processed: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Total chunks: {stats['total_chunks']}")
    if stats['successful'] > 0:
        print(f"Average chunks per document: {stats['total_chunks']/stats['successful']:.1f}")
    else:
        print("No documents were successfully processed")
    
    # Verify database
    pipeline.verify_database()
    
    # Test reconstruction
    if stats['successful'] > 0:
        print("\nTesting document reconstruction...")
        # Get a sample document
        cursor = pipeline.db.aql.execute("FOR m IN metadata LIMIT 1 RETURN m.arxiv_id")
        docs = list(cursor)
        if docs:
            sample_id = docs[0]
            
            reconstructed = pipeline.reconstruct_document(sample_id)
            if reconstructed:
                print(f"Successfully reconstructed document {sample_id}")
                print(f"  Title: {reconstructed['metadata']['title'][:80]}...")
                print(f"  Chunks: {reconstructed['num_chunks']}/{reconstructed['expected_chunks']}")
                print(f"  Text length: {len(reconstructed['full_text'])} chars")
                print(f"  Complete: {'✓' if reconstructed['complete'] else '✗'}")
    
    print("\n✅ Unified pipeline completed successfully!")


if __name__ == "__main__":
    main()