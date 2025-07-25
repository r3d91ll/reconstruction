#!/usr/bin/env python3
"""
Three-Collection Document Processing Pipeline

Implements the proper architecture:
1. metadata - Document-level metadata from Docling
2. documents - Full Docling markdown content
3. chunks - Semantic chunks from Jina with chunk-level metadata

Key features:
- Preserves complete Docling JSON structure
- Enables full JSON reconstruction
- Atomic transactions across all three collections
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
from tqdm import tqdm

# Add infrastructure directory to path
infrastructure_dir = Path(__file__).parent.parent
sys.path.append(str(infrastructure_dir))

# Import our infrastructure components
from irec_infrastructure.embeddings.local_jina_gpu import create_local_jina_processor
from irec_infrastructure.models.metadata import ArxivMetadata, ChunkMetadata, EnrichedDocument

# Import from setup directory
sys.path.append(str(Path(__file__).parent))
from base_pipeline import BaseDocumentProcessor, BasePipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('process_documents_three_collections.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ThreeCollectionProcessor(BaseDocumentProcessor):
    """
    Document processor that preserves full Docling output.
    """
    
    def __init__(self, metadata_dir: Path):
        """Initialize with metadata directory and processors."""
        super().__init__(metadata_dir)
        
        logger.info("Initializing Three Collection Processor...")
        
        # Local GPU Jina v4 processor
        self.jina_processor = create_local_jina_processor()
        
        logger.info("Three collection processor ready!")
    
    def process_enriched_document(self, enriched_doc: EnrichedDocument) -> Dict:
        """Required by base class but not used in three-collection approach."""
        # This method is not used in our three-collection pipeline
        # We use process_document_complete instead
        return {'success': False, 'error': 'Use process_document_complete instead'}
    
    def process_document_complete(self, pdf_path: Path) -> Dict:
        """
        Process document and return complete structure for 3-collection storage.
        
        Returns a dictionary with:
        - docling_metadata: Metadata from Docling
        - docling_content: Full markdown from Docling
        - enriched_text: Text prepared for Jina
        - chunks: Semantic chunks with embeddings
        """
        try:
            # Load arXiv metadata
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
                return {'success': False, 'error': 'Docling extraction failed'}
            
            # Get Docling markdown
            docling_markdown = doc_result.document.export_to_markdown()
            
            if not docling_markdown or len(docling_markdown.strip()) < 100:
                logger.error(f"Insufficient content extracted from {pdf_path.name}")
                return {'success': False, 'error': 'Insufficient content'}
            
            # Get Docling metadata (if available)
            docling_metadata = {}
            if hasattr(doc_result.document, 'metadata'):
                # Extract any metadata Docling provides
                doc_meta = doc_result.document.metadata
                if hasattr(doc_meta, 'to_dict'):
                    docling_metadata = doc_meta.to_dict()
                elif hasattr(doc_meta, '__dict__'):
                    docling_metadata = doc_meta.__dict__
            
            # Create enriched document for Jina processing
            enriched_doc = EnrichedDocument(
                arxiv_id=metadata.arxiv_id,
                metadata=metadata,
                content=docling_markdown
            )
            enriched_doc.create_enriched_text()
            
            # Process with Jina v4
            logger.info(f"Processing {metadata.arxiv_id} with Jina v4 "
                       f"({len(enriched_doc.enriched_text)} chars)")
            
            jina_result = self.jina_processor.encode_with_late_chunking(
                enriched_doc.enriched_text,
                return_chunks=True
            )
            
            # Format chunks with metadata
            chunks = []
            for i, (chunk_text, embedding) in enumerate(zip(
                jina_result['chunks'], 
                jina_result['embeddings']
            )):
                chunk_id = f"{metadata.arxiv_id}_chunk_{i}"
                
                # Extract chunk metadata
                chunk_metadata = self.extract_chunk_metadata(chunk_text, i)
                
                chunks.append({
                    'chunk_id': chunk_id,
                    'arxiv_id': metadata.arxiv_id,
                    'chunk_index': i,
                    'text': chunk_text,
                    'embedding': embedding,
                    'tokens': len(chunk_text.split()),
                    'chunk_metadata': chunk_metadata
                })
            
            # Combine all metadata
            combined_metadata = {
                'arxiv_id': metadata.arxiv_id,
                'arxiv_metadata': metadata.model_dump(),
                'docling_metadata': docling_metadata,
                'processing_info': {
                    'num_chunks': len(chunks),
                    'total_tokens': sum(c['tokens'] for c in chunks),
                    'docling_content_length': len(docling_markdown),
                    'model_version': 'jina-embeddings-v4',
                    'processed_at': datetime.now().isoformat()
                }
            }
            
            return {
                'success': True,
                'arxiv_id': metadata.arxiv_id,
                'metadata': combined_metadata,
                'document_content': docling_markdown,
                'chunks': chunks
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return {
                'success': False,
                'error': str(e),
                'arxiv_id': pdf_path.stem
            }


class ThreeCollectionPipeline(BasePipeline):
    """Pipeline with three collections: metadata, documents, chunks."""
    
    def __init__(self, db_host: str, db_name: str, metadata_dir: Optional[Path] = None):
        """Initialize pipeline with database connection."""
        super().__init__(db_host, db_name, metadata_dir)
        
        # Initialize processor
        self.processor = ThreeCollectionProcessor(self.metadata_dir)
        
        logger.info(f"Initialized three-collection pipeline")
    
    def setup_database(self, clean_start=False):
        """Setup database with three collections."""
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
            'metadata': [  # Combined metadata from all sources
                {'type': 'hash', 'fields': ['arxiv_id'], 'unique': True},
                {'type': 'persistent', 'fields': ['arxiv_metadata.categories[*]']},
                {'type': 'persistent', 'fields': ['arxiv_metadata.published']},
                {'type': 'fulltext', 'fields': ['arxiv_metadata.title']},
                {'type': 'fulltext', 'fields': ['arxiv_metadata.abstract']}
            ],
            'documents': [  # Full Docling markdown
                {'type': 'hash', 'fields': ['arxiv_id'], 'unique': True},
                {'type': 'fulltext', 'fields': ['content']}
            ],
            'chunks': [  # Semantic chunks with embeddings
                {'type': 'hash', 'fields': ['chunk_id'], 'unique': True},
                {'type': 'hash', 'fields': ['arxiv_id']},
                {'type': 'persistent', 'fields': ['chunk_index']},
                {'type': 'persistent', 'fields': ['chunk_metadata.section']},
                {'type': 'persistent', 'fields': ['chunk_metadata.chunk_type']},
                {'type': 'fulltext', 'fields': ['text']}
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
        
        logger.info("Database schema created with metadata/documents/chunks architecture")
    
    def process_and_store_document(self, pdf_path: Path) -> Dict:
        """Process document and store in three collections atomically."""
        try:
            # Process document completely
            result = self.processor.process_document_complete(pdf_path)
            if not result['success']:
                return result
            
            # Prepare records for each collection
            metadata_record = {
                '_key': result['arxiv_id'],
                **result['metadata']
            }
            
            document_record = {
                '_key': result['arxiv_id'],
                'arxiv_id': result['arxiv_id'],
                'content': result['document_content']
            }
            
            chunk_records = []
            for chunk in result['chunks']:
                chunk_record = {
                    '_key': chunk['chunk_id'],
                    'chunk_id': chunk['chunk_id'],
                    'arxiv_id': chunk['arxiv_id'],
                    'chunk_index': chunk['chunk_index'],
                    'text': chunk['text'],
                    'embedding': chunk['embedding'],
                    'chunk_metadata': chunk['chunk_metadata'],
                    'tokens': chunk['tokens']
                }
                chunk_records.append(chunk_record)
            
            # Convert datetime fields to ISO format
            metadata_dict = json.loads(json.dumps(metadata_record, default=str))
            
            # ATOMIC TRANSACTION - All or nothing
            db = self.db
            
            # Start transaction
            transaction = db.begin_transaction(
                write=['metadata', 'documents', 'chunks']
            )
            
            try:
                # Check if document exists and delete old data if reprocessing
                existing = transaction.collection('metadata').get(result['arxiv_id'])
                if existing:
                    logger.info(f"Reprocessing {result['arxiv_id']}, removing old data")
                    # Delete from all collections
                    transaction.collection('metadata').delete(result['arxiv_id'])
                    transaction.collection('documents').delete(result['arxiv_id'])
                    # Delete all chunks for this document
                    transaction.aql.execute(
                        'FOR c IN chunks FILTER c.arxiv_id == @arxiv_id REMOVE c IN chunks',
                        bind_vars={'arxiv_id': result['arxiv_id']}
                    )
                
                # Insert into all three collections
                transaction.collection('metadata').insert(metadata_dict)
                transaction.collection('documents').insert(document_record)
                
                for chunk_dict in chunk_records:
                    transaction.collection('chunks').insert(chunk_dict)
                
                # Commit atomically
                transaction.commit_transaction()
                
                logger.info(f"Atomically stored {result['arxiv_id']}: "
                          f"metadata + document + {len(chunk_records)} chunks")
                
                return {
                    'success': True,
                    'document_id': result['arxiv_id'],
                    'chunks_stored': len(chunk_records),
                    'total_chunks': len(result['chunks'])
                }
                
            except Exception as e:
                # Rollback - nothing was stored
                transaction.abort_transaction()
                logger.error(f"Transaction failed for {result['arxiv_id']}: {e}")
                return {
                    'success': False,
                    'error': f'Transaction failed: {str(e)}',
                    'document_id': result['arxiv_id']
                }
                
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return {'success': False, 'error': str(e)}
    
    def reconstruct_json(self, arxiv_id: str) -> Optional[Dict]:
        """
        Reconstruct the complete JSON object from all three collections.
        
        This demonstrates that we can recover the full document structure.
        """
        try:
            # Get metadata
            metadata = self.db.collection('metadata').get(arxiv_id)
            if not metadata:
                logger.warning(f"No metadata found for {arxiv_id}")
                return None
            
            # Get document content
            document = self.db.collection('documents').get(arxiv_id)
            if not document:
                logger.warning(f"No document found for {arxiv_id}")
                return None
            
            # Get all chunks
            chunks = []
            cursor = self.db.aql.execute(
                'FOR c IN chunks FILTER c.arxiv_id == @arxiv_id SORT c.chunk_index RETURN c',
                bind_vars={'arxiv_id': arxiv_id}
            )
            for chunk in cursor:
                chunks.append(chunk)
            
            # Reconstruct complete JSON
            reconstructed = {
                'arxiv_id': arxiv_id,
                'metadata': metadata,
                'document_content': document['content'],
                'chunks': chunks,
                'reconstruction_info': {
                    'num_chunks_found': len(chunks),
                    'expected_chunks': metadata['processing_info']['num_chunks'],
                    'complete': len(chunks) == metadata['processing_info']['num_chunks'],
                    'reconstructed_at': datetime.now().isoformat()
                }
            }
            
            return reconstructed
            
        except Exception as e:
            logger.error(f"Error reconstructing JSON for {arxiv_id}: {e}")
            return None
    
    def verify_database(self):
        """Verify the three-collection database structure."""
        print("\nThree-Collection Database Verification:")
        print("="*60)
        
        # Collection statistics
        metadata_count = self.db.collection('metadata').count()
        document_count = self.db.collection('documents').count()
        chunk_count = self.db.collection('chunks').count()
        
        print(f"Metadata records: {metadata_count}")
        print(f"Document records: {document_count}")
        print(f"Chunks: {chunk_count}")
        
        if metadata_count > 0:
            # Verify consistency
            print(f"\nConsistency check:")
            print(f"Metadata = Documents: {'✓' if metadata_count == document_count else '✗'}")
            
            # Calculate average chunks per document
            cursor = self.db.aql.execute(
                'FOR m IN metadata RETURN m.processing_info.num_chunks'
            )
            chunk_counts = list(cursor)
            avg_chunks = sum(chunk_counts) / len(chunk_counts)
            expected_total = sum(chunk_counts)
            
            print(f"Average chunks per document: {avg_chunks:.1f}")
            print(f"Expected total chunks: {expected_total}")
            print(f"Actual total chunks: {chunk_count}")
            print(f"Chunk integrity: {'✓' if expected_total == chunk_count else '✗'}")
            
            # Sample reconstruction test
            cursor = self.db.aql.execute("FOR m IN metadata LIMIT 1 RETURN m.arxiv_id")
            sample_id = list(cursor)[0]
            
            print(f"\nTesting JSON reconstruction for {sample_id}...")
            reconstructed = self.reconstruct_json(sample_id)
            if reconstructed:
                print(f"✓ Successfully reconstructed complete JSON")
                print(f"  Metadata fields: {len(reconstructed['metadata'])}")
                print(f"  Document length: {len(reconstructed['document_content'])} chars")
                print(f"  Chunks: {reconstructed['reconstruction_info']['num_chunks_found']}")
                print(f"  Complete: {'✓' if reconstructed['reconstruction_info']['complete'] else '✗'}")


def main():
    """Run the three-collection pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process arXiv documents with three-collection architecture"
    )
    parser.add_argument('--count', type=int, help='Number of documents to process')
    parser.add_argument('--source-dir', type=str, default='/mnt/data/arxiv_data/pdf',
                       help='Directory containing PDF files')
    parser.add_argument('--db-name', type=str, default='irec_three_collections',
                       help='Database name')
    parser.add_argument('--db-host', type=str, default='192.168.1.69',
                       help='Database host/IP')
    parser.add_argument('--clean-start', action='store_true',
                       help='Drop existing database and start fresh')
    
    args = parser.parse_args()
    
    print("\nINFORMATION RECONSTRUCTIONISM - Three Collection Pipeline")
    print("="*60)
    print(f"Architecture: metadata + documents + chunks")
    print(f"Source: {args.source_dir}")
    print(f"Count: {args.count if args.count else 'ALL'}")
    print(f"Database: {args.db_name} @ {args.db_host}")
    print("="*60)
    
    # Initialize pipeline
    pipeline = ThreeCollectionPipeline(
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
    
    # Verify database
    pipeline.verify_database()
    
    print("\n✅ Three-collection pipeline completed successfully!")


if __name__ == "__main__":
    main()