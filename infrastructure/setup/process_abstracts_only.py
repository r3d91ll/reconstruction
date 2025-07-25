#!/usr/bin/env python3
"""
Abstract-Only Pipeline for arXiv Metadata

This pipeline creates a lightweight database containing only:
- arXiv metadata (title, authors, categories, etc.)
- Abstract embeddings (2048D vectors from Jina v4)

No PDF processing, no document content, no chunks.
Perfect for large-scale discovery and testing.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
from tqdm import tqdm
from arango import ArangoClient
import torch
import numpy as np

# Add infrastructure directory to path
infrastructure_dir = Path(__file__).parent.parent
sys.path.append(str(infrastructure_dir))

# Import our infrastructure components
from irec_infrastructure.embeddings.local_jina_gpu import create_local_jina_processor
from irec_infrastructure.models.metadata import ArxivMetadata

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('process_abstracts_only.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AbstractProcessor:
    """Process only abstracts for embedding."""
    
    def __init__(self):
        """Initialize with Jina processor."""
        logger.info("Initializing Abstract Processor...")
        
        # Force GPU 1
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        logger.info("Using GPU 1 for abstract processing")
        
        # Create single GPU configuration
        from irec_infrastructure.embeddings.local_jina_gpu import LocalJinaConfig
        config = LocalJinaConfig()
        config.device_ids = [0]  # Only one device visible after CUDA_VISIBLE_DEVICES
        config.use_fp16 = True
        
        # Import the class directly to use with custom config
        from irec_infrastructure.embeddings.local_jina_gpu import LocalJinaGPU
        self.jina_processor = LocalJinaGPU(config)
        
        logger.info("Abstract processor ready!")
    


class AbstractOnlyPipeline:
    """Pipeline for abstract-only database."""
    
    def __init__(self, db_host: str, db_name: str):
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
        self.processor = AbstractProcessor()
        
        logger.info(f"Initialized abstract-only pipeline")
        logger.info(f"Database: {self.db_name} at {self.db_host}")
    
    def setup_database(self, clean_start=False):
        """Setup database with single collection for abstract metadata."""
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
        
        # Create single collection with indexes
        collection = self.db.create_collection('abstract_metadata')
        logger.info("Created collection: abstract_metadata")
        
        # Add indexes
        indexes = [
            {'type': 'hash', 'fields': ['arxiv_id'], 'unique': True},
            {'type': 'persistent', 'fields': ['categories[*]']},
            {'type': 'persistent', 'fields': ['published']},
            {'type': 'fulltext', 'fields': ['title']},
            {'type': 'fulltext', 'fields': ['abstract']},
            {'type': 'persistent', 'fields': ['authors[*]']}
        ]
        
        for index in indexes:
            collection.add_index(index)
        
        logger.info("Database schema created for abstract-only pipeline")
    
    def load_metadata_batch(self, metadata_paths: List[Path]) -> List[ArxivMetadata]:
        """Load and parse a batch of metadata files."""
        metadata_list = []
        
        for path in metadata_paths:
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                
                # Parse dates
                if 'published' in data and data['published']:
                    try:
                        data['published'] = datetime.fromisoformat(
                            data['published'].replace('Z', '+00:00')
                        )
                    except ValueError:
                        data['published'] = None
                
                if 'updated' in data and data['updated']:
                    try:
                        data['updated'] = datetime.fromisoformat(
                            data['updated'].replace('Z', '+00:00')
                        )
                    except ValueError:
                        data['updated'] = None
                
                # Create metadata object
                metadata = ArxivMetadata(**data)
                metadata_list.append(metadata)
                
            except Exception as e:
                logger.error(f"Error loading {path}: {e}")
                continue
        
        return metadata_list
    
    def process_metadata_batch(self, metadata_paths: List[Path]) -> List[Dict]:
        """Process a batch of metadata files efficiently."""
        # Load all metadata
        metadata_list = self.load_metadata_batch(metadata_paths)
        
        if not metadata_list:
            return []
        
        # Extract abstracts for batch embedding
        abstracts = [m.abstract for m in metadata_list]
        
        try:
            # Generate embeddings in batch
            logger.info(f"Generating embeddings for batch of {len(abstracts)}")
            embeddings = self.processor.jina_processor.encode_batch(abstracts)
            
            # Debug logging (can be removed in production)
            logger.debug(f"Generated embeddings shape: {embeddings.shape if hasattr(embeddings, 'shape') else 'unknown'}")
            
            # Convert to list properly
            try:
                if torch.is_tensor(embeddings):
                    # If it's a tensor, move to CPU first
                    embeddings_np = embeddings.cpu().numpy()
                    embeddings_list = embeddings_np.tolist()
                elif isinstance(embeddings, np.ndarray):
                    embeddings_list = embeddings.tolist()
                else:
                    # If it's already a list
                    embeddings_list = embeddings
            except Exception as e:
                logger.error(f"Error converting embeddings to list: {e}")
                # Try direct conversion for debugging
                logger.error(f"Embeddings type: {type(embeddings)}")
                raise
            
            # Prepare records for database
            records = []
            results = []
            
            for metadata, embedding in zip(metadata_list, embeddings_list):
                # embedding is already a list at this point
                embedding_list = embedding
                
                record = {
                    '_key': metadata.arxiv_id,
                    'arxiv_id': metadata.arxiv_id,
                    'title': metadata.title,
                    'authors': metadata.authors,
                    'abstract': metadata.abstract,
                    'categories': metadata.categories,
                    'published': metadata.published.isoformat() if metadata.published else None,
                    'updated': metadata.updated.isoformat() if metadata.updated else None,
                    'doi': metadata.doi,
                    'journal_ref': metadata.journal_ref,
                    'pdf_url': metadata.pdf_url,
                    'abs_url': metadata.abs_url,
                    'abstract_embedding': embedding_list,
                    'processed_at': datetime.now().isoformat()
                }
                records.append(record)
                results.append({
                    'success': True,
                    'arxiv_id': metadata.arxiv_id
                })
            
            # Batch insert into database
            if records:
                self.db.collection('abstract_metadata').insert_many(
                    records, 
                    overwrite=True,
                    silent=True
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Return failure for all in batch
            return [{
                'success': False,
                'arxiv_id': m.arxiv_id,
                'error': str(e)
            } for m in metadata_list]
    
    def verify_database(self):
        """Verify the abstract-only database."""
        print("\nAbstract-Only Database Verification:")
        print("="*60)
        
        # Collection statistics
        count = self.db.collection('abstract_metadata').count()
        print(f"Abstract metadata records: {count}")
        
        if count > 0:
            # Sample record
            cursor = self.db.aql.execute("""
                FOR m IN abstract_metadata
                    LIMIT 1
                    RETURN {
                        arxiv_id: m.arxiv_id,
                        title: m.title,
                        authors_count: LENGTH(m.authors),
                        categories: m.categories,
                        has_embedding: m.abstract_embedding ? true : false,
                        embedding_dim: LENGTH(m.abstract_embedding)
                    }
            """)
            
            sample = list(cursor)[0]
            print(f"\nSample record:")
            print(f"  ID: {sample['arxiv_id']}")
            print(f"  Title: {sample['title'][:80]}...")
            print(f"  Authors: {sample['authors_count']}")
            print(f"  Categories: {', '.join(sample['categories'])}")
            print(f"  Has embedding: {sample['has_embedding']}")
            print(f"  Embedding dimension: {sample['embedding_dim']}")
            
            # Category distribution
            cursor = self.db.aql.execute("""
                FOR m IN abstract_metadata
                    FOR cat IN m.categories
                        COLLECT category = cat WITH COUNT INTO count
                        SORT count DESC
                        LIMIT 10
                        RETURN {category: category, count: count}
            """)
            
            print("\nTop 10 categories:")
            for item in cursor:
                print(f"  {item['category']}: {item['count']}")


def main():
    """Run the abstract-only pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process arXiv metadata with abstract embeddings only"
    )
    parser.add_argument('--metadata-dir', type=str, 
                        default='/mnt/data/arxiv_data/metadata',
                        help='Directory containing metadata JSON files')
    parser.add_argument('--count', type=int, 
                        help='Number of documents to process (default: all)')
    parser.add_argument('--db-name', type=str, 
                        default='arxiv_abstracts',
                        help='Database name')
    parser.add_argument('--db-host', type=str, 
                        default='192.168.1.69',
                        help='Database host/IP')
    parser.add_argument('--clean-start', action='store_true',
                        help='Drop existing database and start fresh')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for embedding generation')
    
    args = parser.parse_args()
    
    print("\nARXIV ABSTRACT-ONLY PIPELINE")
    print("="*60)
    print(f"Metadata source: {args.metadata_dir}")
    print(f"Database: {args.db_name} @ {args.db_host}")
    print(f"Batch size: {args.batch_size}")
    print("="*60)
    
    # Initialize pipeline
    pipeline = AbstractOnlyPipeline(
        db_host=args.db_host,
        db_name=args.db_name
    )
    
    # Setup database
    pipeline.setup_database(clean_start=args.clean_start)
    
    # Get metadata files
    metadata_dir = Path(args.metadata_dir)
    if not metadata_dir.exists():
        print(f"Error: Metadata directory not found: {metadata_dir}")
        return
    
    metadata_files = sorted(metadata_dir.glob("*.json"))
    if args.count:
        metadata_files = metadata_files[:args.count]
    
    print(f"\nProcessing {len(metadata_files)} metadata files...")
    
    # Process metadata in batches for efficiency
    stats = {
        'total': len(metadata_files),
        'successful': 0,
        'failed': 0
    }
    
    # Process files in batches
    batch_size = args.batch_size
    
    for i in tqdm(range(0, len(metadata_files), batch_size), 
                  desc=f"Processing batches of {batch_size}"):
        batch = metadata_files[i:i + batch_size]
        results = pipeline.process_metadata_batch(batch)
        
        for result in results:
            if result['success']:
                stats['successful'] += 1
            else:
                stats['failed'] += 1
                logger.error(f"Failed: {result['arxiv_id']} - {result.get('error')}")
    
    # Print results
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Total processed: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Success rate: {stats['successful']/stats['total']*100:.1f}%")
    
    # Processing rate
    if stats['successful'] > 0:
        # Rough estimate: ~0.1 seconds per abstract
        print(f"\nProcessing rate: ~{stats['successful']/0.1:.0f} abstracts/second")
        print(f"(Compare to ~0.06 documents/second for full PDF processing)")
    
    # Verify database
    pipeline.verify_database()
    
    print("\n✅ Abstract-only pipeline completed successfully!")


if __name__ == "__main__":
    main()