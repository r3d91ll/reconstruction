#!/usr/bin/env python3
"""
Prepare PDF metadata in database by mapping arxiv IDs to tar files
This updates the pdf_status.tar_source field for existing documents
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Set
from arango import ArangoClient
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PDFMetadataUpdater:
    """Update database with PDF tar source information"""
    
    def __init__(self, db_config: Dict, tar_dir: str = "/mnt/data-cold/arxiv_data"):
        self.db_config = db_config
        self.tar_dir = tar_dir
        self._initialize_db()
        
    def _initialize_db(self):
        """Initialize database connection with error handling"""
        try:
            client = ArangoClient(hosts=f'http://{self.db_config["host"]}:{self.db_config["port"]}')
        except Exception as e:
            logger.error(f"Failed to initialize ArangoDB client: {e}")
            logger.error(f"Please check that ArangoDB is running on {self.db_config['host']}:{self.db_config['port']}")
            raise
            
        try:
            self.db = client.db(
                self.db_config['name'],
                username=self.db_config['username'],
                password=self.db_config['password']
            )
        except Exception as e:
            logger.error(f"Failed to connect to database '{self.db_config['name']}': {e}")
            logger.error("Please check database name, username, and password")
            raise
            
        try:
            self.collection = self.db.collection(self.db_config['collection'])
        except Exception as e:
            logger.error(f"Failed to access collection '{self.db_config['collection']}': {e}")
            logger.error("Please ensure the collection exists in the database")
            raise
        
    def get_tar_mappings(self, tar_dir: str) -> Dict[str, List[str]]:
        """Build mapping of tar files to arxiv IDs they contain"""
        tar_dir_path = Path(tar_dir)
        tar_files = sorted(tar_dir_path.glob("arXiv_pdf_*.tar"))
        
        logger.info(f"Found {len(tar_files)} tar files in {tar_dir}")
        
        # Build mapping based on tar file naming convention
        # Example: arXiv_pdf_2301_001.tar contains papers from 2301.xxxxx
        mappings = {}
        
        for tar_path in tar_files:
            # Extract year_month and batch from filename
            # Format: arXiv_pdf_YYMM_BBB.tar
            parts = tar_path.stem.split('_')
            if len(parts) >= 4:
                year_month = parts[2]  # e.g., "2301"
                batch = parts[3]       # e.g., "001"
                
                # This tar contains papers from this year/month
                tar_name = tar_path.name
                mappings[tar_name] = {
                    'year_month': year_month,
                    'batch': batch,
                    'path': str(tar_path)
                }
                
        return mappings
        
    def update_documents_batch(self, updates: List[Dict]) -> int:
        """Update a batch of documents with tar source info using ArangoDB transaction"""
        if not updates:
            return 0
            
        # Define transaction JavaScript function
        transaction_js = """
        function(params) {
            const db = require('@arangodb').db;
            const collection = db._collection(params.collection);
            let successCount = 0;
            
            for (let update of params.updates) {
                try {
                    collection.update(update._key, {
                        pdf_status: {
                            state: 'unprocessed',
                            tar_source: update.tar_source,
                            last_updated: null,
                            retry_count: 0,
                            error_message: null,
                            processing_time_seconds: null
                        }
                    });
                    successCount++;
                } catch (e) {
                    // Log error but continue with other updates
                    console.error(`Failed to update ${update._key}: ${e.message}`);
                }
            }
            
            return successCount;
        }
        """
        
        try:
            # Execute transaction
            result = self.db.transaction(
                {'write': [self.db_config['collection']]},
                transaction_js,
                params={
                    'collection': self.db_config['collection'],
                    'updates': updates
                }
            )
            
            return result
        except Exception as e:
            logger.error(f"Transaction failed for batch update: {e}")
            return 0
        
    def map_documents_to_tars(self, limit: int = None):
        """Map documents to their tar sources based on arxiv ID patterns"""
        
        # Get tar mappings
        tar_mappings = self.get_tar_mappings(self.tar_dir)
        
        # Build year_month to tar mapping
        ym_to_tar = {}
        for tar_name, info in tar_mappings.items():
            year_month = info['year_month']
            if year_month not in ym_to_tar:
                ym_to_tar[year_month] = []
            ym_to_tar[year_month].append(tar_name)
            
        logger.info(f"Built mappings for {len(ym_to_tar)} year/month combinations")
        
        # Query documents without tar sources
        query = """
        FOR doc IN @@collection
            FILTER doc.pdf_status.tar_source == null OR doc.pdf_status.tar_source == ""
            LIMIT @limit
            RETURN {
                _key: doc._key,
                submitted_date: doc.submitted_date
            }
        """
        
        cursor = self.db.aql.execute(
            query,
            bind_vars={
                '@collection': self.db_config['collection'],
                'limit': limit or 10000000
            }
        )
        
        documents = list(cursor)
        logger.info(f"Found {len(documents)} documents needing tar source mapping")
        
        # Process in batches
        batch_size = 1000
        updates = []
        updated_count = 0
        
        for doc in tqdm(documents, desc="Mapping documents to tar files"):
            arxiv_id = doc['_key']
            
            # Extract year/month from arxiv ID
            # New format: "2301.00001" -> "2301"
            # Old format: "math/9901001" -> extract differently
            if '.' in arxiv_id:
                year_month = arxiv_id.split('.')[0]
            elif '/' in arxiv_id:
                # Old format - extract year/month from the numeric part
                parts = arxiv_id.split('/')
                if len(parts) == 2 and parts[1]:
                    # Extract YYMM from old format like "math/9901001"
                    numeric_part = parts[1]
                    if len(numeric_part) >= 4:
                        year_month = numeric_part[:4]  # First 4 digits are YYMM
                    else:
                        logger.warning(f"Could not extract year/month from old format arxiv ID: {arxiv_id}")
                        continue
                else:
                    logger.warning(f"Unexpected old format arxiv ID: {arxiv_id}")
                    continue
            else:
                logger.warning(f"Unexpected arxiv ID format: {arxiv_id}")
                continue
                
                # Find corresponding tar file
                if year_month in ym_to_tar:
                    # For now, use the first tar file for this year/month
                    # In practice, might need more sophisticated mapping
                    tar_source = ym_to_tar[year_month][0]
                    
                    updates.append({
                        '_key': arxiv_id,
                        'tar_source': tar_source
                    })
                    
                    if len(updates) >= batch_size:
                        count = self.update_documents_batch(updates)
                        updated_count += count
                        updates = []
                        
        # Update remaining
        if updates:
            count = self.update_documents_batch(updates)
            updated_count += count
            
        logger.info(f"Updated {updated_count} documents with tar source information")
        
    def verify_mapping(self, sample_size: int = 100):
        """Verify that documents have been mapped correctly"""
        query = """
        FOR doc IN @@collection
            FILTER doc.pdf_status.tar_source != null AND doc.pdf_status.tar_source != ""
            LIMIT @limit
            RETURN {
                _key: doc._key,
                tar_source: doc.pdf_status.tar_source,
                state: doc.pdf_status.state
            }
        """
        
        cursor = self.db.aql.execute(
            query,
            bind_vars={
                '@collection': self.db_config['collection'],
                'limit': sample_size
            }
        )
        
        samples = list(cursor)
        logger.info(f"\nSample of mapped documents ({len(samples)} total):")
        for i, doc in enumerate(samples[:10]):
            logger.info(f"  {doc['_key']} -> {doc['tar_source']} [{doc['state']}]")
            
        # Count by state
        state_counts = {}
        for doc in samples:
            state = doc['state']
            state_counts[state] = state_counts.get(state, 0) + 1
            
        logger.info(f"\nState distribution in sample:")
        for state, count in state_counts.items():
            logger.info(f"  {state}: {count}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare PDF metadata for processing")
    parser.add_argument('--db-name', type=str, default='arxiv_single_collection')
    parser.add_argument('--db-host', type=str, default='192.168.1.69')
    parser.add_argument('--tar-dir', type=str, default='/mnt/data-cold/arxiv_data', help='Directory containing tar files')
    parser.add_argument('--limit', type=int, help='Limit number of documents to update')
    parser.add_argument('--verify-only', action='store_true', help='Only verify existing mappings')
    
    args = parser.parse_args()
    
    if not os.environ.get('ARANGO_PASSWORD'):
        logger.error("ARANGO_PASSWORD environment variable not set")
        sys.exit(1)
        
    db_config = {
        'name': args.db_name,
        'collection': 'arxiv_documents',
        'host': args.db_host,
        'port': 8529,
        'username': 'root',
        'password': os.environ['ARANGO_PASSWORD']
    }
    
    updater = PDFMetadataUpdater(db_config, tar_dir=args.tar_dir)
    
    if args.verify_only:
        updater.verify_mapping()
    else:
        logger.info("Mapping documents to tar sources...")
        updater.map_documents_to_tars(limit=args.limit)
        logger.info("\nVerifying mappings...")
        updater.verify_mapping()

if __name__ == '__main__':
    main()