#!/usr/bin/env python3
"""
Step 2: Load papers and chunks into ArangoDB for experiment 2
"""

import os
import json
from pathlib import Path
from arango import ArangoClient
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Configuration
    db_name = os.environ.get("EXP2_DB_NAME", "information_reconstructionism_exp2")
    results_dir = os.environ.get("EXP2_RESULTS_DIR", ".")
    chunks_dir = os.path.join(results_dir, "chunks")
    
    # ArangoDB connection
    arango_host = os.environ.get('ARANGO_HOST', 'http://192.168.1.69:8529')
    username = os.environ.get('ARANGO_USERNAME', 'root')
    password = os.environ.get('ARANGO_PASSWORD', '')
    
    logger.info("=" * 60)
    logger.info("STEP 2: LOAD CHUNKS TO ARANGODB")
    logger.info("=" * 60)
    logger.info(f"Database: {db_name}")
    logger.info(f"Chunks directory: {chunks_dir}")
    
    # Connect to database
    client = ArangoClient(hosts=arango_host)
    db = client.db(db_name, username=username, password=password)
    
    # Get collections
    papers_coll = db.collection('papers_exp2')
    chunks_coll = db.collection('chunks_exp2')
    hierarchy_coll = db.collection('chunk_hierarchy_exp2')
    
    # Clear existing data
    logger.info("Clearing existing data...")
    papers_coll.truncate()
    chunks_coll.truncate()
    hierarchy_coll.truncate()
    
    # Load chunk files
    chunk_files = list(Path(chunks_dir).glob("*_chunks.json"))
    logger.info(f"Found {len(chunk_files)} chunk files to load")
    
    papers_loaded = 0
    chunks_loaded = 0
    
    for chunk_file in chunk_files:
        try:
            with open(chunk_file, 'r') as f:
                data = json.load(f)
            
            paper_id = data['paper_id']
            
            # Create paper document
            paper_doc = {
                '_key': paper_id.replace('.', '_').replace('/', '_'),
                'arxiv_id': paper_id,
                'title': data.get('title', ''),
                'num_chunks': data['num_chunks'],
                'processing_timestamp': data.get('timestamp', datetime.now().isoformat())
            }
            
            # Insert paper
            paper_result = papers_coll.insert(paper_doc)
            papers_loaded += 1
            
            # Insert chunks
            for chunk in data['chunks']:
                # Ensure _key is valid
                chunk['_key'] = chunk['_key'].replace('.', '_').replace('/', '_')
                
                # Insert chunk
                chunk_result = chunks_coll.insert(chunk)
                chunks_loaded += 1
                
                # Create hierarchy edge (chunk belongs to paper)
                hierarchy_edge = {
                    '_from': f"chunks_exp2/{chunk['_key']}",
                    '_to': f"papers_exp2/{paper_doc['_key']}",
                    'relationship': 'belongs_to',
                    'chunk_index': chunk['chunk_index']
                }
                hierarchy_coll.insert(hierarchy_edge)
            
            if papers_loaded % 10 == 0:
                logger.info(f"Progress: {papers_loaded} papers, {chunks_loaded} chunks loaded")
                
        except Exception as e:
            logger.error(f"Error loading {chunk_file.name}: {e}")
    
    # Verify results
    logger.info("\nVerifying loaded data:")
    
    # Count documents
    papers_count = papers_coll.count()
    chunks_count = chunks_coll.count()
    hierarchy_count = hierarchy_coll.count()
    
    logger.info(f"  Papers loaded: {papers_count}")
    logger.info(f"  Chunks loaded: {chunks_count}")
    logger.info(f"  Hierarchy edges: {hierarchy_count}")
    
    # Sample query
    query = """
    FOR paper IN papers_exp2
        LET chunk_count = LENGTH(
            FOR chunk IN chunks_exp2
                FILTER chunk.paper_id == paper.arxiv_id
                RETURN 1
        )
        RETURN {
            paper: paper.title,
            chunks: chunk_count
        }
    """
    
    logger.info("\nSample papers and chunk counts:")
    for result in db.aql.execute(query, count=True, batch_size=5):
        logger.info(f"  {result['paper'][:60]}... - {result['chunks']} chunks")
    
    logger.info("\n✓ STEP 2 COMPLETE")
    logger.info(f"Loaded {papers_loaded} papers with {chunks_loaded} chunks")
    
    return 0

if __name__ == "__main__":
    exit(main())