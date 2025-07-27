#!/usr/bin/env python3
"""Check if late chunking is working in the database"""

import os
import sys
from arango import ArangoClient

# Get database configuration from environment variables
DB_HOST = os.environ.get('ARANGO_HOST', 'http://localhost:8529')
DB_NAME = os.environ.get('ARANGO_DB', 'irec_three_collections')
DB_USER = os.environ.get('ARANGO_USER', 'root')
DB_PASS = os.environ.get('ARANGO_PASSWORD', '')

if not DB_PASS:
    print("Error: ARANGO_PASSWORD environment variable not set")
    print("Please set database credentials using environment variables:")
    print("  export ARANGO_HOST='http://your-host:8529'")
    print("  export ARANGO_DB='your-database'")
    print("  export ARANGO_USER='your-username'")
    print("  export ARANGO_PASSWORD='your-password'")
    sys.exit(1)

try:
    # Connect to database
    client = ArangoClient(hosts=DB_HOST)
    db = client.db(DB_NAME, username=DB_USER, password=DB_PASS)
    
    # Check chunks collection
    chunks_coll = db.collection('chunks')
    docs_coll = db.collection('documents')
    
    print(f"Documents count: {docs_coll.count()}")
    print(f"Chunks count: {chunks_coll.count()}")
    
    # Check a few chunks
    if chunks_coll.count() > 0:
        print("\nSample chunks:")
        try:
            for chunk in chunks_coll.find().limit(3):
                try:
                    print(f"\nChunk ID: {chunk.get('chunk_id')}")
                    print(f"ArXiv ID: {chunk.get('arxiv_id')}")
                    print(f"Late chunking: {chunk.get('late_chunking', False)}")
                    print(f"Chunk index: {chunk.get('chunk_index')}")
                    print(f"Text length: {len(chunk.get('text', ''))}")
                    print(f"Embedding shape: {len(chunk.get('embedding', []))}")
                    if 'chunk_metadata' in chunk:
                        print(f"Metadata: {chunk['chunk_metadata']}")
                except Exception as e:
                    print(f"Error processing chunk: {e}")
                    continue
        except Exception as e:
            print(f"Error iterating over chunks: {e}")
    else:
        print("\nNo chunks found yet - late chunking may still be processing...")
        
except Exception as e:
    print(f"Error connecting to database or accessing collections: {e}")
    print("\nPlease check your database configuration:")
    print(f"  Host: {DB_HOST}")
    print(f"  Database: {DB_NAME}")
    print(f"  User: {DB_USER}")
    sys.exit(1)