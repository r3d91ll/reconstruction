#!/usr/bin/env python3
"""
STEP 2: Load papers into ArangoDB
Just nodes, no edges yet
"""

import os
import json
import glob
from arango import ArangoClient

def main():
    # Configuration
    input_dir = os.environ.get("VALIDATION_OUTPUT_DIR", "/home/todd/reconstructionism/validation/data/papers_with_embeddings")
    db_name = "information_reconstructionism"
    
    # Connect to ArangoDB using environment variables
    arango_host = os.environ.get('ARANGO_HOST', 'http://192.168.1.69:8529')
    client = ArangoClient(hosts=arango_host)
    
    # Get credentials from environment
    username = os.environ.get('ARANGO_USERNAME', 'root')
    password = os.environ.get('ARANGO_PASSWORD', '')
    
    # Connect to system database
    sys_db = client.db('_system', username=username, password=password)
    
    # Create database if needed
    if not sys_db.has_database(db_name):
        sys_db.create_database(db_name)
        print(f"Created database: {db_name}")
    
    # Connect to our database
    db = client.db(db_name, username=username, password=password)
    
    # Create papers collection
    if not db.has_collection('papers'):
        papers_collection = db.create_collection('papers')
        print("Created collection: papers")
    else:
        papers_collection = db.collection('papers')
        # Clear existing data for clean test
        papers_collection.truncate()
        print("Cleared existing papers collection")
    
    # Load papers
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    print(f"\nLoading {len(json_files)} papers...")
    
    success = 0
    for i, json_file in enumerate(json_files):
        try:
            with open(json_file, 'r') as f:
                paper = json.load(f)
            
            # Create document for ArangoDB
            doc = {
                '_key': paper['arxiv_id'].replace('.', '_'),  # ArangoDB key format
                'arxiv_id': paper['arxiv_id'],
                'title': paper['title'],
                'year': paper.get('year', 2020),
                'categories': paper.get('categories', []),
                'primary_category': paper.get('primary_category', ''),
                
                # Store embeddings
                'embeddings': paper['dimensions']['WHAT']['embeddings'],
                'embedding_dim': paper['dimensions']['WHAT']['embedding_dim'],
                
                # We'll compute this later
                'physical_grounding_factor': None
            }
            
            papers_collection.insert(doc)
            success += 1
            
            if (i + 1) % 10 == 0:
                print(f"Loaded {i + 1}/{len(json_files)}")
                
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    print(f"\n✓ STEP 2 COMPLETE")
    print(f"Loaded {success}/{len(json_files)} papers into ArangoDB")
    
    # Verify with simple queries
    print("\nVerification queries:")
    
    # Count papers
    count = papers_collection.count()
    print(f"Total papers in database: {count}")
    
    # Papers by year
    query = """
    FOR paper IN papers
        COLLECT year = paper.year WITH COUNT INTO count
        SORT year
        RETURN {year: year, count: count}
    """
    cursor = db.aql.execute(query)
    year_counts = list(cursor)
    print(f"\nPapers by year: {len(year_counts)} different years")
    for item in year_counts[-5:]:  # Last 5 years
        print(f"  {item['year']}: {item['count']} papers")
    
    # Check embedding storage
    query = """
    FOR paper IN papers
        LIMIT 1
        RETURN {
            title: paper.title,
            has_embeddings: LENGTH(paper.embeddings) > 0,
            embedding_dim: paper.embedding_dim
        }
    """
    cursor = db.aql.execute(query)
    sample = list(cursor)[0]
    print(f"\nSample paper check:")
    print(f"  Title: {sample['title'][:60]}...")
    print(f"  Has embeddings: {sample['has_embeddings']}")
    print(f"  Embedding dimension: {sample['embedding_dim']}")

if __name__ == "__main__":
    main()