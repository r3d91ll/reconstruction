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
    # Updated to load from papers directory with Docling-extracted content
    input_dir = os.environ.get("PAPERS_DIR", "/home/todd/olympus/Erebus/unstructured/papers")
    db_name = "information_reconstructionism"
    limit = int(os.environ.get("PAPER_LIMIT", "1000"))  # Default to 1000 papers
    
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
    
    # Load papers with Docling-extracted content
    json_files = glob.glob(os.path.join(input_dir, "*.json"))[:limit]
    print(f"\nLoading up to {limit} papers from {len(json_files)} available...")
    
    success = 0
    skipped_no_content = 0
    errors = 0
    
    for i, json_file in enumerate(json_files):
        try:
            with open(json_file, 'r') as f:
                paper = json.load(f)
            
            # Skip if no pdf_content (not processed by Docling)
            if 'pdf_content' not in paper:
                skipped_no_content += 1
                continue
            
            # Check for embeddings in expected location
            has_embeddings = False
            embeddings = None
            embedding_dim = 0
            
            if 'dimensions' in paper and 'WHAT' in paper['dimensions']:
                if 'embeddings' in paper['dimensions']['WHAT']:
                    has_embeddings = True
                    embeddings = paper['dimensions']['WHAT']['embeddings']
                    embedding_dim = paper['dimensions']['WHAT'].get('embedding_dim', len(embeddings))
            
            if not has_embeddings:
                print(f"Warning: No embeddings found for {os.path.basename(json_file)}")
                continue
            
            # Extract content statistics
            pdf_content = paper['pdf_content']
            content_stats = {
                'markdown_length': len(pdf_content.get('markdown', '')),
                'num_sections': len(pdf_content.get('sections', [])),
                'num_images': len(pdf_content.get('images', [])),
                'num_tables': len(pdf_content.get('tables', [])),
                'num_equations': len(pdf_content.get('equations', [])),
                'num_code_blocks': len(pdf_content.get('code_blocks', [])),
                'num_references': len(pdf_content.get('references', []))
            }
            
            # Create document for ArangoDB
            doc = {
                '_key': paper.get('id', os.path.basename(json_file).replace('.json', '')).replace('.', '_').replace('/', '_'),
                'arxiv_id': paper.get('id', ''),
                'title': paper.get('title', ''),
                'abstract': paper.get('abstract', ''),
                'year': paper.get('year', 2020),
                'categories': paper.get('categories', []),
                'primary_category': paper.get('categories', [''])[0] if paper.get('categories') else '',
                
                # Store embeddings
                'embeddings': embeddings,
                'embedding_dim': embedding_dim,
                'embedding_method': paper['dimensions']['WHAT'].get('embedding_method', 'jina-v4-docling'),
                
                # Content metadata
                'has_full_content': paper['dimensions']['WHAT'].get('has_full_content', False),
                'content_length': paper['dimensions']['WHAT'].get('context_length', 0),
                'content_stats': content_stats,
                
                # Store markdown content (truncated for DB)
                'content_preview': pdf_content.get('markdown', '')[:5000],  # First 5K chars
                
                # We'll compute this later
                'physical_grounding_factor': None
            }
            
            papers_collection.insert(doc)
            success += 1
            
            if (i + 1) % 100 == 0:
                print(f"Progress: {i + 1}/{len(json_files)} - Loaded: {success}, Skipped: {skipped_no_content}")
                
        except Exception as e:
            errors += 1
            print(f"Error loading {os.path.basename(json_file)}: {e}")
    
    print(f"\n✓ STEP 2 COMPLETE")
    print(f"Results:")
    print(f"  Loaded: {success} papers")
    print(f"  Skipped (no content): {skipped_no_content}")
    print(f"  Errors: {errors}")
    print(f"  Total processed: {len(json_files)}")
    
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