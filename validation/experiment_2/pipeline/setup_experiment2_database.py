#!/usr/bin/env python3
"""
Set up ArangoDB for Experiment 2 with late chunking
Creates a separate database to preserve experiment_1 results
"""

import os
from arango import ArangoClient
from datetime import datetime

def setup_experiment2_database():
    """Create and configure database for experiment 2"""
    
    # Configuration
    db_name = "information_reconstructionism_exp2"
    arango_host = os.environ.get('ARANGO_HOST', 'http://192.168.1.69:8529')
    username = os.environ.get('ARANGO_USERNAME', 'root')
    password = os.environ.get('ARANGO_PASSWORD', '')
    
    print("=" * 60)
    print("EXPERIMENT 2 DATABASE SETUP")
    print("=" * 60)
    print(f"Creating database: {db_name}")
    print(f"Timestamp: {datetime.now()}")
    
    # Connect to ArangoDB
    client = ArangoClient(hosts=arango_host)
    sys_db = client.db('_system', username=username, password=password)
    
    # Check if database exists
    if sys_db.has_database(db_name):
        print(f"\n⚠️  Database '{db_name}' already exists!")
        response = input("Delete and recreate? (y/n): ")
        if response.lower() == 'y':
            sys_db.delete_database(db_name)
            print(f"Deleted existing database: {db_name}")
        else:
            print("Keeping existing database")
            return
    
    # Create database
    sys_db.create_database(db_name)
    print(f"✓ Created database: {db_name}")
    
    # Connect to new database
    db = client.db(db_name, username=username, password=password)
    
    # Create collections
    collections = {
        'papers_exp2': {
            'type': 'document',
            'description': 'Paper metadata and document-level embeddings'
        },
        'chunks_exp2': {
            'type': 'document', 
            'description': 'Individual chunks with embeddings from late chunking'
        },
        'chunk_similarities_exp2': {
            'type': 'edge',
            'description': 'Chunk-to-chunk similarity edges'
        },
        'document_similarities_exp2': {
            'type': 'edge',
            'description': 'Document-level similarities aggregated from chunks'
        },
        'chunk_hierarchy_exp2': {
            'type': 'edge',
            'description': 'Chunk-to-document relationships'
        }
    }
    
    print("\nCreating collections:")
    for coll_name, config in collections.items():
        if config['type'] == 'edge':
            collection = db.create_collection(coll_name, edge=True)
        else:
            collection = db.create_collection(coll_name)
        print(f"  ✓ {coll_name} ({config['type']}): {config['description']}")
    
    # Create indexes for performance
    print("\nCreating indexes:")
    
    # Papers indexes
    papers = db.collection('papers_exp2')
    papers.add_persistent_index(fields=['year'])
    papers.add_persistent_index(fields=['primary_category'])
    print("  ✓ papers_exp2: year, primary_category")
    
    # Chunks indexes
    chunks = db.collection('chunks_exp2')
    chunks.add_persistent_index(fields=['paper_id'])
    chunks.add_persistent_index(fields=['chunk_index'])
    chunks.add_persistent_index(fields=['chunk_type'])  # section, paragraph, etc.
    print("  ✓ chunks_exp2: paper_id, chunk_index, chunk_type")
    
    # Similarity indexes
    chunk_sim = db.collection('chunk_similarities_exp2')
    chunk_sim.add_persistent_index(fields=['similarity'])
    chunk_sim.add_persistent_index(fields=['context'])
    print("  ✓ chunk_similarities_exp2: similarity, context")
    
    doc_sim = db.collection('document_similarities_exp2')
    doc_sim.add_persistent_index(fields=['similarity'])
    doc_sim.add_persistent_index(fields=['aggregation_method'])
    print("  ✓ document_similarities_exp2: similarity, aggregation_method")
    
    # Create views for analysis
    print("\nCreating analysis views:")
    
    # High similarity chunks view
    view_definition = {
        'links': {
            'chunk_similarities_exp2': {
                'analyzers': ['identity'],
                'fields': {
                    'similarity': {},
                    'context': {}
                },
                'includeAllFields': False,
                'storeValues': 'id',
                'trackListPositions': False
            }
        }
    }
    
    db.create_arangosearch_view(
        name='high_similarity_chunks',
        properties=view_definition
    )
    print("  ✓ high_similarity_chunks view")
    
    # Database statistics
    print("\nDatabase configuration complete!")
    print("\nDatabase statistics:")
    print(f"  Name: {db_name}")
    print(f"  Collections: {len(collections)}")
    print(f"  Host: {arango_host}")
    
    # Save configuration
    config_file = "/home/todd/reconstructionism/validation/experiment_2/database_config.json"
    import json
    config = {
        'database': db_name,
        'collections': list(collections.keys()),
        'created': datetime.now().isoformat(),
        'host': arango_host,
        'expected_papers': 2000,
        'expected_chunks_per_paper': '50-200',
        'features': {
            'late_chunking': True,
            'multi_scale_similarity': True,
            'chunk_hierarchy': True
        }
    }
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nConfiguration saved to: {config_file}")
    print("\n✓ Experiment 2 database ready for late chunking pipeline!")
    
    return db

if __name__ == "__main__":
    setup_experiment2_database()