#!/usr/bin/env python3
"""
Setup ArangoDB Schema for Information Reconstructionism with TRUE Late Chunking

This creates the database schema optimized for:
- TRUE late chunking (Jina creates chunks from full documents)
- Full arXiv metadata integration
- Dimensional analysis (WHERE, WHAT, CONVEYANCE, TIME, FRAME)
- Citation network for FRAME discovery
"""

import argparse
import logging
from datetime import datetime
from typing import Dict, List
import getpass

from arango import ArangoClient
from arango.database import StandardDatabase


def create_irec_schema(
    db_name: str = "irec_late_chunking",
    host: str = "localhost",
    port: int = 8529,
    username: str = "root",
    password: str = "",
    clean_start: bool = False
) -> StandardDatabase:
    """
    Create the complete database schema for Information Reconstructionism.
    
    Args:
        db_name: Name of the database
        host: ArangoDB host
        port: ArangoDB port
        username: Database username
        password: Database password
        clean_start: If True, drop existing database and start fresh
        
    Returns:
        Database connection object
    """
    logging.info(f"Setting up IREC database schema: {db_name}")
    
    # Connect to ArangoDB
    client = ArangoClient(hosts=f'http://{host}:{port}')
    sys_db = client.db('_system', username=username, password=password)
    
    # Handle existing database
    if sys_db.has_database(db_name):
        if clean_start:
            logging.warning(f"Dropping existing database: {db_name}")
            sys_db.delete_database(db_name)
        else:
            logging.info(f"Using existing database: {db_name}")
            return client.db(db_name, username=username, password=password)
    
    # Create new database
    sys_db.create_database(db_name)
    db = client.db(db_name, username=username, password=password)
    
    # Define collections with proper schema
    collections = {
        # Main document collection with full metadata
        "documents": {
            "indexes": [
                {"type": "hash", "fields": ["arxiv_id"], "unique": True},
                {"type": "fulltext", "fields": ["title"]},
                {"type": "fulltext", "fields": ["abstract"]},
                {"type": "persistent", "fields": ["categories[*]"]},
                {"type": "persistent", "fields": ["published"]},
                {"type": "persistent", "fields": ["authors[*]"]},
            ],
            "schema": {
                "arxiv_id": "string (required)",
                "title": "string",
                "authors": "array of strings",
                "abstract": "string",
                "categories": "array of strings",
                "published": "datetime string",
                "updated": "datetime string",
                "full_text": "string (extracted by Docling)",
                "text_length": "integer",
                "num_chunks": "integer",
                "processing_timestamp": "datetime string",
                "pdf_url": "string",
                "abs_url": "string"
            }
        },
        
        # Chunks created by Jina's TRUE late chunking
        "chunks": {
            "indexes": [
                {"type": "hash", "fields": ["chunk_id"], "unique": True},
                {"type": "hash", "fields": ["arxiv_id"]},
                {"type": "persistent", "fields": ["chunk_index"]},
                {"type": "fulltext", "fields": ["text"]},
            ],
            "schema": {
                "chunk_id": "string (arxiv_id_chunk_N)",
                "arxiv_id": "string (link to document)",
                "chunk_index": "integer",
                "text": "string (chunk content)",
                "tokens": "integer (approx)",
                "embedding": "array of floats (768 dims)"
            }
        },
        
        # Dimensional scores for Information Reconstructionism
        "dimensional_scores": {
            "indexes": [
                {"type": "hash", "fields": ["arxiv_id"], "unique": True},
                {"type": "persistent", "fields": ["WHERE_score"]},
                {"type": "persistent", "fields": ["WHAT_score"]},
                {"type": "persistent", "fields": ["CONVEYANCE_score"]},
                {"type": "persistent", "fields": ["TIME_score"]},
                {"type": "persistent", "fields": ["total_information"]},
            ],
            "schema": {
                "arxiv_id": "string",
                "WHERE_score": "float [0,1]",
                "WHAT_score": "float [0,1]",
                "CONVEYANCE_score": "float [0,1]",
                "TIME_score": "float [0,1]",
                "context_alpha": "float (1.5-2.0)",
                "total_information": "float (multiplicative result)"
            }
        },
        
        # Document similarities (aggregated from chunks)
        "document_similarities": {
            "edge": True,
            "indexes": [
                {"type": "persistent", "fields": ["similarity"]},
                {"type": "hash", "fields": ["_from", "_to"], "unique": True},
            ],
            "schema": {
                "_from": "documents/arxiv_id",
                "_to": "documents/arxiv_id",
                "similarity": "float [0,1]",
                "chunk_pairs": "integer (contributing chunks)",
                "method": "string (aggregation method)"
            }
        },
        
        # Chunk-level similarities
        "chunk_similarities": {
            "edge": True,
            "indexes": [
                {"type": "persistent", "fields": ["similarity"]},
                {"type": "hash", "fields": ["_from", "_to"], "unique": True},
            ],
            "schema": {
                "_from": "chunks/chunk_id",
                "_to": "chunks/chunk_id",
                "similarity": "float [0,1]",
                "cross_document": "boolean"
            }
        },
        
        # Citation network for FRAME discovery
        "citations": {
            "edge": True,
            "indexes": [
                {"type": "persistent", "fields": ["citation_type"]},
                {"type": "hash", "fields": ["_from", "_to"], "unique": True},
            ],
            "schema": {
                "_from": "documents/arxiv_id (citing)",
                "_to": "documents/arxiv_id (cited)",
                "citation_type": "string (reference, extends, contradicts)",
                "context": "string (citation context)",
                "frame_compatibility": "float (discovered)"
            }
        },
        
        # Implementation tracking for CONVEYANCE validation
        "implementations": {
            "indexes": [
                {"type": "hash", "fields": ["arxiv_id"]},
                {"type": "persistent", "fields": ["platform"]},
                {"type": "persistent", "fields": ["stars"]},
                {"type": "persistent", "fields": ["implementation_date"]},
            ],
            "schema": {
                "arxiv_id": "string",
                "platform": "string (github, gitlab, etc)",
                "repository_url": "string",
                "stars": "integer",
                "forks": "integer",
                "implementation_date": "datetime string",
                "language": "string",
                "conveyance_evidence": "float"
            }
        },
        
        # Processing metadata
        "processing_log": {
            "indexes": [
                {"type": "persistent", "fields": ["processing_date"]},
                {"type": "persistent", "fields": ["status"]},
            ],
            "schema": {
                "batch_id": "string",
                "processing_date": "datetime string",
                "documents_processed": "integer",
                "chunks_created": "integer",
                "processing_mode": "string (local_gpu, api, etc)",
                "gpu_info": "object",
                "status": "string (completed, failed, partial)",
                "error_log": "array"
            }
        }
    }
    
    # Create collections
    for coll_name, config in collections.items():
        logging.info(f"Creating collection: {coll_name}")
        
        # Check if edge collection
        is_edge = config.get("edge", False)
        
        # Create collection
        if is_edge:
            collection = db.create_collection(coll_name, edge=True)
        else:
            collection = db.create_collection(coll_name)
        
        # Create indexes
        for index_config in config.get("indexes", []):
            try:
                if index_config["type"] == "hash":
                    collection.add_hash_index(
                        fields=index_config["fields"],
                        unique=index_config.get("unique", False)
                    )
                elif index_config["type"] == "persistent":
                    collection.add_persistent_index(
                        fields=index_config["fields"],
                        unique=index_config.get("unique", False)
                    )
                elif index_config["type"] == "fulltext":
                    collection.add_fulltext_index(
                        fields=index_config["fields"]
                    )
                logging.info(f"  Added {index_config['type']} index on: {index_config['fields']}")
            except Exception as e:
                logging.warning(f"  Index creation warning: {e}")
    
    # Create graph for relationships
    logging.info("Creating graph: irec_knowledge_graph")
    graph = db.create_graph("irec_knowledge_graph")
    
    # Add vertex collections
    graph.create_vertex_collection("documents")
    graph.create_vertex_collection("chunks")
    graph.create_vertex_collection("implementations")
    
    # Add edge definitions
    graph.create_edge_definition(
        edge_collection="document_similarities",
        from_vertex_collections=["documents"],
        to_vertex_collections=["documents"]
    )
    
    graph.create_edge_definition(
        edge_collection="chunk_similarities",
        from_vertex_collections=["chunks"],
        to_vertex_collections=["chunks"]
    )
    
    graph.create_edge_definition(
        edge_collection="citations",
        from_vertex_collections=["documents"],
        to_vertex_collections=["documents"]
    )
    
    # Store creation metadata
    db.collection("processing_log").insert({
        "_key": "schema_creation",
        "batch_id": "initial_setup",
        "processing_date": datetime.now().isoformat(),
        "documents_processed": 0,
        "chunks_created": 0,
        "processing_mode": "schema_only",
        "status": "completed",
        "schema_version": "2.0_late_chunking",
        "collections": list(collections.keys()),
        "expected_documents": 1960,
        "expected_chunks": 100000,
        "features": [
            "TRUE late chunking",
            "Full arXiv metadata",
            "Dimensional scoring",
            "Citation network",
            "Implementation tracking"
        ]
    })
    
    logging.info("IREC database schema creation complete!")
    
    return db


def verify_schema(db: StandardDatabase) -> Dict:
    """Verify the database schema is correctly set up."""
    verification = {
        "collections": {},
        "indexes": {},
        "graph": False,
        "sample_queries": []
    }
    
    # Check collections exist
    for coll_name in db.collections():
        if not coll_name.startswith("_"):  # Skip system collections
            coll = db.collection(coll_name)
            verification["collections"][coll_name] = {
                "exists": True,
                "count": coll.count(),
                "indexes": len(coll.indexes())
            }
    
    # Check graph exists
    if db.has_graph("irec_knowledge_graph"):
        verification["graph"] = True
    
    # Test sample queries
    test_queries = [
        {
            "name": "Find documents by category",
            "aql": """
            FOR doc IN documents
                FILTER 'cs.AI' IN doc.categories
                LIMIT 5
                RETURN {id: doc.arxiv_id, title: doc.title}
            """
        },
        {
            "name": "Count chunks per document",
            "aql": """
            FOR chunk IN chunks
                COLLECT arxiv_id = chunk.arxiv_id WITH COUNT INTO num_chunks
                LIMIT 5
                RETURN {arxiv_id, num_chunks}
            """
        }
    ]
    
    for query in test_queries:
        try:
            cursor = db.aql.execute(query["aql"])
            verification["sample_queries"].append({
                "query": query["name"],
                "success": True,
                "results": len(list(cursor))
            })
        except Exception as e:
            verification["sample_queries"].append({
                "query": query["name"],
                "success": False,
                "error": str(e)
            })
    
    return verification


def print_schema_info(db_name: str):
    """Print detailed schema information"""
    print(f"\nIREC Database Schema: {db_name}")
    print("="*60)
    print("\nCollections:")
    print("-"*60)
    
    collections_info = {
        "documents": "Full arXiv papers with metadata",
        "chunks": "Semantic chunks created by Jina (TRUE late chunking)",
        "dimensional_scores": "WHERE, WHAT, CONVEYANCE, TIME scores",
        "document_similarities": "Document-level similarity edges",
        "chunk_similarities": "Chunk-level similarity edges",
        "citations": "Citation network for FRAME discovery",
        "implementations": "GitHub/implementation tracking",
        "processing_log": "Processing history and metadata"
    }
    
    for coll, desc in collections_info.items():
        print(f"  • {coll}: {desc}")
    
    print("\nKey Features:")
    print("-"*60)
    print("  ✓ TRUE late chunking (Jina creates chunks from full docs)")
    print("  ✓ Full arXiv metadata integration")
    print("  ✓ Dimensional scoring for Information Reconstructionism")
    print("  ✓ Citation network for FRAME discovery")
    print("  ✓ Implementation tracking for CONVEYANCE validation")
    print("  ✓ Graph structure for advanced queries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Setup IREC database schema with TRUE late chunking support"
    )
    parser.add_argument("--db-name", default="irec_late_chunking",
                        help="Database name (default: irec_late_chunking)")
    parser.add_argument("--host", default="localhost",
                        help="ArangoDB host (default: localhost)")
    parser.add_argument("--port", type=int, default=8529,
                        help="ArangoDB port (default: 8529)")
    parser.add_argument("--username", default="root",
                        help="Database username (default: root)")
    parser.add_argument("--clean-start", action="store_true",
                        help="Drop existing database and start fresh")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify existing schema")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Get password
    if args.verify_only:
        password = getpass.getpass(f"Enter password for {args.username}: ")
    else:
        password = getpass.getpass(f"Enter password for {args.username}: ")
    
    if args.verify_only:
        # Just verify existing schema
        client = ArangoClient(hosts=f'http://{args.host}:{args.port}')
        db = client.db(args.db_name, username=args.username, password=password)
        verification = verify_schema(db)
        
        print("\nSchema Verification:")
        print(f"Collections: {len(verification['collections'])}")
        for coll, info in verification['collections'].items():
            print(f"  - {coll}: {info['count']} documents, {info['indexes']} indexes")
        print(f"Graph exists: {verification['graph']}")
        
        if verification['sample_queries']:
            print("\nSample queries:")
            for q in verification['sample_queries']:
                status = "✓" if q['success'] else "✗"
                print(f"  {status} {q['query']}")
    else:
        # Create schema
        db = create_irec_schema(
            db_name=args.db_name,
            host=args.host,
            port=args.port,
            username=args.username,
            password=password,
            clean_start=args.clean_start
        )
        
        # Verify
        verification = verify_schema(db)
        print(f"\n✅ Database '{args.db_name}' created successfully!")
        print(f"Collections: {len(verification['collections'])}")
        print(f"Graph: {'✓' if verification['graph'] else '✗'}")
        
        # Print schema info
        print_schema_info(args.db_name)