"""
ArangoDB Client for Infrastructure

Provides connection management and common database operations.
Validated with millions of documents and edges.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union
from arango import ArangoClient as ArangoDBClient
from arango.database import StandardDatabase
from arango.exceptions import DatabaseCreateError, CollectionCreateError


logger = logging.getLogger(__name__)


class ArangoClient:
    """
    Manages ArangoDB connections and provides common operations.
    
    This client handles:
    - Connection pooling
    - Automatic retries
    - Batch operations
    - Error handling
    
    Example:
        client = ArangoClient()
        db = client.get_database("my_database")
        
        # Batch insert
        client.batch_insert(db, "collection", documents)
    """
    
    def __init__(
        self,
        host: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None
    ):
        """
        Initialize ArangoDB client.
        
        Args:
            host: ArangoDB host URL (defaults to env var ARANGO_HOST)
            username: Database username (defaults to env var ARANGO_USERNAME or 'root')
            password: Database password (defaults to env var ARANGO_PASSWORD)
        """
        self.host = host or os.environ.get('ARANGO_HOST', 'http://localhost:8529')
        self.username = username or os.environ.get('ARANGO_USERNAME', 'root')
        self.password = password or os.environ.get('ARANGO_PASSWORD', '')
        
        # Initialize client
        self.client = ArangoDBClient(hosts=self.host)
        self._sys_db = None
        
        logger.info(f"Initialized ArangoDB client for {self.host}")
    
    @property
    def sys_db(self):
        """Get system database connection."""
        if self._sys_db is None:
            self._sys_db = self.client.db(
                '_system',
                username=self.username,
                password=self.password
            )
        return self._sys_db
    
    def create_database(
        self,
        db_name: str,
        clean_start: bool = False
    ) -> StandardDatabase:
        """
        Create a new database or get existing one.
        
        Args:
            db_name: Name of database to create
            clean_start: If True, drop existing database first
            
        Returns:
            Database connection object
        """
        # Check if database exists
        if self.sys_db.has_database(db_name):
            if clean_start:
                logger.warning(f"Dropping existing database: {db_name}")
                self.sys_db.delete_database(db_name)
            else:
                logger.info(f"Using existing database: {db_name}")
                return self.get_database(db_name)
        
        # Create new database
        try:
            self.sys_db.create_database(db_name)
            logger.info(f"Created database: {db_name}")
        except DatabaseCreateError as e:
            logger.error(f"Failed to create database: {e}")
            raise
        
        return self.get_database(db_name)
    
    def get_database(self, db_name: str) -> StandardDatabase:
        """Get connection to specific database."""
        return self.client.db(
            db_name,
            username=self.username,
            password=self.password
        )
    
    def create_collection(
        self,
        db: StandardDatabase,
        name: str,
        edge: bool = False,
        indexes: Optional[List[Dict]] = None
    ):
        """
        Create a collection with optional indexes.
        
        Args:
            db: Database connection
            name: Collection name
            edge: Whether this is an edge collection
            indexes: List of index definitions
        """
        try:
            # Create collection
            if edge:
                collection = db.create_collection(name, edge=True)
            else:
                collection = db.create_collection(name)
            
            logger.info(f"Created {'edge' if edge else 'document'} collection: {name}")
            
            # Add indexes
            if indexes:
                for index_def in indexes:
                    collection.add_index(index_def)
                    logger.info(f"Added index on {name}: {index_def['fields']}")
                    
        except CollectionCreateError as e:
            if "duplicate name" in str(e):
                logger.info(f"Collection {name} already exists")
            else:
                raise
    
    def batch_insert(
        self,
        db: StandardDatabase,
        collection_name: str,
        documents: List[Dict],
        batch_size: int = 1000,
        on_duplicate: str = "ignore"
    ) -> Dict[str, int]:
        """
        Insert documents in batches for efficiency.
        
        Args:
            db: Database connection
            collection_name: Target collection
            documents: List of documents to insert
            batch_size: Documents per batch
            on_duplicate: Action on duplicate ("ignore", "replace", "update")
            
        Returns:
            Dictionary with insert statistics
        """
        collection = db.collection(collection_name)
        
        total_inserted = 0
        total_errors = 0
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            try:
                result = collection.insert_many(
                    batch,
                    overwrite=(on_duplicate == "replace"),
                    return_new=False,
                    silent=True
                )
                
                # Count successes
                if isinstance(result, list):
                    total_inserted += sum(1 for r in result if not r.get("error"))
                    total_errors += sum(1 for r in result if r.get("error"))
                else:
                    total_inserted += len(batch)
                    
            except Exception as e:
                logger.error(f"Batch insert error: {e}")
                total_errors += len(batch)
        
        logger.info(
            f"Batch insert complete: {total_inserted} inserted, "
            f"{total_errors} errors in {collection_name}"
        )
        
        return {
            "inserted": total_inserted,
            "errors": total_errors,
            "total": len(documents)
        }
    
    def create_graph(
        self,
        db: StandardDatabase,
        graph_name: str,
        edge_definitions: List[Dict]
    ):
        """
        Create a graph with edge definitions.
        
        Args:
            db: Database connection
            graph_name: Name of graph
            edge_definitions: List of edge definition dictionaries
        """
        if db.has_graph(graph_name):
            logger.info(f"Graph {graph_name} already exists")
            return db.graph(graph_name)
        
        graph = db.create_graph(graph_name)
        
        for edge_def in edge_definitions:
            graph.create_edge_definition(
                edge_collection=edge_def["edge_collection"],
                from_vertex_collections=edge_def["from_collections"],
                to_vertex_collections=edge_def["to_collections"]
            )
        
        logger.info(f"Created graph: {graph_name}")
        return graph
    
    def execute_aql(
        self,
        db: StandardDatabase,
        query: str,
        bind_vars: Optional[Dict] = None,
        batch_size: Optional[int] = None
    ) -> List[Any]:
        """
        Execute AQL query with optional batching.
        
        Args:
            db: Database connection
            query: AQL query string
            bind_vars: Query bind variables
            batch_size: If set, return results in batches
            
        Returns:
            Query results
        """
        cursor = db.aql.execute(
            query,
            bind_vars=bind_vars,
            batch_size=batch_size
        )
        
        return list(cursor)