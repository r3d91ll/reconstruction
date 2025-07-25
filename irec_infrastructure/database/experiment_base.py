"""
Base class for experiments using pre-computed infrastructure

All experiments inherit from this to get instant access to:
- Pre-computed embeddings
- Similarity matrices  
- ISNE embeddings
- Implementation tracking data
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pickle

from arango import ArangoClient
from arango.database import StandardDatabase


class ExperimentBase:
    """
    Base class for all experiments.
    
    Provides instant access to pre-computed infrastructure without
    any embedding generation or similarity computation.
    
    Example:
        class Experiment1(ExperimentBase):
            def run(self):
                # All data is already available!
                papers = self.get_papers()
                similarities = self.get_paper_similarities()
                
                # Focus on hypothesis testing
                results = self.test_multiplicative_model(papers, similarities)
    """
    
    def __init__(
        self,
        db_name: str = "information_reconstructionism_base",
        isne_model_path: str = "/models/isne_base.pkl",
        cache_dir: str = "./cache"
    ):
        self.db_name = db_name
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Connect to pre-populated database
        self._connect_database()
        
        # Load pre-trained ISNE model
        self._load_isne_model(isne_model_path)
        
        # Setup caching for frequent queries
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Verify infrastructure is ready
        self._verify_infrastructure()
    
    def _connect_database(self):
        """Connect to the pre-populated database."""
        client = ArangoClient(hosts='http://localhost:8529')
        self.db = client.db(self.db_name, username='root', password='')
        self.logger.info(f"Connected to database: {self.db_name}")
    
    def _load_isne_model(self, model_path: str):
        """Load pre-trained ISNE model."""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            self.isne_model = model_data
            self.logger.info(f"Loaded ISNE model from {model_path}")
        except FileNotFoundError:
            self.logger.warning("ISNE model not found. Some features may be unavailable.")
            self.isne_model = None
    
    def _verify_infrastructure(self):
        """Verify all required data is present."""
        required_collections = [
            'papers', 'paper_embeddings', 'chunks', 'chunk_embeddings',
            'paper_similarities', 'chunk_similarities', 'implementations'
        ]
        
        for collection in required_collections:
            if not self.db.has_collection(collection):
                raise RuntimeError(f"Required collection missing: {collection}")
            
            count = self.db.collection(collection).count()
            self.logger.info(f"Collection {collection}: {count} documents")
    
    # Data Access Methods - No computation needed!
    
    def get_papers(self, limit: Optional[int] = None) -> List[Dict]:
        """Get papers from database."""
        query = "FOR p IN papers RETURN p"
        if limit:
            query = f"FOR p IN papers LIMIT {limit} RETURN p"
        
        return list(self.db.aql.execute(query))
    
    def get_paper_embeddings(self, paper_ids: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Get pre-computed paper embeddings."""
        if paper_ids:
            query = """
            FOR e IN paper_embeddings
                FILTER e.paper_id IN @paper_ids
                RETURN {paper_id: e.paper_id, embedding: e.embedding}
            """
            cursor = self.db.aql.execute(query, bind_vars={'paper_ids': paper_ids})
        else:
            query = """
            FOR e IN paper_embeddings
                RETURN {paper_id: e.paper_id, embedding: e.embedding}
            """
            cursor = self.db.aql.execute(query)
        
        return {doc['paper_id']: np.array(doc['embedding']) for doc in cursor}
    
    def get_paper_similarities(self, min_similarity: float = 0.7) -> List[Dict]:
        """Get pre-computed paper similarities."""
        query = """
        FOR e IN paper_similarities
            FILTER e.similarity_score >= @min_sim
            RETURN e
        """
        return list(self.db.aql.execute(query, bind_vars={'min_sim': min_similarity}))
    
    def get_chunks_for_paper(self, paper_id: str) -> List[Dict]:
        """Get chunks for a specific paper."""
        query = """
        FOR c IN chunks
            FILTER c.paper_id == @paper_id
            SORT c.chunk_index
            RETURN c
        """
        return list(self.db.aql.execute(query, bind_vars={'paper_id': paper_id}))
    
    def get_implementations(self, paper_ids: Optional[List[str]] = None) -> List[Dict]:
        """Get implementation data."""
        if paper_ids:
            query = """
            FOR i IN implementations
                FILTER i.paper_id IN @paper_ids
                RETURN i
            """
            return list(self.db.aql.execute(query, bind_vars={'paper_ids': paper_ids}))
        else:
            query = "FOR i IN implementations RETURN i"
            return list(self.db.aql.execute(query))
    
    def get_dimensional_scores(self, paper_ids: Optional[List[str]] = None) -> Dict[str, Dict]:
        """Get pre-computed dimensional scores."""
        if paper_ids:
            query = """
            FOR d IN dimensional_scores
                FILTER d.paper_id IN @paper_ids
                RETURN d
            """
            cursor = self.db.aql.execute(query, bind_vars={'paper_ids': paper_ids})
        else:
            query = "FOR d IN dimensional_scores RETURN d"
            cursor = self.db.aql.execute(query)
        
        return {doc['paper_id']: doc for doc in cursor}
    
    def get_citation_network(self) -> List[Dict]:
        """Get citation edges."""
        query = "FOR c IN citation_network RETURN c"
        return list(self.db.aql.execute(query))
    
    # ISNE Methods
    
    def get_isne_embeddings(self, node_ids: List[str]) -> np.ndarray:
        """Get ISNE embeddings for nodes."""
        if not self.isne_model:
            raise RuntimeError("ISNE model not loaded")
        
        # In real implementation, this would use the ISNE model's transform method
        # For now, return placeholder
        return np.random.randn(len(node_ids), 100)
    
    # Utility Methods
    
    def run_experiment(self):
        """Override this method in subclasses to implement experiment logic."""
        raise NotImplementedError("Subclasses must implement run_experiment()")
    
    def save_results(self, results: Dict, filename: str):
        """Save experiment results."""
        output_path = self.cache_dir / filename
        with open(output_path, 'w') as f:
            import json
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to {output_path}")