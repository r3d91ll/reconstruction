#!/usr/bin/env python3
"""
ArangoDB setup for Information Reconstructionism
Multi-scale network: Papers → Citations → Concepts → Implementations
"""

from arango import ArangoClient
import numpy as np
from typing import Dict, List
import json


class InformationNetworkGraph:
    """
    Graph structure for Information Reconstructionism in ArangoDB
    
    Collections:
    - papers (documents): The actual papers with embeddings
    - concepts (documents): Semantic concept nodes
    - implementations (documents): Code implementations
    
    Edge Collections:
    - cites (edges): Paper → Paper citations
    - contains (edges): Paper → Concept relationships
    - implements (edges): Paper/Concept → Implementation
    - semantic_similarity (edges): Paper ↔ Paper with Context scores
    """
    
    def __init__(self, db_url='http://localhost:8529'):
        # Connect to ArangoDB
        self.client = ArangoClient(hosts=db_url)
        self.db = None
        
    def setup_database(self, db_name='information_reconstructionism'):
        """Create database and collections"""
        
        # Create database if needed
        sys_db = self.client.db('_system', username='root', password='')
        if not sys_db.has_database(db_name):
            sys_db.create_database(db_name)
            
        # Connect to our database
        self.db = self.client.db(db_name, username='root', password='')
        
        # Create vertex collections
        if not self.db.has_collection('papers'):
            self.db.create_collection('papers')
        if not self.db.has_collection('concepts'):
            self.db.create_collection('concepts')
        if not self.db.has_collection('implementations'):
            self.db.create_collection('implementations')
            
        # Create edge collections
        if not self.db.has_collection('cites'):
            self.db.create_collection('cites', edge=True)
        if not self.db.has_collection('contains'):
            self.db.create_collection('contains', edge=True)
        if not self.db.has_collection('implements'):
            self.db.create_collection('implements', edge=True)
        if not self.db.has_collection('semantic_similarity'):
            self.db.create_collection('semantic_similarity', edge=True)
            
        print(f"Database '{db_name}' setup complete")
        
    def insert_paper(self, paper: Dict):
        """Insert paper with all dimensional data"""
        
        # Extract key fields
        doc = {
            '_key': paper['arxiv_id'].replace('.', '_'),  # ArangoDB key format
            'arxiv_id': paper['arxiv_id'],
            'title': paper['title'],
            'year': paper['year'],
            'categories': paper.get('categories', []),
            
            # WHEN dimension
            'when': paper['year'],
            
            # WHERE dimension (in graph structure itself)
            'where': paper['dimensions']['WHERE'],
            
            # WHAT dimension (embeddings)
            'what': {
                'embeddings': paper['dimensions']['WHAT'].get('embeddings'),
                'abstract': paper['abstract']
            },
            
            # CONVEYANCE dimension (computed)
            'conveyance': {
                'physical_grounding_factor': paper.get('physical_grounding_factor', 0),
                'context_amplification': None  # Computed from edges
            }
        }
        
        # Insert into papers collection
        self.db.collection('papers').insert(doc)
        
    def compute_semantic_edges(self, threshold=0.7, alpha=1.5):
        """
        Compute semantic similarity edges between papers
        This is where Context^α amplification happens!
        """
        
        # AQL query to compute similarities
        query = """
        FOR p1 IN papers
            FOR p2 IN papers
                FILTER p1._key < p2._key  // Avoid duplicates
                LET similarity = COSINE_SIMILARITY(p1.what.embeddings, p2.what.embeddings)
                FILTER similarity > @threshold
                
                // Compute Context^α amplification
                LET context = similarity
                LET amplified_context = POW(context, @alpha)
                
                // Calculate edge weight (Information transfer potential)
                LET base_conveyance = (p1.conveyance.physical_grounding_factor + 
                                      p2.conveyance.physical_grounding_factor) / 2
                LET edge_weight = base_conveyance * amplified_context
                
                INSERT {
                    _from: p1._id,
                    _to: p2._id,
                    similarity: similarity,
                    context: context,
                    context_amplified: amplified_context,
                    alpha: @alpha,
                    information_transfer: edge_weight,
                    year_gap: ABS(p1.year - p2.year)
                } INTO semantic_similarity
        """
        
        self.db.aql.execute(query, bind_vars={'threshold': threshold, 'alpha': alpha})
        print(f"Computed semantic similarity edges with Context^{alpha}")
        
    def find_gravity_wells(self, min_connections=10):
        """Find papers that act as semantic gravity wells"""
        
        query = """
        FOR paper IN papers
            LET incoming = LENGTH(
                FOR v, e IN 1..1 INBOUND paper semantic_similarity
                    FILTER e.information_transfer > 0.5
                    RETURN 1
            )
            LET outgoing = LENGTH(
                FOR v, e IN 1..1 OUTBOUND paper semantic_similarity
                    FILTER e.information_transfer > 0.5
                    RETURN 1
            )
            LET total_connections = incoming + outgoing
            FILTER total_connections >= @min_connections
            
            SORT total_connections DESC
            RETURN {
                paper: paper.title,
                year: paper.year,
                arxiv_id: paper.arxiv_id,
                connections: total_connections,
                incoming: incoming,
                outgoing: outgoing,
                categories: paper.categories
            }
        """
        
        cursor = self.db.aql.execute(query, bind_vars={'min_connections': min_connections})
        return list(cursor)
        
    def trace_concept_evolution(self, concept_keywords: List[str]):
        """Trace how concepts evolve through the network over time"""
        
        query = """
        FOR paper IN papers
            FILTER CONTAINS(LOWER(paper.title), LOWER(@keyword)) OR 
                   CONTAINS(LOWER(paper.what.abstract), LOWER(@keyword))
            
            // Find semantic neighbors
            LET neighbors = (
                FOR v, e IN 1..2 ANY paper semantic_similarity
                    FILTER e.information_transfer > 0.3
                    RETURN {
                        neighbor: v,
                        distance: LENGTH_OF_PATH(),
                        transfer_strength: e.information_transfer,
                        year: v.year
                    }
            )
            
            SORT paper.year
            RETURN {
                paper: paper.title,
                year: paper.year,
                arxiv_id: paper.arxiv_id,
                neighbor_count: LENGTH(neighbors),
                avg_transfer: AVG(neighbors[*].transfer_strength),
                year_spread: MAX(neighbors[*].year) - MIN(neighbors[*].year)
            }
        """
        
        results = []
        for keyword in concept_keywords:
            cursor = self.db.aql.execute(query, bind_vars={'keyword': keyword})
            results.extend(list(cursor))
            
        return results
        
    def get_network_stats(self):
        """Get overall network statistics"""
        
        stats = {
            'papers': self.db.collection('papers').count(),
            'semantic_edges': self.db.collection('semantic_similarity').count(),
            'citations': self.db.collection('cites').count() if self.db.has_collection('cites') else 0
        }
        
        # Average information transfer
        query = """
        FOR e IN semantic_similarity
            RETURN e.information_transfer
        """
        cursor = self.db.aql.execute(query)
        transfers = list(cursor)
        
        if transfers:
            stats['avg_information_transfer'] = np.mean(transfers)
            stats['max_information_transfer'] = np.max(transfers)
            stats['min_information_transfer'] = np.min(transfers)
            
        return stats


def demonstrate_framework():
    """Demonstrate the framework with ArangoDB"""
    
    print("=== ARANGODB FOR INFORMATION RECONSTRUCTIONISM ===\n")
    
    print("Benefits of using ArangoDB:")
    print("1. Native graph operations for semantic networks")
    print("2. Built-in vector similarity (COSINE_SIMILARITY)")
    print("3. Multi-scale analysis (papers → concepts → implementations)")
    print("4. Temporal queries along citation paths")
    print("5. Context^α computation in AQL queries")
    print("\nGraph Structure:")
    print("- Nodes: Papers, Concepts, Implementations")
    print("- Edges: Semantic similarity with Context^α weights")
    print("         Citations with temporal ordering")
    print("         Contains/Implements relationships")
    
    print("\nKey Queries Enabled:")
    print("- Find gravity wells (highly connected papers)")
    print("- Trace concept evolution over time")
    print("- Measure information transfer potential")
    print("- Discover theory-practice bridges")
    
    print("\nTo use:")
    print("1. Start ArangoDB locally")
    print("2. Generate embeddings for papers")
    print("3. Load into graph with semantic edges")
    print("4. Run network analysis queries")


if __name__ == "__main__":
    demonstrate_framework()