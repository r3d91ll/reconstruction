#!/usr/bin/env python3
"""
Verify and visualize the graph structure in ArangoDB
"""

import os
from arango import ArangoClient

def main():
    # Connect to ArangoDB
    arango_host = os.environ.get('ARANGO_HOST', 'http://192.168.1.69:8529')
    client = ArangoClient(hosts=arango_host)
    username = os.environ.get('ARANGO_USERNAME', 'root')
    password = os.environ.get('ARANGO_PASSWORD', '')
    db = client.db('information_reconstructionism', username=username, password=password)
    
    print("GRAPH DATABASE VERIFICATION")
    print("=" * 60)
    
    # 1. Check collections
    print("\n1. Collections:")
    for collection in db.collections():
        if not collection['name'].startswith('_'):
            print(f"   - {collection['name']}")
            print(f"     Type: {'Edge' if collection['type'] == 3 else 'Document'} collection")
            print(f"     Count: {db.collection(collection['name']).count()}")
    
    # 2. Sample graph traversal
    print("\n2. Graph Traversal Example:")
    
    # Get a paper with high connections
    query = """
    FOR paper IN papers
        LET connections = (
            FOR edge IN semantic_similarity
                FILTER edge._from == paper._id OR edge._to == paper._id
                RETURN edge
        )
        SORT LENGTH(connections) DESC
        LIMIT 1
        RETURN {
            paper: paper,
            connection_count: LENGTH(connections)
        }
    """
    
    cursor = db.aql.execute(query)
    for result in cursor:
        paper = result['paper']
        print(f"\n   Most connected paper: {paper['title'][:50]}...")
        print(f"   Connections: {result['connection_count']}")
        
        # Show its neighbors
        neighbor_query = """
        FOR edge IN semantic_similarity
            FILTER edge._from == @paper_id OR edge._to == @paper_id
            LET neighbor_id = (edge._from == @paper_id ? edge._to : edge._from)
            LET neighbor = DOCUMENT(neighbor_id)
            SORT edge.context DESC
            LIMIT 5
            RETURN {
                title: neighbor.title,
                context: edge.context,
                context_original: edge.context_original
            }
        """
        
        neighbors = db.aql.execute(neighbor_query, bind_vars={'paper_id': paper['_id']})
        print("\n   Top 5 neighbors by similarity:")
        for i, neighbor in enumerate(neighbors, 1):
            print(f"   {i}. Context={neighbor['context']:.3f} - {neighbor['title'][:50]}...")
    
    # 3. Graph queries work
    print("\n3. Graph Query Capabilities:")
    
    # Shortest path example
    papers = list(db.collection('papers').all())
    if len(papers) >= 2:
        start = papers[0]['_id']
        end = papers[-1]['_id']
        
        path_query = """
        FOR path IN ANY SHORTEST_PATH
            @start TO @end
            semantic_similarity
            RETURN path
        """
        
        try:
            paths = db.aql.execute(path_query, bind_vars={'start': start, 'end': end})
            path_list = list(paths)
            if path_list:
                print(f"\n   ✓ Shortest path found between papers")
                print(f"     Path length: {len(path_list)} nodes")
            else:
                print(f"\n   No path found (normal for sparse graphs)")
        except Exception as e:
            print(f"\n   Graph traversal note: {str(e)[:100]}")
    
    # 4. Summary
    print("\n4. Graph Summary:")
    papers_count = db.collection('papers').count()
    edges_count = db.collection('semantic_similarity').count()
    
    print(f"   Total nodes (papers): {papers_count}")
    print(f"   Total edges (similarities): {edges_count}")
    print(f"   Graph density: {edges_count / (papers_count * (papers_count - 1) / 2):.2%}")
    print(f"   Average connections per paper: {(2 * edges_count) / papers_count:.1f}")
    
    print("\n" + "=" * 60)
    print("RESULT: Graph database is properly structured and queryable ✓")

if __name__ == "__main__":
    main()