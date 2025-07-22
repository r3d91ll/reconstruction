#!/usr/bin/env python3
"""
Quick check for edge duplication issue
"""

import os
from arango import ArangoClient

# Connect
arango_host = os.environ.get('ARANGO_HOST', 'http://192.168.1.69:8529')
client = ArangoClient(hosts=arango_host)
username = os.environ.get('ARANGO_USERNAME', 'root')
password = os.environ.get('ARANGO_PASSWORD', '')
db = client.db('information_reconstructionism', username=username, password=password)

print("Quick edge analysis...")

# 1. Total counts
papers_count = db.collection('papers').count()
edges_count = db.collection('semantic_similarity').count()

print(f"\nTotal papers: {papers_count:,}")
print(f"Total edges: {edges_count:,}")

# 2. Sample a few edges to check structure
print("\nSampling edges to check for bidirectionality...")
sample_query = """
FOR edge IN semantic_similarity
    LIMIT 5
    RETURN {
        from: edge._from,
        to: edge._to,
        context: edge.context
    }
"""

samples = list(db.aql.execute(sample_query))
print("\nSample edges:")
for s in samples:
    print(f"  {s['from']} → {s['to']} (context: {s['context']:.3f})")

# 3. Check if reverse edges exist
print("\nChecking if edges are stored bidirectionally...")
for s in samples[:2]:  # Check first 2
    reverse_query = """
    FOR edge IN semantic_similarity
        FILTER edge._from == @to AND edge._to == @from
        RETURN edge
    """
    reverse = list(db.aql.execute(reverse_query, bind_vars={'from': s['from'], 'to': s['to']}))
    if reverse:
        print(f"  Found reverse of {s['from']} → {s['to']}")
        print(f"    Original context: {s['context']:.3f}")
        print(f"    Reverse context: {reverse[0]['context']:.3f}")

# 4. Expected calculations
expected_comparisons = papers_count * (papers_count - 1) // 2
expected_edges_one_way = int(expected_comparisons * 0.97)  # ~97% > 0.5
expected_edges_bidirectional = expected_edges_one_way * 2

print(f"\nExpected comparisons: {expected_comparisons:,}")
print(f"Expected edges (one-way): {expected_edges_one_way:,}")
print(f"Expected edges (bidirectional): {expected_edges_bidirectional:,}")
print(f"Actual edges: {edges_count:,}")

if abs(edges_count - expected_edges_bidirectional) < 100000:
    print("\n✓ Edges are stored bidirectionally (A→B and B→A)")
else:
    print("\n? Edge count doesn't match expected patterns")