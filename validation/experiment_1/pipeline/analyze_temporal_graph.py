#!/usr/bin/env python3
"""
Analyze temporal distribution of edges and check for duplicates
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from arango import ArangoClient
from collections import defaultdict
import pandas as pd

def main():
    # Connect to ArangoDB
    arango_host = os.environ.get('ARANGO_HOST', 'http://192.168.1.69:8529')
    client = ArangoClient(hosts=arango_host)
    username = os.environ.get('ARANGO_USERNAME', 'root')
    password = os.environ.get('ARANGO_PASSWORD', '')
    db = client.db('information_reconstructionism', username=username, password=password)
    
    print("Analyzing temporal graph structure...")
    
    # 1. Check for duplicate edges
    print("\n1. Checking for duplicate edges...")
    dup_query = """
    FOR edge IN semantic_similarity
        COLLECT from = edge._from, to = edge._to WITH COUNT INTO count
        FILTER count > 1
        SORT count DESC
        LIMIT 10
        RETURN {
            from: from,
            to: to,
            count: count
        }
    """
    
    duplicates = list(db.aql.execute(dup_query))
    if duplicates:
        print(f"Found {len(duplicates)} duplicate edge pairs (showing top 10):")
        for dup in duplicates:
            print(f"  {dup['from']} ↔ {dup['to']}: {dup['count']} copies")
    else:
        print("No duplicate edges found")
    
    # 2. Check if edges are bidirectional
    print("\n2. Checking edge directionality...")
    bidir_query = """
    LET forward_edges = (
        FOR edge IN semantic_similarity
            RETURN CONCAT(edge._from, "->", edge._to)
    )
    
    LET reverse_count = (
        FOR edge IN semantic_similarity
            FILTER CONCAT(edge._to, "->", edge._from) IN forward_edges
            COLLECT WITH COUNT INTO count
            RETURN count
    )[0]
    
    RETURN {
        total_edges: LENGTH(semantic_similarity),
        bidirectional_pairs: reverse_count / 2
    }
    """
    
    bidir_result = next(db.aql.execute(bidir_query))
    print(f"Total edges: {bidir_result['total_edges']:,}")
    print(f"Bidirectional pairs: {bidir_result['bidirectional_pairs']:,}")
    
    # 3. Get temporal distribution
    print("\n3. Analyzing temporal distribution...")
    
    # Papers by year
    year_query = """
    FOR p IN papers
        COLLECT year = p.year WITH COUNT INTO count
        SORT year
        RETURN {year: year, count: count}
    """
    
    year_data = list(db.aql.execute(year_query))
    years = [d['year'] for d in year_data]
    paper_counts = [d['count'] for d in year_data]
    
    print("\nPapers by year:")
    for y, c in zip(years, paper_counts):
        print(f"  {y}: {c} papers")
    
    # 4. Edge distribution by year pairs
    print("\n4. Computing edge distribution across time...")
    edge_year_query = """
    FOR edge IN semantic_similarity
        LET from_paper = DOCUMENT(edge._from)
        LET to_paper = DOCUMENT(edge._to)
        FILTER from_paper != null AND to_paper != null
        COLLECT 
            from_year = from_paper.year,
            to_year = to_paper.year
        WITH COUNT INTO count
        RETURN {
            from_year: from_year,
            to_year: to_year,
            count: count
        }
    """
    
    edge_year_data = list(db.aql.execute(edge_year_query))
    
    # Create year-pair matrix
    year_set = sorted(set(years))
    year_matrix = np.zeros((len(year_set), len(year_set)))
    year_to_idx = {year: idx for idx, year in enumerate(year_set)}
    
    for item in edge_year_data:
        if item['from_year'] in year_to_idx and item['to_year'] in year_to_idx:
            i = year_to_idx[item['from_year']]
            j = year_to_idx[item['to_year']]
            year_matrix[i, j] = item['count']
    
    # 5. Analyze top-k distribution by year
    print("\n5. Top-k edge distribution by source year...")
    topk_query = """
    FOR p IN papers
        LET edge_count = LENGTH(
            FOR edge IN semantic_similarity
                FILTER edge._from == p._id OR edge._to == p._id
                RETURN 1
        )
        COLLECT year = p.year
        AGGREGATE 
            avg_edges = AVG(edge_count),
            min_edges = MIN(edge_count),
            max_edges = MAX(edge_count),
            total_edges = SUM(edge_count)
        SORT year
        RETURN {
            year: year,
            avg_edges: avg_edges,
            min_edges: min_edges,
            max_edges: max_edges,
            total_edges: total_edges
        }
    """
    
    topk_data = list(db.aql.execute(topk_query))
    
    print("\nAverage edges per paper by year:")
    for item in topk_data:
        print(f"  {item['year']}: avg={item['avg_edges']:.1f}, "
              f"min={item['min_edges']}, max={item['max_edges']}, "
              f"total={item['total_edges']:,}")
    
    # 6. Create visualizations
    print("\n6. Creating visualizations...")
    
    # Plot 1: Papers per year
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.bar(years, paper_counts)
    plt.xlabel('Year')
    plt.ylabel('Number of Papers')
    plt.title('Papers by Year')
    plt.xticks(rotation=45)
    
    # Plot 2: Year-pair heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(year_matrix, cmap='YlOrRd', aspect='auto')
    plt.colorbar(label='Number of Edges')
    plt.xlabel('To Year')
    plt.ylabel('From Year')
    plt.title('Edge Distribution Across Years')
    tick_positions = range(0, len(year_set), max(1, len(year_set)//10))
    plt.xticks(tick_positions, [year_set[i] for i in tick_positions], rotation=45)
    plt.yticks(tick_positions, [year_set[i] for i in tick_positions])
    
    # Plot 3: Average edges per paper by year
    plt.subplot(1, 3, 3)
    avg_edges_by_year = [item['avg_edges'] for item in topk_data]
    years_topk = [item['year'] for item in topk_data]
    plt.plot(years_topk, avg_edges_by_year, 'bo-')
    plt.xlabel('Year')
    plt.ylabel('Average Edges per Paper')
    plt.title('Connectivity by Year')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    output_path = '/home/todd/reconstructionism/validation/experiment_1/temporal_analysis.png'
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved visualization to: {output_path}")
    
    # 7. Diagnose the edge count issue
    print("\n7. Diagnosing edge count discrepancy...")
    
    # Expected vs actual
    total_papers = sum(paper_counts)
    expected_comparisons = total_papers * (total_papers - 1) // 2
    expected_edges = int(expected_comparisons * 0.97)  # ~97% are > 0.5 based on output
    actual_edges = bidir_result['total_edges']
    
    print(f"\nEdge count analysis:")
    print(f"  Total papers: {total_papers:,}")
    print(f"  Expected comparisons: {expected_comparisons:,}")
    print(f"  Expected edges (97% > 0.5): {expected_edges:,}")
    print(f"  Actual edges in database: {actual_edges:,}")
    print(f"  Ratio actual/expected: {actual_edges/expected_edges:.2f}")
    
    if actual_edges > expected_edges * 1.5:
        print("\n⚠️  Database has significantly more edges than expected!")
        print("  Likely cause: Bidirectional edges (A→B and B→A stored separately)")

if __name__ == "__main__":
    main()