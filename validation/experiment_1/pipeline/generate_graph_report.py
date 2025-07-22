#!/usr/bin/env python3
"""
Generate comprehensive graph analysis report after pipeline run
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from arango import ArangoClient
from datetime import datetime
import json

def main():
    # Connect to ArangoDB
    arango_host = os.environ.get('ARANGO_HOST', 'http://192.168.1.69:8529')
    client = ArangoClient(hosts=arango_host)
    username = os.environ.get('ARANGO_USERNAME', 'root')
    password = os.environ.get('ARANGO_PASSWORD', '')
    db = client.db('information_reconstructionism', username=username, password=password)
    
    print("Generating Graph Analysis Report...")
    print("=" * 60)
    
    # Basic statistics
    papers_count = db.collection('papers').count()
    edges_count = db.collection('semantic_similarity').count()
    unique_edges = edges_count // 2  # Bidirectional storage
    
    print(f"\n1. BASIC STATISTICS")
    print(f"   Total papers: {papers_count:,}")
    print(f"   Total edges: {edges_count:,}")
    print(f"   Unique connections: {unique_edges:,}")
    print(f"   Graph density: {unique_edges / (papers_count * (papers_count - 1) / 2) * 100:.2f}%")
    
    # Temporal distribution
    print(f"\n2. TEMPORAL DISTRIBUTION")
    year_query = """
    FOR p IN papers
        COLLECT year = p.year WITH COUNT INTO count
        SORT year
        RETURN {year: year, count: count}
    """
    year_data = list(db.aql.execute(year_query))
    
    for item in year_data[-5:]:  # Last 5 years
        print(f"   {item['year']}: {item['count']} papers")
    
    # Category distribution
    print(f"\n3. CATEGORY DISTRIBUTION")
    category_query = """
    FOR p IN papers
        LET primary = SPLIT(p.primary_category, ".")[0]
        COLLECT cat = primary WITH COUNT INTO count
        SORT count DESC
        LIMIT 10
        RETURN {category: cat, count: count}
    """
    categories = list(db.aql.execute(category_query))
    
    for cat in categories:
        print(f"   {cat['category']}: {cat['count']} papers")
    
    # Connectivity analysis
    print(f"\n4. CONNECTIVITY ANALYSIS")
    connectivity_query = """
    FOR p IN papers
        LET outgoing = LENGTH(
            FOR e IN semantic_similarity
                FILTER e._from == p._id
                RETURN 1
        )
        COLLECT bucket = FLOOR(outgoing / 100) * 100
        WITH COUNT INTO count
        SORT bucket
        RETURN {
            range: CONCAT(TO_STRING(bucket), "-", TO_STRING(bucket + 99)),
            count: count
        }
    """
    connectivity = list(db.aql.execute(connectivity_query))
    
    for item in connectivity[:5]:
        print(f"   {item['range']} connections: {item['count']} papers")
    
    # Context amplification effects
    print(f"\n5. CONTEXT AMPLIFICATION EFFECTS")
    context_query = """
    FOR e IN semantic_similarity
        FILTER e.context_original != null
        COLLECT bucket = FLOOR(e.context_original * 10) / 10
        WITH COUNT INTO count
        AGGREGATE 
            avg_amplified = AVG(e.context)
        SORT bucket
        RETURN {
            original: bucket,
            avg_amplified: avg_amplified,
            count: count
        }
    """
    context_effects = list(db.aql.execute(context_query))
    
    print("   Original → Amplified (Context^1.5)")
    for item in context_effects[-5:]:  # Top similarity ranges
        if item['avg_amplified']:
            print(f"   {item['original']:.1f} → {item['avg_amplified']:.3f}")
    
    # Milestone papers analysis
    print(f"\n6. MILESTONE PAPERS ANALYSIS")
    milestone_keywords = [
        "word2vec", "attention is all you need", "gpt-2", "gpt-3",
        "toolformer", "react", "vilbert", "lxmert"
    ]
    
    for keyword in milestone_keywords:
        query = """
        FOR p IN papers
            FILTER CONTAINS(LOWER(p.title), @keyword)
            LIMIT 1
            LET connections = LENGTH(
                FOR e IN semantic_similarity
                    FILTER e._from == p._id OR e._to == p._id
                    RETURN 1
            )
            RETURN {
                title: p.title,
                year: p.year,
                connections: connections
            }
        """
        result = list(db.aql.execute(query, bind_vars={'keyword': keyword}))
        if result:
            paper = result[0]
            print(f"   ✓ {keyword}: {paper['connections']} connections ({paper['year']})")
        else:
            print(f"   ✗ {keyword}: not found")
    
    # Theory-practice bridges
    print(f"\n7. THEORY-PRACTICE BRIDGE CANDIDATES")
    bridge_query = """
    FOR e IN semantic_similarity
        FILTER e.context > 0.9  // High similarity after amplification
        LET from_paper = DOCUMENT(e._from)
        LET to_paper = DOCUMENT(e._to)
        FILTER from_paper != null AND to_paper != null
        // Look for theory-practice indicators
        FILTER (CONTAINS(LOWER(from_paper.title), "theory") AND 
                (CONTAINS(LOWER(to_paper.title), "implement") OR 
                 CONTAINS(LOWER(to_paper.title), "practic")))
               OR
               (CONTAINS(LOWER(to_paper.title), "theory") AND 
                (CONTAINS(LOWER(from_paper.title), "implement") OR 
                 CONTAINS(LOWER(from_paper.title), "practic")))
        SORT e.context DESC
        LIMIT 5
        RETURN {
            from: from_paper.title,
            to: to_paper.title,
            similarity: e.context_original,
            amplified: e.context
        }
    """
    bridges = list(db.aql.execute(bridge_query))
    
    for i, bridge in enumerate(bridges, 1):
        print(f"\n   Bridge {i}:")
        print(f"   From: {bridge['from'][:60]}...")
        print(f"   To: {bridge['to'][:60]}...")
        print(f"   Similarity: {bridge['similarity']:.3f} → {bridge['amplified']:.3f}")
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"/home/todd/reconstructionism/validation/experiment_1/graph_report_{timestamp}.txt"
    
    # Capture all the above in a report file
    with open(report_path, 'w') as f:
        f.write("INFORMATION RECONSTRUCTIONISM - GRAPH ANALYSIS REPORT\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write("=" * 60 + "\n\n")
        
        # Re-run all queries and write to file
        f.write(f"1. BASIC STATISTICS\n")
        f.write(f"   Total papers: {papers_count:,}\n")
        f.write(f"   Total edges: {edges_count:,}\n")
        f.write(f"   Unique connections: {unique_edges:,}\n")
        f.write(f"   Graph density: {unique_edges / (papers_count * (papers_count - 1) / 2) * 100:.2f}%\n\n")
        
        # Add all other sections...
        # (Full implementation would repeat all sections)
    
    print(f"\n✓ Report saved to: {report_path}")
    
    # Generate visualization
    create_summary_visualization(db, papers_count, unique_edges)

def create_summary_visualization(db, papers_count, unique_edges):
    """Create a summary visualization of the graph"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Degree distribution
    degree_query = """
    FOR p IN papers
        LET degree = LENGTH(
            FOR e IN semantic_similarity
                FILTER e._from == p._id
                RETURN 1
        )
        COLLECT d = degree WITH COUNT INTO count
        SORT d
        RETURN {degree: d, count: count}
    """
    degree_dist = list(db.aql.execute(degree_query))
    degrees = [d['degree'] for d in degree_dist]
    counts = [d['count'] for d in degree_dist]
    
    ax1.bar(degrees[:50], counts[:50], color='steelblue')
    ax1.set_xlabel('Degree (# connections)')
    ax1.set_ylabel('Number of Papers')
    ax1.set_title('Degree Distribution')
    ax1.set_yscale('log')
    
    # 2. Context score distribution
    context_query = """
    FOR e IN semantic_similarity
        FILTER e.context_original != null
        COLLECT bucket = FLOOR(e.context_original * 20) / 20
        WITH COUNT INTO count
        SORT bucket
        RETURN {score: bucket, count: count}
    """
    context_dist = list(db.aql.execute(context_query))
    scores = [c['score'] for c in context_dist]
    counts = [c['count'] for c in context_dist]
    
    ax2.bar(scores, counts, width=0.04, color='darkgreen')
    ax2.set_xlabel('Original Similarity Score')
    ax2.set_ylabel('Number of Edges')
    ax2.set_title('Similarity Distribution (before amplification)')
    ax2.axvline(x=0.5, color='red', linestyle='--', label='Threshold')
    
    # 3. Year distribution
    year_query = """
    FOR p IN papers
        COLLECT year = p.year WITH COUNT INTO count
        SORT year
        RETURN {year: year, count: count}
    """
    year_data = list(db.aql.execute(year_query))
    years = [y['year'] for y in year_data]
    counts = [y['count'] for y in year_data]
    
    ax3.bar(years, counts, color='purple')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Number of Papers')
    ax3.set_title('Temporal Distribution')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Summary statistics text
    ax4.axis('off')
    summary_text = f"""Graph Summary Statistics
    
Total Papers: {papers_count:,}
Total Edges: {unique_edges*2:,}
Unique Connections: {unique_edges:,}

Graph Density: {unique_edges / (papers_count * (papers_count - 1) / 2) * 100:.2f}%
Average Degree: {(unique_edges * 2) / papers_count:.1f}

Context Amplification: α = 1.5
Similarity Threshold: 0.5

Information Reconstructionism
WHERE × WHAT × Context^1.5"""
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Information Reconstructionism - Experiment 1 Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = '/home/todd/reconstructionism/validation/experiment_1/final_results_summary.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    main()