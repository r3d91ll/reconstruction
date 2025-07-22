#!/usr/bin/env python3
"""
Create temporal visualization of paper connections
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from arango import ArangoClient

def main():
    # Connect to ArangoDB
    arango_host = os.environ.get('ARANGO_HOST', 'http://192.168.1.69:8529')
    client = ArangoClient(hosts=arango_host)
    username = os.environ.get('ARANGO_USERNAME', 'root')
    password = os.environ.get('ARANGO_PASSWORD', '')
    db = client.db('information_reconstructionism', username=username, password=password)
    
    print("Creating temporal visualizations...")
    
    # 1. Get papers by year
    year_query = """
    FOR p IN papers
        COLLECT year = p.year WITH COUNT INTO count
        SORT year
        RETURN {year: year, count: count}
    """
    
    year_data = list(db.aql.execute(year_query))
    years = [d['year'] for d in year_data]
    paper_counts = [d['count'] for d in year_data]
    
    # 2. Get average connections per paper by year
    connections_query = """
    FOR p IN papers
        LET connections = LENGTH(
            FOR edge IN semantic_similarity
                FILTER edge._from == p._id
                RETURN 1
        )
        COLLECT year = p.year 
        AGGREGATE 
            avg_connections = AVG(connections),
            min_connections = MIN(connections),
            max_connections = MAX(connections)
        SORT year
        RETURN {
            year: year,
            avg: avg_connections,
            min: min_connections,
            max: max_connections
        }
    """
    
    conn_data = list(db.aql.execute(connections_query))
    
    # 3. Sample year-to-year connections
    print("Computing year-to-year connection matrix...")
    year_matrix_query = """
    // Sample 10% of edges for speed
    FOR edge IN semantic_similarity
        FILTER RAND() < 0.1
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
            count: count * 10  // Scale back up
        }
    """
    
    year_connections = list(db.aql.execute(year_matrix_query))
    
    # Build year matrix
    year_set = sorted(set(years))
    year_matrix = np.zeros((len(year_set), len(year_set)))
    year_to_idx = {year: idx for idx, year in enumerate(year_set)}
    
    for item in year_connections:
        if item['from_year'] in year_to_idx and item['to_year'] in year_to_idx:
            i = year_to_idx[item['from_year']]
            j = year_to_idx[item['to_year']]
            year_matrix[i, j] = item['count']
    
    # 4. Get context score distribution by year gap
    gap_query = """
    // Sample for speed
    FOR edge IN semantic_similarity
        FILTER RAND() < 0.01
        LET from_paper = DOCUMENT(edge._from)
        LET to_paper = DOCUMENT(edge._to)
        FILTER from_paper != null AND to_paper != null
        LET year_gap = ABS(from_paper.year - to_paper.year)
        COLLECT gap = year_gap
        AGGREGATE avg_context = AVG(edge.context)
        SORT gap
        FILTER gap <= 10
        RETURN {
            gap: gap,
            avg_context: avg_context
        }
    """
    
    gap_data = list(db.aql.execute(gap_query))
    
    # Create visualizations
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Papers per year
    ax1 = plt.subplot(2, 3, 1)
    bars = ax1.bar(years, paper_counts, color='steelblue')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Number of Papers', fontsize=12)
    ax1.set_title('Papers by Year', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, paper_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Average connections per paper by year
    ax2 = plt.subplot(2, 3, 2)
    conn_years = [d['year'] for d in conn_data]
    avg_conns = [d['avg'] for d in conn_data]
    min_conns = [d['min'] for d in conn_data]
    max_conns = [d['max'] for d in conn_data]
    
    ax2.plot(conn_years, avg_conns, 'o-', color='darkgreen', linewidth=2, markersize=8, label='Average')
    ax2.fill_between(conn_years, min_conns, max_conns, alpha=0.2, color='green', label='Min-Max range')
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Connections per Paper', fontsize=12)
    ax2.set_title('Average Connectivity by Year', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Year-to-year heatmap
    ax3 = plt.subplot(2, 3, 3)
    im = ax3.imshow(year_matrix, cmap='YlOrRd', aspect='auto')
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Number of Connections', fontsize=10)
    ax3.set_xlabel('To Year', fontsize=12)
    ax3.set_ylabel('From Year', fontsize=12)
    ax3.set_title('Connections Between Years', fontsize=14, fontweight='bold')
    
    # Set ticks for every few years
    tick_positions = range(0, len(year_set), max(1, len(year_set)//10))
    ax3.set_xticks(tick_positions)
    ax3.set_xticklabels([year_set[i] for i in tick_positions], rotation=45)
    ax3.set_yticks(tick_positions)
    ax3.set_yticklabels([year_set[i] for i in tick_positions])
    
    # Plot 4: Context score by year gap
    ax4 = plt.subplot(2, 3, 4)
    if gap_data:
        gaps = [d['gap'] for d in gap_data]
        contexts = [d['avg_context'] for d in gap_data]
        ax4.plot(gaps, contexts, 'o-', color='purple', linewidth=2, markersize=8)
        ax4.set_xlabel('Year Gap Between Papers', fontsize=12)
        ax4.set_ylabel('Average Context Score', fontsize=12)
        ax4.set_title('Similarity vs Time Distance', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0.5, 0.8)
    
    # Plot 5: Distribution of papers across years (pie chart)
    ax5 = plt.subplot(2, 3, 5)
    # Group years into ranges
    year_ranges = {
        '2020-2021': sum(c for y, c in zip(years, paper_counts) if 2020 <= y <= 2021),
        '2022': sum(c for y, c in zip(years, paper_counts) if y == 2022),
        '2023': sum(c for y, c in zip(years, paper_counts) if y == 2023),
        '2024': sum(c for y, c in zip(years, paper_counts) if y == 2024),
        '2025': sum(c for y, c in zip(years, paper_counts) if y == 2025),
        'Other': sum(c for y, c in zip(years, paper_counts) if y < 2020 or y > 2025)
    }
    
    colors = plt.cm.Set3(range(len(year_ranges)))
    wedges, texts, autotexts = ax5.pie(year_ranges.values(), labels=year_ranges.keys(), 
                                       autopct='%1.1f%%', colors=colors)
    ax5.set_title('Paper Distribution by Year Range', fontsize=14, fontweight='bold')
    
    # Plot 6: Summary statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    total_papers = sum(paper_counts)
    total_edges = db.collection('semantic_similarity').count()
    avg_edges_per_paper = total_edges / (2 * total_papers)  # Divide by 2 for bidirectional
    
    summary_text = f"""Summary Statistics
    
Total Papers: {total_papers:,}
Total Connections: {total_edges:,}
Bidirectional Edges: Yes
Unique Connections: {total_edges//2:,}

Average Edges per Paper: {avg_edges_per_paper:.0f}
Density: {(total_edges/2) / (total_papers*(total_papers-1)/2) * 100:.1f}%

Year Range: {min(years)} - {max(years)}
Most Papers: {max(paper_counts)} ({years[paper_counts.index(max(paper_counts))]})
Least Papers: {min(paper_counts)} ({years[paper_counts.index(min(paper_counts))]})
"""
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Temporal Analysis of Paper Similarity Network', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save the figure
    output_path = '/home/todd/reconstructionism/validation/experiment_1/temporal_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")
    plt.close()
    
    # Create a second figure for detailed year matrix
    plt.figure(figsize=(12, 10))
    im = plt.imshow(year_matrix, cmap='YlOrRd', aspect='auto', origin='lower')
    plt.colorbar(im, label='Number of Connections')
    
    # Set ticks
    tick_positions = range(0, len(year_set), max(1, len(year_set)//10))
    plt.xticks(tick_positions, [year_set[i] for i in tick_positions], rotation=45)
    plt.yticks(tick_positions, [year_set[i] for i in tick_positions])
    
    plt.xlabel('To Year', fontsize=12)
    plt.ylabel('From Year', fontsize=12)
    plt.title('Detailed Year-to-Year Connection Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    detail_path = '/home/todd/reconstructionism/validation/experiment_1/year_matrix_detail.png'
    plt.savefig(detail_path, dpi=150)
    print(f"Saved detailed matrix to: {detail_path}")
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main()