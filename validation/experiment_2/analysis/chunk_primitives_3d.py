#!/usr/bin/env python3
"""
3D Visualization of Semantic Primitives from Chunked Embeddings
Uses late-chunked embeddings to identify core primitives by plotting:
- Y-axis: Semantic similarity values within chunks
- X-axis: Spread across semantic space (using PCA/TSNE on chunk embeddings)
- Z-axis: Reference frequency across chunks
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from collections import Counter
import json
from glob import glob
import os
import re

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def extract_primitives_from_chunks(chunks_dir=None, limit=100):
    """Extract semantic primitives from late-chunked embeddings.
    
    Args:
        chunks_dir: Directory containing chunk files (default: relative to script)
        limit: Maximum number of papers to process (default: 100)
    """
    
    if chunks_dir is None:
        chunks_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'chunks')
    
    print("Loading chunked embeddings...")
    chunk_files = glob(os.path.join(chunks_dir, "*_chunks.json"))[:limit]
    
    all_chunks = []
    primitive_counts = Counter()
    
    # Keywords representing semantic primitives in ML/AI
    primitive_patterns = {
        # Core ML concepts
        'attention': r'\b(attention|self-attention|multi-head|attention mechanism)\b',
        'transformer': r'\b(transformer|bert|gpt|encoder|decoder)\b',
        'embedding': r'\b(embedding|embed|representation|vector space)\b',
        'neural': r'\b(neural|neuron|layer|deep learning)\b',
        'optimization': r'\b(optimiz|gradient|sgd|adam|loss function)\b',
        
        # Mathematical primitives
        'matrix': r'\b(matrix|matrices|tensor|multiplication)\b',
        'probability': r'\b(probability|distribution|likelihood|bayesian)\b',
        'dimension': r'\b(dimension|dimensional|space|projection)\b',
        
        # Algorithmic primitives
        'algorithm': r'\b(algorithm|procedure|method|approach)\b',
        'training': r'\b(train|training|learning|convergence)\b',
        
        # Theory primitives
        'theorem': r'\b(theorem|proof|lemma|proposition)\b',
        'complexity': r'\b(complexity|computational|efficiency|runtime)\b',
        
        # Application primitives
        'classification': r'\b(classification|classify|categorize)\b',
        'generation': r'\b(generat|synthesis|create|produce)\b',
        'prediction': r'\b(predict|forecast|estimate)\b'
    }
    
    for chunk_file in chunk_files:
        try:
            with open(chunk_file, 'r') as f:
                data = json.load(f)
            
            paper_id = data.get('paper_id', 'unknown')
            
            for chunk in data.get('chunks', []):
                chunk_text = chunk.get('text', '').lower()
                chunk_embedding = np.array(chunk.get('embedding', []))
                
                if len(chunk_embedding) == 0:
                    continue
                
                # Extract primitives from chunk text
                found_primitives = []
                for primitive, pattern in primitive_patterns.items():
                    if re.search(pattern, chunk_text, re.IGNORECASE):
                        found_primitives.append(primitive)
                        primitive_counts[primitive] += 1
                
                if found_primitives:
                    all_chunks.append({
                        'text': chunk_text[:200],  # First 200 chars
                        'embedding': chunk_embedding,
                        'primitives': found_primitives,
                        'paper_id': paper_id,
                        'chunk_index': chunk.get('chunk_index', 0)
                    })
        
        except Exception as e:
            print(f"Error processing {chunk_file}: {e}")
            continue
    
    print(f"Loaded {len(all_chunks)} chunks with primitives")
    print(f"Found {len(primitive_counts)} unique primitives")
    
    return all_chunks, primitive_counts

def cluster_chunks_by_similarity(chunks, n_clusters=20):
    """Cluster chunks based on embedding similarity."""
    
    if not chunks:
        return []
    
    # Extract embeddings
    embeddings = np.array([c['embedding'] for c in chunks])
    
    # Reduce dimensions for clustering
    pca = PCA(n_components=50, random_state=42)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Cluster using DBSCAN
    clustering = DBSCAN(eps=0.3, min_samples=5, metric='cosine')
    cluster_labels = clustering.fit_predict(reduced_embeddings)
    
    # Add cluster labels to chunks
    for i, chunk in enumerate(chunks):
        chunk['cluster'] = int(cluster_labels[i])
    
    return chunks

def create_3d_chunk_primitive_graph(chunks, primitive_counts, output_path):
    """Create 3D visualization of semantic primitives from chunks."""
    
    # Filter to top primitives
    top_primitives = sorted(primitive_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    top_primitive_names = [p[0] for p in top_primitives]
    
    # Prepare data for each primitive
    primitive_data = {}
    
    for primitive in top_primitive_names:
        # Get chunks containing this primitive
        primitive_chunks = [c for c in chunks if primitive in c['primitives']]
        
        if not primitive_chunks:
            continue
            
        # Extract embeddings
        embeddings = np.array([c['embedding'] for c in primitive_chunks])
        
        # Reduce to 2D for X-axis spread
        if len(embeddings) > 3:
            pca = PCA(n_components=2, random_state=42)
            reduced = pca.fit_transform(embeddings)
            
            # Calculate semantic coherence (average pairwise similarity)
            # Normalize embeddings once to avoid repeated norm calculations
            norms = np.linalg.norm(embeddings[:min(len(embeddings), 50)], axis=1)
            # Filter out zero norms to avoid division by zero
            valid_indices = norms > 0
            if np.sum(valid_indices) > 1:
                valid_embeddings = embeddings[:min(len(embeddings), 50)][valid_indices]
                valid_norms = norms[valid_indices]
                normalized_embeddings = valid_embeddings / valid_norms[:, np.newaxis]
                
                similarities = []
                for i in range(len(normalized_embeddings)):
                    for j in range(i+1, len(normalized_embeddings)):
                        sim = np.dot(normalized_embeddings[i], normalized_embeddings[j])
                        similarities.append(sim)
            else:
                similarities = []
            
            avg_similarity = np.mean(similarities) if similarities else 0.5
            
            primitive_data[primitive] = {
                'x_positions': reduced[:, 0],
                'y_positions': reduced[:, 1], 
                'semantic_coherence': avg_similarity,
                'frequency': primitive_counts[primitive],
                'n_chunks': len(primitive_chunks)
            }
    
    # Create 3D plot
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color gradient based on frequency
    norm = plt.Normalize(vmin=0, vmax=max(primitive_counts.values()))
    cmap = plt.cm.viridis
    
    for primitive, data in primitive_data.items():
        if 'x_positions' not in data:
            continue
            
        x = data['x_positions']
        y = np.full_like(x, data['semantic_coherence'])  # Semantic coherence on Y
        z = np.full_like(x, data['frequency'])  # Frequency on Z
        
        # Add small jitter for visibility
        y = y + np.random.normal(0, 0.02, len(y))
        z = z + np.random.normal(0, data['frequency']*0.05, len(z))
        
        # Color based on frequency
        color = cmap(norm(data['frequency']))
        
        # Size based on number of chunks
        sizes = np.sqrt(data['n_chunks']) * 20
        
        scatter = ax.scatter(x, y, z, 
                           c=[color],
                           s=sizes,
                           alpha=0.6,
                           edgecolors='black',
                           linewidth=0.5,
                           label=f"{primitive}")
        
        # Add label at centroid
        ax.text(np.mean(x), data['semantic_coherence'], data['frequency'] + 50, 
                primitive,
                fontsize=10, 
                fontweight='bold',
                ha='center', 
                va='bottom')
    
    # Customize plot
    ax.set_xlabel('Semantic Space (PCA 1)', fontsize=12, labelpad=10)
    ax.set_ylabel('Semantic Coherence', fontsize=12, labelpad=10)
    ax.set_zlabel('Reference Frequency', fontsize=12, labelpad=10)
    ax.set_title('3D Semantic Primitive Landscape from Chunked Embeddings\n' + 
                 'Identifying Core Primitives in Semantically Coherent Chunks', 
                 fontsize=16, pad=20)
    
    # Set viewing angle
    ax.view_init(elev=25, azim=45)
    
    # Add colorbar for frequency
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Reference Frequency', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig

def create_primitive_cluster_graph(chunks, output_path):
    """Visualize how primitives cluster together in semantic space."""
    
    # Cluster chunks
    clustered_chunks = cluster_chunks_by_similarity(chunks)
    
    # Analyze primitive co-occurrence by cluster
    cluster_primitives = {}
    for chunk in clustered_chunks:
        cluster = chunk['cluster']
        if cluster not in cluster_primitives:
            cluster_primitives[cluster] = Counter()
        
        for primitive in chunk['primitives']:
            cluster_primitives[cluster][primitive] += 1
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Get top clusters by size
    cluster_sizes = [(c, sum(counts.values())) for c, counts in cluster_primitives.items()]
    top_clusters = sorted(cluster_sizes, key=lambda x: x[1], reverse=True)[:10]
    
    # Prepare data for heatmap
    all_primitives = set()
    for cluster, _ in top_clusters:
        all_primitives.update(cluster_primitives[cluster].keys())
    
    primitive_list = sorted(list(all_primitives))
    
    # Create matrix
    matrix = np.zeros((len(top_clusters), len(primitive_list)))
    for i, (cluster, _) in enumerate(top_clusters):
        for j, primitive in enumerate(primitive_list):
            matrix[i, j] = cluster_primitives[cluster].get(primitive, 0)
    
    # Normalize by row (cluster)
    matrix_norm = matrix / (matrix.sum(axis=1, keepdims=True) + 1e-10)
    
    # Create heatmap
    sns.heatmap(matrix_norm,
                xticklabels=primitive_list,
                yticklabels=[f"Cluster {c[0]}" for c in top_clusters],
                cmap='YlOrRd',
                annot=False,
                fmt='.2f',
                cbar_kws={'label': 'Normalized Frequency'},
                ax=ax)
    
    ax.set_title('Semantic Primitive Distribution Across Chunk Clusters', fontsize=16)
    ax.set_xlabel('Semantic Primitives', fontsize=12)
    ax.set_ylabel('Chunk Clusters', fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig

def analyze_primitive_evolution(chunks, output_path):
    """Analyze how primitives combine and evolve through chunks."""
    
    # Track primitive combinations
    combinations = Counter()
    for chunk in chunks:
        if len(chunk['primitives']) > 1:
            # Sort to ensure consistent ordering
            combo = tuple(sorted(chunk['primitives']))
            combinations[combo] += 1
    
    # Get top combinations
    top_combos = combinations.most_common(20)
    
    # Create network graph visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Build co-occurrence matrix
    all_primitives = set()
    for combo, _ in top_combos:
        all_primitives.update(combo)
    
    primitive_list = sorted(list(all_primitives))
    n = len(primitive_list)
    
    co_occurrence = np.zeros((n, n))
    for combo, count in top_combos:
        for i, p1 in enumerate(primitive_list):
            for j, p2 in enumerate(primitive_list):
                if p1 in combo and p2 in combo and i != j:
                    co_occurrence[i, j] += count
    
    # Create heatmap
    mask = np.triu(np.ones_like(co_occurrence), k=1)  # Upper triangle mask
    
    sns.heatmap(co_occurrence,
                xticklabels=primitive_list,
                yticklabels=primitive_list,
                mask=mask,
                cmap='Blues',
                annot=True,
                fmt='.0f',
                square=True,
                cbar_kws={'label': 'Co-occurrence Count'},
                ax=ax)
    
    ax.set_title('Semantic Primitive Co-occurrence in Chunks', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig, top_combos

def main():
    """Generate chunk-based semantic primitive visualizations."""
    
    print("Generating 3D Semantic Primitive Visualizations from Chunks...")
    
    # Create output directory
    output_dir = os.environ.get('CHUNK_PRIMITIVES_OUTPUT_DIR', 
                               os.path.join(os.path.dirname(__file__), 'chunk_primitives'))
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract primitives from chunks
    print("Extracting primitives from chunked embeddings...")
    chunks, primitive_counts = extract_primitives_from_chunks()
    
    if not chunks:
        print("No chunks found! Please run the late chunking pipeline first.")
        return
    
    # Create 3D visualization
    print("Creating 3D primitive landscape...")
    create_3d_chunk_primitive_graph(
        chunks, 
        primitive_counts,
        os.path.join(output_dir, "chunk_primitives_3d.png")
    )
    
    # Create cluster analysis
    print("Creating primitive cluster analysis...")
    create_primitive_cluster_graph(
        chunks,
        os.path.join(output_dir, "primitive_clusters.png")
    )
    
    # Analyze primitive combinations
    print("Analyzing primitive combinations...")
    fig, top_combos = analyze_primitive_evolution(
        chunks,
        os.path.join(output_dir, "primitive_cooccurrence.png")
    )
    
    # Save statistics
    stats = {
        'total_chunks_analyzed': len(chunks),
        'unique_primitives': len(primitive_counts),
        'total_primitive_references': sum(primitive_counts.values()),
        'top_10_primitives': dict(sorted(primitive_counts.items(), 
                                       key=lambda x: x[1], 
                                       reverse=True)[:10]),
        'top_10_combinations': [
            {
                'primitives': list(combo),
                'count': count
            } for combo, count in top_combos[:10]
        ],
        'avg_primitives_per_chunk': np.mean([len(c['primitives']) for c in chunks])
    }
    
    with open(os.path.join(output_dir, "chunk_primitive_statistics.json"), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nVisualization complete! Output saved to: {output_dir}")
    print("\nKey findings:")
    print(f"- Total chunks analyzed: {stats['total_chunks_analyzed']}")
    print(f"- Unique primitives: {stats['unique_primitives']}")
    print(f"- Avg primitives per chunk: {stats['avg_primitives_per_chunk']:.2f}")
    if stats['top_10_primitives']:
        top_prim = list(stats['top_10_primitives'].items())[0]
        print(f"- Top primitive: {top_prim[0]} ({top_prim[1]} references)")

if __name__ == "__main__":
    main()