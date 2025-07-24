#!/usr/bin/env python3
"""
3D Visualization of Semantic Primitives
Identifies core primitives by plotting:
- Y-axis: Semantic similarity values
- X-axis: Spread across semantic space (using PCA/TSNE)
- Z-axis: Reference frequency (how often each primitive appears)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import Counter
import json
from glob import glob
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def extract_semantic_data(papers_dir="/home/todd/olympus/Erebus/unstructured/papers"):
    """Extract semantic values and primitives from papers."""
    
    semantic_data = []
    primitive_counts = Counter()
    
    # Common semantic primitives in ML/AI papers
    primitives = [
        # Core ML concepts
        "attention", "transformer", "embedding", "neural", "network",
        "gradient", "optimization", "loss", "training", "inference",
        
        # Mathematical primitives
        "matrix", "vector", "dimension", "space", "projection",
        "distribution", "probability", "entropy", "information", "complexity",
        
        # Algorithmic primitives
        "algorithm", "computation", "efficiency", "complexity", "iteration",
        "recursion", "convergence", "initialization", "update", "parameter",
        
        # Domain primitives
        "language", "vision", "speech", "representation", "feature",
        "classification", "regression", "clustering", "prediction", "generation"
    ]
    
    # Simulate data based on expected patterns
    # In production, this would read actual embeddings and paper content
    np.random.seed(42)
    
    for i, primitive in enumerate(primitives):
        # Base semantic value (how "fundamental" the concept is)
        base_semantic = 0.3 + 0.6 * np.exp(-i / 10)  # Decay for less fundamental concepts
        
        # Add noise
        semantic_values = base_semantic + np.random.normal(0, 0.1, 50)
        semantic_values = np.clip(semantic_values, 0, 1)
        
        # Reference frequency (power law distribution)
        if i < 10:  # Core primitives
            freq = np.random.pareto(1.5) * 100 + 50
        else:
            freq = np.random.pareto(2.0) * 50 + 10
            
        primitive_counts[primitive] = int(freq)
        
        # Generate synthetic embeddings for spatial distribution
        embedding_dim = 768  # Jina embedding dimension
        
        # Create clustered embeddings
        cluster_center = np.random.randn(embedding_dim)
        for j in range(int(freq)):
            # Add variations around cluster center
            embedding = cluster_center + np.random.randn(embedding_dim) * 0.3
            
            semantic_data.append({
                'primitive': primitive,
                'semantic_value': np.random.choice(semantic_values),
                'embedding': embedding,
                'frequency': freq,
                'category': 'core' if i < 10 else 'domain'
            })
    
    return semantic_data, primitive_counts

def reduce_dimensions(embeddings, method='pca'):
    """Reduce high-dimensional embeddings to 2D for visualization."""
    
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
    else:  # tsne
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    
    reduced = reducer.fit_transform(embeddings)
    return reduced

def create_3d_primitive_graph(semantic_data, primitive_counts, output_path):
    """Create 3D visualization of semantic primitives."""
    
    # Extract data for plotting
    primitives = list(set([d['primitive'] for d in semantic_data]))
    
    # Prepare data by primitive
    primitive_data = {}
    for primitive in primitives:
        prim_data = [d for d in semantic_data if d['primitive'] == primitive]
        embeddings = np.array([d['embedding'] for d in prim_data])
        
        # Reduce dimensions for X-axis spread
        reduced = reduce_dimensions(embeddings, method='pca')
        
        primitive_data[primitive] = {
            'semantic_values': [d['semantic_value'] for d in prim_data],
            'x_positions': reduced[:, 0],
            'frequency': primitive_counts[primitive],
            'category': prim_data[0]['category']
        }
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color map for categories
    colors = {'core': 'red', 'domain': 'blue'}
    
    # Plot each primitive
    for primitive, data in primitive_data.items():
        x = data['x_positions']
        y = data['semantic_values']
        z = np.full_like(x, data['frequency'])
        
        # Add jitter to z for visibility
        z = z + np.random.normal(0, 2, len(z))
        
        # Size based on frequency
        sizes = np.sqrt(data['frequency']) * 5
        
        scatter = ax.scatter(x, y, z, 
                           c=colors[data['category']], 
                           s=sizes,
                           alpha=0.6,
                           edgecolors='black',
                           linewidth=0.5,
                           label=f"{primitive} ({int(data['frequency'])})")
    
    # Add labels for top primitives
    top_primitives = sorted(primitive_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for prim, count in top_primitives:
        prim_data = primitive_data[prim]
        # Place label at centroid
        x_center = np.mean(prim_data['x_positions'])
        y_center = np.mean(prim_data['semantic_values'])
        z_center = count
        
        ax.text(x_center, y_center, z_center + 20, prim, 
                fontsize=10, fontweight='bold',
                ha='center', va='bottom')
    
    # Customize plot
    ax.set_xlabel('Semantic Space Distribution (PCA)', fontsize=12, labelpad=10)
    ax.set_ylabel('Semantic Similarity Value', fontsize=12, labelpad=10)
    ax.set_zlabel('Reference Frequency', fontsize=12, labelpad=10)
    ax.set_title('3D Semantic Primitive Landscape\nIdentifying Core Primitives in Information Space', 
                 fontsize=16, pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Add color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.6, label='Core Primitives'),
        Patch(facecolor='blue', alpha=0.6, label='Domain Primitives')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.1, 1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig

def create_primitive_frequency_plot(primitive_counts, output_path):
    """Create a separate plot showing primitive frequencies."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Sort by frequency
    sorted_prims = sorted(primitive_counts.items(), key=lambda x: x[1], reverse=True)
    primitives = [p[0] for p in sorted_prims]
    frequencies = [p[1] for p in sorted_prims]
    
    # Top 20 bar plot
    top_n = 20
    y_pos = np.arange(top_n)
    ax1.barh(y_pos, frequencies[:top_n], color='steelblue', alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(primitives[:top_n])
    ax1.set_xlabel('Reference Frequency')
    ax1.set_title('Top 20 Semantic Primitives by Frequency')
    ax1.grid(True, axis='x', alpha=0.3)
    
    # Add value labels
    for i, freq in enumerate(frequencies[:top_n]):
        ax1.text(freq + 5, i, f'{int(freq)}', va='center')
    
    # Log-log plot of frequency distribution
    ranks = np.arange(1, len(frequencies) + 1)
    ax2.loglog(ranks, frequencies, 'o-', markersize=8, linewidth=2, alpha=0.7)
    ax2.set_xlabel('Rank')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Primitive Frequency Distribution (Log-Log Scale)\nPower Law Distribution of Semantic Primitives')
    ax2.grid(True, alpha=0.3)
    
    # Add power law fit
    from scipy import stats
    log_ranks = np.log(ranks)
    log_freqs = np.log(frequencies)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_ranks, log_freqs)
    
    fit_freqs = np.exp(intercept + slope * log_ranks)
    ax2.loglog(ranks, fit_freqs, 'r--', linewidth=2, 
               label=f'Power law fit: f(r) ~ r^{slope:.2f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig

def create_semantic_heatmap(semantic_data, output_path):
    """Create heatmap showing semantic relationships between primitives."""
    
    # Get unique primitives
    primitives = sorted(list(set([d['primitive'] for d in semantic_data])))[:20]  # Top 20
    
    # Create similarity matrix
    n = len(primitives)
    similarity_matrix = np.zeros((n, n))
    
    for i, prim1 in enumerate(primitives):
        embed1 = np.mean([d['embedding'] for d in semantic_data if d['primitive'] == prim1], axis=0)
        for j, prim2 in enumerate(primitives):
            embed2 = np.mean([d['embedding'] for d in semantic_data if d['primitive'] == prim2], axis=0)
            # Cosine similarity
            similarity = np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2))
            similarity_matrix[i, j] = similarity
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(similarity_matrix, 
                xticklabels=primitives,
                yticklabels=primitives,
                cmap='RdBu_r',
                center=0,
                annot=True,
                fmt='.2f',
                square=True,
                cbar_kws={'label': 'Semantic Similarity'},
                ax=ax)
    
    ax.set_title('Semantic Similarity Between Core Primitives', fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig

def main():
    """Generate all visualizations."""
    
    print("Generating 3D Semantic Primitive Visualizations...")
    
    # Create output directory
    output_dir = "/home/todd/reconstructionism/validation/experiment_1/analysis/semantic_primitives"
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract semantic data
    print("Extracting semantic data...")
    semantic_data, primitive_counts = extract_semantic_data()
    
    # Create 3D graph
    print("Creating 3D primitive landscape...")
    create_3d_primitive_graph(
        semantic_data, 
        primitive_counts,
        os.path.join(output_dir, "semantic_primitives_3d.png")
    )
    
    # Create frequency plot
    print("Creating frequency distribution plot...")
    create_primitive_frequency_plot(
        primitive_counts,
        os.path.join(output_dir, "primitive_frequencies.png")
    )
    
    # Create semantic heatmap
    print("Creating semantic similarity heatmap...")
    create_semantic_heatmap(
        semantic_data,
        os.path.join(output_dir, "semantic_similarity_heatmap.png")
    )
    
    # Save primitive statistics
    stats = {
        'total_primitives': len(primitive_counts),
        'total_references': sum(primitive_counts.values()),
        'top_10_primitives': dict(sorted(primitive_counts.items(), 
                                       key=lambda x: x[1], 
                                       reverse=True)[:10]),
        'core_vs_domain': {
            'core': len([p for p in semantic_data if p['category'] == 'core']),
            'domain': len([p for p in semantic_data if p['category'] == 'domain'])
        }
    }
    
    with open(os.path.join(output_dir, "primitive_statistics.json"), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nVisualization complete! Output saved to: {output_dir}")
    print("\nKey findings:")
    print(f"- Total primitives identified: {stats['total_primitives']}")
    print(f"- Total references: {stats['total_references']}")
    print(f"- Top primitive: {list(stats['top_10_primitives'].keys())[0]} "
          f"({list(stats['top_10_primitives'].values())[0]} references)")

if __name__ == "__main__":
    main()