#!/usr/bin/env python3
"""
3D Semantic Landscape from Whole Document Embeddings
Creates a landscape view showing:
- Document clusters in semantic space
- Density of similar documents (peaks = common topics)
- Evolution of topics over time
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
import json
from glob import glob
import os
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

def load_document_embeddings(embeddings_dir="/home/todd/reconstructionism/validation/experiment_1/data/papers_with_embeddings"):
    """Load whole document embeddings if available."""
    
    documents = []
    
    # Try to load from experiment_1 output
    embedding_files = glob(os.path.join(embeddings_dir, "*.json"))
    
    if not embedding_files:
        print("No embeddings found. Generating synthetic data for demonstration...")
        # Generate synthetic data for demonstration
        np.random.seed(42)
        
        # Simulate different document clusters
        clusters = {
            'transformer': {'center': np.random.randn(768), 'count': 150},
            'optimization': {'center': np.random.randn(768), 'count': 100},
            'computer_vision': {'center': np.random.randn(768), 'count': 80},
            'nlp': {'center': np.random.randn(768), 'count': 120},
            'theory': {'center': np.random.randn(768), 'count': 50}
        }
        
        for topic, info in clusters.items():
            for i in range(info['count']):
                # Add noise around cluster center
                embedding = info['center'] + np.random.randn(768) * 0.3
                
                documents.append({
                    'id': f"{topic}_{i}",
                    'embedding': embedding,
                    'topic': topic,
                    'year': np.random.choice(range(2018, 2024)),
                    'semantic_value': 0.7 + np.random.randn() * 0.1
                })
    else:
        # Load actual embeddings
        for file_path in embedding_files[:500]:  # Limit for performance
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if 'embeddings' in data:
                    documents.append({
                        'id': data.get('id', 'unknown'),
                        'embedding': np.array(data['embeddings']),
                        'year': data.get('year', 2020),
                        'categories': data.get('categories', []),
                        'title': data.get('title', '')
                    })
            except Exception as e:
                continue
    
    return documents

def create_semantic_landscape_3d(documents, output_path):
    """Create 3D landscape visualization of document semantic space."""
    
    if not documents:
        print("No documents to visualize!")
        return
    
    # Extract embeddings
    embeddings = np.array([d['embedding'] for d in documents])
    
    # Reduce to 3D
    print("Reducing dimensions with PCA...")
    pca = PCA(n_components=3, random_state=42)
    reduced_3d = pca.fit_transform(embeddings)
    
    # Also reduce to 2D for density calculation
    pca_2d = PCA(n_components=2, random_state=42)
    reduced_2d = pca_2d.fit_transform(embeddings)
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate density landscape
    print("Calculating semantic density landscape...")
    x = reduced_2d[:, 0]
    y = reduced_2d[:, 1]
    
    # Create grid
    xi = np.linspace(x.min() - 1, x.max() + 1, 50)
    yi = np.linspace(y.min() - 1, y.max() + 1, 50)
    xi, yi = np.meshgrid(xi, yi)
    
    # Calculate kernel density
    positions = np.vstack([xi.ravel(), yi.ravel()])
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)
    zi = kernel(positions).reshape(xi.shape)
    
    # Plot surface
    surf = ax.plot_surface(xi, yi, zi, cmap='viridis', alpha=0.6, 
                          linewidth=0, antialiased=True)
    
    # Scatter plot of actual documents
    scatter = ax.scatter(reduced_3d[:, 0], reduced_3d[:, 1], reduced_3d[:, 2],
                        c=reduced_3d[:, 2], cmap='plasma', 
                        s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Find peaks (cluster centers)
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(reduced_2d)
    centers_2d = kmeans.cluster_centers_
    
    # Project cluster centers to 3D
    for i, center in enumerate(centers_2d):
        # Find height at center position
        center_density = kernel(center.reshape(-1, 1))[0]
        ax.scatter([center[0]], [center[1]], [center_density], 
                  color='red', s=200, marker='*', 
                  edgecolors='black', linewidth=2)
        
        # Add cluster label
        ax.text(center[0], center[1], center_density + 0.5, 
                f'Cluster {i+1}', fontsize=12, fontweight='bold')
    
    # Customize plot
    ax.set_xlabel('Semantic Dimension 1', fontsize=12, labelpad=10)
    ax.set_ylabel('Semantic Dimension 2', fontsize=12, labelpad=10)
    ax.set_zlabel('Document Density', fontsize=12, labelpad=10)
    ax.set_title('3D Semantic Landscape of Document Embeddings\n' +
                 'Peaks indicate areas of high semantic similarity', 
                 fontsize=16, pad=20)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Density')
    
    # Set viewing angle
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig

def create_temporal_semantic_flow(documents, output_path):
    """Visualize how semantic space evolves over time."""
    
    # Group by year
    years = sorted(set(d.get('year', 2020) for d in documents))
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract embeddings and reduce
    embeddings = np.array([d['embedding'] for d in documents])
    pca = PCA(n_components=3, random_state=42)
    reduced = pca.fit_transform(embeddings)
    
    # Color map for years
    norm = plt.Normalize(vmin=min(years), vmax=max(years))
    cmap = plt.cm.coolwarm
    
    # Plot documents colored by year
    for i, doc in enumerate(documents):
        year = doc.get('year', 2020)
        color = cmap(norm(year))
        
        ax.scatter(reduced[i, 0], reduced[i, 1], reduced[i, 2],
                  c=[color], s=50, alpha=0.6)
    
    # Add year trajectories
    year_centers = {}
    for year in years:
        year_docs = [i for i, d in enumerate(documents) if d.get('year', 2020) == year]
        if year_docs:
            center = reduced[year_docs].mean(axis=0)
            year_centers[year] = center
    
    # Draw trajectory
    sorted_years = sorted(year_centers.keys())
    if len(sorted_years) > 1:
        trajectory = np.array([year_centers[y] for y in sorted_years])
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                'k-', linewidth=3, alpha=0.7)
        
        # Add year labels
        for year, center in year_centers.items():
            ax.text(center[0], center[1], center[2], str(year),
                   fontsize=10, fontweight='bold')
    
    # Customize
    ax.set_xlabel('PC 1', fontsize=12)
    ax.set_ylabel('PC 2', fontsize=12)
    ax.set_zlabel('PC 3', fontsize=12)
    ax.set_title('Temporal Evolution of Semantic Space\n' +
                 'Document trajectory over years', fontsize=16)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Year', fontsize=10)
    
    ax.view_init(elev=20, azim=60)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig

def create_semantic_clustering_analysis(documents, output_path):
    """Analyze semantic clusters and their characteristics."""
    
    # Extract embeddings
    embeddings = np.array([d['embedding'] for d in documents])
    
    # Reduce dimensions
    pca = PCA(n_components=50, random_state=42)
    reduced = pca.fit_transform(embeddings)
    
    # Cluster
    n_clusters = 8
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(reduced)
    
    # Analyze clusters
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Cluster sizes
    cluster_sizes = np.bincount(cluster_labels)
    ax1.bar(range(n_clusters), cluster_sizes, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Cluster ID')
    ax1.set_ylabel('Number of Documents')
    ax1.set_title('Document Distribution Across Semantic Clusters')
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for i, size in enumerate(cluster_sizes):
        ax1.text(i, size + 5, str(size), ha='center', fontweight='bold')
    
    # Cluster coherence (intra-cluster similarity)
    coherences = []
    for i in range(n_clusters):
        cluster_docs = np.where(cluster_labels == i)[0]
        if len(cluster_docs) > 1:
            cluster_embeddings = embeddings[cluster_docs]
            # Sample for efficiency
            sample_size = min(50, len(cluster_docs))
            sample_idx = np.random.choice(len(cluster_docs), sample_size, replace=False)
            sample_embeddings = cluster_embeddings[sample_idx]
            
            # Calculate average pairwise similarity
            similarities = []
            for j in range(len(sample_embeddings)):
                for k in range(j+1, len(sample_embeddings)):
                    sim = np.dot(sample_embeddings[j], sample_embeddings[k]) / (
                        np.linalg.norm(sample_embeddings[j]) * 
                        np.linalg.norm(sample_embeddings[k])
                    )
                    similarities.append(sim)
            
            coherences.append(np.mean(similarities) if similarities else 0)
        else:
            coherences.append(0)
    
    ax2.bar(range(n_clusters), coherences, color='darkgreen', alpha=0.7)
    ax2.set_xlabel('Cluster ID')
    ax2.set_ylabel('Average Intra-cluster Similarity')
    ax2.set_title('Semantic Coherence of Document Clusters')
    ax2.set_ylim(0, 1)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for i, coh in enumerate(coherences):
        ax2.text(i, coh + 0.02, f'{coh:.3f}', ha='center', fontweight='bold')
    
    plt.suptitle('Semantic Cluster Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig, cluster_labels

def main():
    """Generate document-level semantic visualizations."""
    
    print("Generating Document Semantic Landscape Visualizations...")
    
    # Create output directory
    output_dir = "/home/todd/reconstructionism/validation/experiment_1/analysis/semantic_landscape"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load documents
    print("Loading document embeddings...")
    documents = load_document_embeddings()
    
    print(f"Loaded {len(documents)} documents")
    
    # Create semantic landscape
    print("Creating 3D semantic landscape...")
    create_semantic_landscape_3d(
        documents,
        os.path.join(output_dir, "semantic_landscape_3d.png")
    )
    
    # Create temporal flow
    print("Creating temporal semantic flow...")
    create_temporal_semantic_flow(
        documents,
        os.path.join(output_dir, "temporal_semantic_flow.png")
    )
    
    # Analyze clusters
    print("Analyzing semantic clusters...")
    fig, cluster_labels = create_semantic_clustering_analysis(
        documents,
        os.path.join(output_dir, "semantic_clusters.png")
    )
    
    # Save statistics
    stats = {
        'total_documents': len(documents),
        'years_covered': sorted(set(d.get('year', 2020) for d in documents)),
        'n_clusters': len(set(cluster_labels)),
        'largest_cluster_size': int(np.max(np.bincount(cluster_labels))),
        'embeddings_dim': len(documents[0]['embedding']) if documents else 0
    }
    
    with open(os.path.join(output_dir, "landscape_statistics.json"), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nVisualization complete! Output saved to: {output_dir}")
    print(f"Total documents analyzed: {stats['total_documents']}")

if __name__ == "__main__":
    main()