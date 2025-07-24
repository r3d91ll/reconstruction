#!/usr/bin/env python3
"""
Step 6: GPU-Accelerated Multi-Scale Analysis
Analyzes context amplification at multiple scales using GPU
"""

import os
import json
import torch
import numpy as np
from datetime import datetime
import logging
from arango import ArangoClient
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import gc

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GPUMultiscaleAnalyzer:
    def __init__(self, device_ids=[0, 1]):
        """
        Initialize GPU multiscale analyzer
        
        Args:
            device_ids: List of GPU IDs to use
        """
        self.device_ids = device_ids
        self.primary_device = torch.device(f'cuda:{device_ids[0]}')
        
        logger.info(f"GPU Multiscale Analyzer initialized")
        logger.info(f"Using GPUs: {device_ids}")
    
    def compute_context_amplification(self, similarities, contexts, alpha_range=(1.0, 3.0)):
        """
        Compute context amplification factor (α) on GPU
        
        Args:
            similarities: Tensor of similarity scores
            contexts: Tensor of context values
            alpha_range: Range of α values to test
        
        Returns:
            Optimal α value and fit metrics
        """
        # Convert to GPU tensors
        sims = torch.tensor(similarities, device=self.primary_device)
        ctxs = torch.tensor(contexts, device=self.primary_device)
        
        # Test different α values
        alphas = torch.linspace(alpha_range[0], alpha_range[1], 100, device=self.primary_device)
        best_alpha = alpha_range[0]
        best_error = float('inf')
        
        for alpha in alphas:
            # Compute predicted similarities: sim * context^α
            predicted = sims[0] * torch.pow(ctxs, alpha)
            
            # Compute MSE
            error = F.mse_loss(predicted, sims)
            
            if error < best_error:
                best_error = error
                best_alpha = alpha.item()
        
        # Compute R² for best α
        predicted_best = sims[0] * torch.pow(ctxs, best_alpha)
        ss_tot = torch.sum((sims - sims.mean()) ** 2)
        ss_res = torch.sum((sims - predicted_best) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return {
            'alpha': best_alpha,
            'mse': best_error.item(),
            'r_squared': r_squared.item()
        }
    
    def analyze_scale_hierarchy(self, embeddings, scale_levels=4):
        """
        Analyze hierarchical structure at multiple scales on GPU
        
        Args:
            embeddings: Document/chunk embeddings
            scale_levels: Number of hierarchical levels to analyze
        
        Returns:
            Scale hierarchy analysis results
        """
        # Convert to GPU tensor
        emb_tensor = torch.tensor(embeddings, device=self.primary_device)
        
        hierarchy_results = []
        
        for level in range(scale_levels):
            # Compute similarity matrix at this scale
            sim_matrix = torch.mm(emb_tensor, emb_tensor.t())
            
            # Apply threshold based on scale
            threshold = 0.7 - (level * 0.15)  # Decreasing threshold for broader scales
            
            # Find clusters at this scale
            adjacency = (sim_matrix > threshold).float()
            
            # Compute cluster statistics
            degrees = adjacency.sum(dim=1)
            avg_degree = degrees.mean().item()
            
            # Identify clusters (simplified spectral clustering on GPU)
            # Symmetrize the adjacency matrix to ensure it's symmetric
            adjacency_symmetric = (adjacency + adjacency.T) / 2
            eigenvalues, eigenvectors = torch.linalg.eigh(adjacency_symmetric)
            
            # Use top eigenvectors for clustering
            n_clusters = min(10, len(embeddings) // 10)
            top_eigenvectors = eigenvectors[:, -n_clusters:]
            
            # Simple k-means style clustering on GPU
            centroids = top_eigenvectors[torch.randperm(len(top_eigenvectors))[:n_clusters]]
            
            for _ in range(10):  # K-means iterations
                # Assign points to clusters
                distances = torch.cdist(top_eigenvectors, centroids)
                assignments = distances.argmin(dim=1)
                
                # Update centroids
                for k in range(n_clusters):
                    mask = assignments == k
                    if mask.any():
                        centroids[k] = top_eigenvectors[mask].mean(dim=0)
            
            # Compute cluster sizes
            cluster_sizes = torch.bincount(assignments, minlength=n_clusters)
            
            hierarchy_results.append({
                'level': level,
                'threshold': threshold,
                'avg_degree': avg_degree,
                'n_clusters': n_clusters,
                'cluster_sizes': cluster_sizes.cpu().numpy().tolist(),
                'modularity': self._compute_modularity(adjacency, assignments)
            })
        
        return hierarchy_results
    
    def _compute_modularity(self, adjacency, assignments):
        """Compute modularity score on GPU"""
        m = adjacency.sum() / 2
        if m == 0:
            return 0.0
        
        # Compute modularity matrix
        degrees = adjacency.sum(dim=1)
        expected = torch.outer(degrees, degrees) / (2 * m)
        modularity_matrix = adjacency - expected
        
        # Sum within clusters
        modularity = 0
        for k in torch.unique(assignments):
            mask = assignments == k
            cluster_mod = modularity_matrix[mask][:, mask].sum()
            modularity += cluster_mod
        
        return (modularity / (2 * m)).item()

def main():
    # Configuration
    db_name = os.environ.get("EXP2_DB_NAME", "information_reconstructionism_exp2_gpu")
    results_dir = os.environ.get("EXP2_RESULTS_DIR", ".")
    analysis_dir = os.path.join(results_dir, "multiscale_analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # GPU configuration
    gpu_ids = [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")]
    
    # ArangoDB connection
    arango_host = os.environ.get('ARANGO_HOST', 'http://192.168.1.69:8529')
    username = os.environ.get('ARANGO_USERNAME', 'root')
    password = os.environ.get('ARANGO_PASSWORD', '')
    
    logger.info("=" * 60)
    logger.info("STEP 6: GPU-ACCELERATED MULTI-SCALE ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Database: {db_name}")
    logger.info(f"Using GPUs: {gpu_ids}")
    logger.info(f"Analysis directory: {analysis_dir}")
    
    # Initialize GPU analyzer
    analyzer = GPUMultiscaleAnalyzer(device_ids=gpu_ids)
    
    # Connect to database
    client = ArangoClient(hosts=arango_host)
    db = client.db(db_name, username=username, password=password)
    
    start_time = datetime.now()
    
    # 1. Analyze Context Amplification
    logger.info("\n1. Analyzing Context Amplification (α)...")
    
    # Get similarity data with context information
    query = """
    FOR edge IN chunk_similarities_exp2
        LIMIT 10000
        LET from_chunk = DOCUMENT(edge._from)
        LET to_chunk = DOCUMENT(edge._to)
        LET context_distance = ABS(from_chunk.chunk_id - to_chunk.chunk_id)
        RETURN {
            similarity: edge.similarity,
            context_distance: context_distance,
            same_paper: edge.from_paper == edge.to_paper
        }
    """
    
    results = list(db.aql.execute(query))
    
    if results:
        # Separate by same/different paper
        same_paper_sims = [(r['similarity'], 1.0 / (1 + r['context_distance'])) 
                          for r in results if r['same_paper']]
        diff_paper_sims = [(r['similarity'], 0.5) 
                          for r in results if not r['same_paper']]
        
        # Analyze context amplification for same-paper chunks
        if same_paper_sims:
            sims, contexts = zip(*same_paper_sims)
            context_result = analyzer.compute_context_amplification(sims, contexts)
            
            logger.info(f"  Context amplification factor (α): {context_result['alpha']:.3f}")
            logger.info(f"  R² score: {context_result['r_squared']:.3f}")
    
    # 2. Multi-Scale Hierarchy Analysis
    logger.info("\n2. Analyzing Multi-Scale Hierarchy...")
    
    # Load document embeddings
    query = """
    FOR paper IN papers_exp2
        LIMIT 1000
        LET chunks = (
            FOR chunk IN chunks_exp2
                FILTER chunk.paper_id == paper.paper_id
                LIMIT 1
                RETURN chunk.embedding
        )
        FILTER LENGTH(chunks) > 0
        RETURN {
            paper_id: paper.paper_id,
            embedding: chunks[0]
        }
    """
    
    doc_data = list(db.aql.execute(query))
    
    if doc_data:
        embeddings = np.array([d['embedding'] for d in doc_data])
        paper_ids = [d['paper_id'] for d in doc_data]
        
        # Analyze hierarchy
        hierarchy_results = analyzer.analyze_scale_hierarchy(embeddings, scale_levels=4)
        
        logger.info("\n  Scale Hierarchy:")
        for result in hierarchy_results:
            logger.info(f"    Level {result['level']}: {result['n_clusters']} clusters, "
                       f"avg degree: {result['avg_degree']:.1f}, "
                       f"modularity: {result['modularity']:.3f}")
    
    # 3. Theory-Practice Bridge Analysis
    logger.info("\n3. Analyzing Theory-Practice Bridges...")
    
    # Find high-conveyance connections
    query = """
    FOR edge IN document_similarities_exp2
        FILTER edge.similarity > 0.7
        LIMIT 100
        LET from_doc = DOCUMENT(edge._from)
        LET to_doc = DOCUMENT(edge._to)
        RETURN {
            from_id: from_doc.paper_id,
            to_id: to_doc.paper_id,
            similarity: edge.similarity,
            from_meta: from_doc.metadata,
            to_meta: to_doc.metadata
        }
    """
    
    bridges = list(db.aql.execute(query))
    
    # Classify bridges (simplified - in reality would use metadata)
    theory_practice_bridges = []
    for bridge in bridges:
        # Simple heuristic: check if titles suggest theory vs practice
        from_title = str(bridge.get('from_meta', {}).get('filename', '')).lower()
        to_title = str(bridge.get('to_meta', {}).get('filename', '')).lower()
        
        theory_keywords = ['theory', 'framework', 'model', 'analysis']
        practice_keywords = ['implementation', 'application', 'case', 'study']
        
        from_is_theory = any(kw in from_title for kw in theory_keywords)
        from_is_practice = any(kw in from_title for kw in practice_keywords)
        to_is_theory = any(kw in to_title for kw in theory_keywords)
        to_is_practice = any(kw in to_title for kw in practice_keywords)
        
        if (from_is_theory and to_is_practice) or (to_is_theory and from_is_practice):
            theory_practice_bridges.append(bridge)
    
    logger.info(f"  Found {len(theory_practice_bridges)} potential theory-practice bridges")
    
    # 4. Dimensional Analysis
    logger.info("\n4. Performing Dimensional Analysis on GPU...")
    
    # PCA on GPU for dimensional analysis
    if len(embeddings) > 100:
        emb_tensor = torch.tensor(embeddings[:1000], device=analyzer.primary_device)
        
        # Center the data
        emb_centered = emb_tensor - emb_tensor.mean(dim=0)
        
        # Compute SVD on GPU
        U, S, V = torch.svd(emb_centered)
        
        # Compute explained variance
        explained_variance = (S ** 2) / (len(embeddings) - 1)
        total_variance = explained_variance.sum()
        explained_variance_ratio = explained_variance / total_variance
        
        # Find number of dimensions explaining 95% variance
        cumsum = torch.cumsum(explained_variance_ratio, dim=0)
        n_dims_95 = (cumsum < 0.95).sum().item() + 1
        
        logger.info(f"  Dimensions explaining 95% variance: {n_dims_95}")
        logger.info(f"  Top 10 eigenvalues: {S[:10].cpu().numpy()}")
    
    # 5. Generate Visualizations
    logger.info("\n5. Generating Visualizations...")
    
    # Context amplification plot
    if 'context_result' in locals() and 'same_paper_sims' in locals() and same_paper_sims:
        plt.figure(figsize=(10, 6))
        plt.scatter([c for _, c in same_paper_sims[:1000]], 
                   [s for s, _ in same_paper_sims[:1000]], 
                   alpha=0.5, label='Actual')
        
        # Plot fitted curve
        contexts = np.linspace(0.1, 1.0, 100)
        predicted = same_paper_sims[0][0] * (contexts ** context_result['alpha'])
        plt.plot(contexts, predicted, 'r-', label=f'Context^{context_result["alpha"]:.2f}')
        
        plt.xlabel('Context Score')
        plt.ylabel('Similarity')
        plt.title('Context Amplification Analysis')
        plt.legend()
        plt.savefig(os.path.join(analysis_dir, 'context_amplification.png'))
        plt.close()
    
    # Hierarchy visualization
    if hierarchy_results:
        plt.figure(figsize=(12, 8))
        levels = [r['level'] for r in hierarchy_results]
        n_clusters = [r['n_clusters'] for r in hierarchy_results]
        modularities = [r['modularity'] for r in hierarchy_results]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.plot(levels, n_clusters, 'bo-')
        ax1.set_xlabel('Hierarchy Level')
        ax1.set_ylabel('Number of Clusters')
        ax1.set_title('Multi-Scale Cluster Hierarchy')
        
        ax2.plot(levels, modularities, 'ro-')
        ax2.set_xlabel('Hierarchy Level')
        ax2.set_ylabel('Modularity')
        ax2.set_title('Modularity at Different Scales')
        
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'scale_hierarchy.png'))
        plt.close()
    
    # Calculate final statistics
    duration = (datetime.now() - start_time).total_seconds()
    
    logger.info("\n" + "=" * 60)
    logger.info("MULTI-SCALE ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {duration:.1f} seconds")
    
    # GPU memory stats
    for gpu_id in gpu_ids:
        allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
        reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
        logger.info(f"\nGPU {gpu_id} memory usage:")
        logger.info(f"  Allocated: {allocated:.2f} GB")
        logger.info(f"  Reserved: {reserved:.2f} GB")
    
    # Save comprehensive analysis report
    analysis_report = {
        'context_amplification': context_result if 'context_result' in locals() else None,
        'hierarchy_analysis': hierarchy_results if 'hierarchy_results' in locals() else None,
        'theory_practice_bridges': len(theory_practice_bridges) if 'theory_practice_bridges' in locals() else 0,
        'dimensional_analysis': {
            'dims_for_95_variance': n_dims_95 if 'n_dims_95' in locals() else None
        },
        'duration_seconds': duration,
        'gpu_ids': gpu_ids,
        'timestamp': datetime.now().isoformat()
    }
    
    report_path = os.path.join(analysis_dir, "multiscale_analysis_report.json")
    with open(report_path, 'w') as f:
        json.dump(analysis_report, f, indent=2)
    
    logger.info(f"\nAnalysis report saved to: {report_path}")
    logger.info(f"Visualizations saved to: {analysis_dir}")
    logger.info("\n✓ GPU-ACCELERATED MULTI-SCALE ANALYSIS COMPLETE")
    
    return 0

if __name__ == "__main__":
    exit(main())