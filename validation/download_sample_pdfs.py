#!/usr/bin/env python3
"""
Download sample PDFs from arXiv for testing the pipeline
"""

import os
import requests
from pathlib import Path
import time
from tqdm import tqdm

def download_pdf(arxiv_id, output_dir):
    """Download a single PDF from arXiv"""
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    output_path = output_dir / f"{arxiv_id.replace('/', '_')}.pdf"
    
    if output_path.exists():
        print(f"  Already exists: {output_path.name}")
        return True
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"  {arxiv_id}") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return True
    except Exception as e:
        print(f"  Error downloading {arxiv_id}: {e}")
        # Clean up partial file if it exists
        if output_path.exists():
            try:
                output_path.unlink()
            except Exception:
                pass
        return False

def main():
    # Create output directory
    output_dir = Path("validation/data/raw_pdfs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading sample PDFs from arXiv...")
    print(f"Output directory: {output_dir}")
    
    # Sample papers related to information theory, embeddings, and AI
    sample_papers = [
        # Information Theory
        "2106.12062",  # "On the Information Bottleneck Theory of Deep Learning"
        "1802.07572",  # "Information-theoretic limits of matrix completion"
        
        # Embeddings and Representation Learning
        "1810.04805",  # "BERT: Pre-training of Deep Bidirectional Transformers"
        "2005.14165",  # "Language Models are Few-Shot Learners" (GPT-3)
        "1706.03762",  # "Attention Is All You Need" (Transformer)
        
        # RAG and Retrieval
        "2005.11401",  # "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
        "2104.08663",  # "Improving language models by retrieving from trillions of tokens"
        
        # Theory and Practice
        "1611.03530",  # "Understanding deep learning requires rethinking generalization"
        "1901.05639",  # "Rethinking the Value of Network Pruning"
        "1803.08494",  # "The Lottery Ticket Hypothesis"
        
        # Context and Attention
        "1909.01380",  # "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
        "2001.04451",  # "Reformer: The Efficient Transformer"
        
        # Multi-scale Analysis
        "1412.6980",  # "Adam: A Method for Stochastic Optimization"
        "1502.03167",  # "Batch Normalization"
        "1608.06993",  # "DenseNet: Densely Connected Convolutional Networks"
        
        # Additional papers for diversity
        "1908.10084",  # "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
        "1910.10683",  # "T5: Text-to-Text Transfer Transformer"
        "2203.15556",  # "Training Compute-Optimal Large Language Models" (Chinchilla)
        "2302.13971",  # "LLaMA: Open and Efficient Foundation Language Models"
        "2305.10403",  # "RWKV: Reinventing RNNs for the Transformer Era"
    ]
    
    # Download papers
    successful = 0
    for i, arxiv_id in enumerate(sample_papers, 1):
        print(f"\n[{i}/{len(sample_papers)}] Downloading {arxiv_id}...")
        if download_pdf(arxiv_id, output_dir):
            successful += 1
        
        # Be nice to arXiv servers
        if i < len(sample_papers):
            time.sleep(1)
    
    print(f"\n✓ Downloaded {successful}/{len(sample_papers)} papers successfully")
    print(f"PDFs saved to: {output_dir}")
    
    # Also create symlinks in experiment directories
    exp1_dir = Path("validation/experiment_1/data/raw_pdfs")
    exp2_dir = Path("validation/experiment_2/data/raw_pdfs")
    
    for exp_dir in [exp1_dir, exp2_dir]:
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create symlinks
        for pdf in output_dir.glob("*.pdf"):
            symlink = exp_dir / pdf.name
            try:
                # Remove existing symlink if it exists
                if symlink.exists() and symlink.is_symlink():
                    symlink.unlink()
                # Create new symlink
                symlink.symlink_to(pdf.absolute())
            except Exception as e:
                print(f"Warning: Could not create symlink {symlink}: {e}")
    
    print(f"\n✓ Created symlinks in experiment directories")

if __name__ == "__main__":
    main()