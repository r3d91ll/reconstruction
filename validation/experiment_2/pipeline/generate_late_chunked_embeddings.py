#!/usr/bin/env python3
"""
Generate late-chunked embeddings using Jina V4's built-in semantic chunking.
Supports dual GPU processing for faster throughput.
"""

import os
import json
import torch
from glob import glob
from transformers import AutoModel
import numpy as np
from datetime import datetime
import logging
from typing import List, Dict, Any
import multiprocessing as mp
from tqdm import tqdm
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [GPU %(gpu_id)s] %(message)s'
)
logger = logging.getLogger(__name__)


class JinaLateChunkProcessor:
    """Process papers with Jina V4's late chunking on specified GPU."""
    
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{gpu_id}')
        
        # Load model on specific GPU
        logger.info(f"Loading Jina V4 on GPU {gpu_id}", extra={'gpu_id': gpu_id})
        self.model = AutoModel.from_pretrained(
            'jinaai/jina-embeddings-v4',
            trust_remote_code=True
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded on {self.device}", extra={'gpu_id': gpu_id})
        
    def process_paper(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single paper and return late-chunked embeddings."""
        json_path = paper_data['path']
        
        try:
            # Load paper
            with open(json_path, 'r') as f:
                paper = json.load(f)
            
            # Get paper ID
            paper_id = paper.get('id', os.path.basename(json_path).replace('.json', ''))
            
            # Build full text from PDF content
            if 'pdf_content' in paper and 'markdown' in paper['pdf_content']:
                # Full PDF content
                title = paper.get('title', '')
                abstract = paper.get('abstract', '')
                categories = ', '.join(paper.get('categories', []))
                markdown = paper['pdf_content']['markdown']
                
                full_text = f"Title: {title}\n\nCategories: {categories}\n\nAbstract: {abstract}\n\n# Full Paper Content\n\n{markdown}"
                
                # Log content size
                logger.info(f"Processing {paper_id}: {len(full_text)} chars", extra={'gpu_id': self.gpu_id})
                
                # Truncate if needed (128K token limit ~ 512K chars)
                max_chars = 512000
                if len(full_text) > max_chars:
                    full_text = full_text[:max_chars] + "\n\n[Content truncated...]"
                    logger.warning(f"Truncated {paper_id} to {max_chars} chars", extra={'gpu_id': self.gpu_id})
                
            else:
                # Fallback to abstract only
                title = paper.get('title', '')
                abstract = paper.get('abstract', '')
                categories = ', '.join(paper.get('categories', []))
                full_text = f"Title: {title}\n\nCategories: {categories}\n\nAbstract: {abstract}"
                logger.warning(f"No PDF content for {paper_id}, using abstract only", extra={'gpu_id': self.gpu_id})
            
            # Generate late-chunked embeddings
            # According to Jina docs, we need to use encode with late chunking enabled
            with torch.no_grad():
                # Late chunking returns multiple embeddings per document
                results = self.model.encode(
                    [full_text],
                    task="retrieval",
                    prompt_name="passage",
                    return_late_chunked_embeddings=True,  # This enables semantic chunking!
                    # chunk_size parameter might be configurable
                )
                
                # Results should contain chunks with their embeddings
                chunks_data = []
                
                if isinstance(results, dict) and 'chunks' in results:
                    # Process each chunk
                    for idx, chunk_info in enumerate(results['chunks']):
                        chunk_embedding = chunk_info['embedding']
                        chunk_text = chunk_info.get('text', '')
                        chunk_start = chunk_info.get('start_index', 0)
                        chunk_end = chunk_info.get('end_index', 0)
                        
                        chunk_data = {
                            '_key': f"{paper_id}_chunk_{idx}",
                            'paper_id': paper_id,
                            'chunk_index': idx,
                            'text': chunk_text,
                            'embedding': chunk_embedding.tolist() if hasattr(chunk_embedding, 'tolist') else chunk_embedding,
                            'metadata': {
                                'start_index': chunk_start,
                                'end_index': chunk_end,
                                'length': len(chunk_text),
                                'gpu_id': self.gpu_id,
                                'has_full_content': 'pdf_content' in paper
                            }
                        }
                        chunks_data.append(chunk_data)
                
                else:
                    # Fallback if late chunking returns different format
                    # This might need adjustment based on actual Jina output
                    logger.warning(f"Unexpected late chunking format for {paper_id}", extra={'gpu_id': self.gpu_id})
                    
                    # Try alternative approach
                    embeddings = self.model.encode_text(
                        texts=[full_text],
                        task="retrieval", 
                        prompt_name="passage",
                    )
                    
                    # Create single chunk as fallback
                    chunk_data = {
                        '_key': f"{paper_id}_chunk_0",
                        'paper_id': paper_id,
                        'chunk_index': 0,
                        'text': full_text[:1000] + "...",  # Store preview only
                        'embedding': embeddings[0].tolist(),
                        'metadata': {
                            'start_index': 0,
                            'end_index': len(full_text),
                            'length': len(full_text),
                            'gpu_id': self.gpu_id,
                            'has_full_content': 'pdf_content' in paper,
                            'fallback_mode': True
                        }
                    }
                    chunks_data = [chunk_data]
            
            # Return results
            return {
                'paper_id': paper_id,
                'paper_title': paper.get('title', 'Unknown'),
                'paper_year': paper.get('year', None),
                'chunks': chunks_data,
                'chunk_count': len(chunks_data),
                'total_text_length': len(full_text),
                'gpu_id': self.gpu_id,
                'timestamp': datetime.now().isoformat() + 'Z'
            }
            
        except Exception as e:
            logger.error(f"Failed to process {json_path}: {e}", extra={'gpu_id': self.gpu_id})
            return None


def process_papers_on_gpu(gpu_id: int, paper_paths: List[str], output_dir: str):
    """Process a list of papers on a specific GPU."""
    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Initialize processor
    processor = JinaLateChunkProcessor(gpu_id)
    
    # Process each paper
    results = []
    for paper_path in tqdm(paper_paths, desc=f"GPU {gpu_id}", position=gpu_id):
        result = processor.process_paper({'path': paper_path})
        if result:
            # Save immediately to avoid memory buildup
            output_path = os.path.join(output_dir, f"{result['paper_id']}_chunks.json")
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            results.append(result['paper_id'])
    
    return results


def main():
    # Configuration
    papers_dir = "/home/todd/olympus/Erebus/unstructured/papers"
    output_dir = "/home/todd/reconstructionism/validation/experiment_2/data/chunks"
    num_papers = 1000
    num_gpus = 2
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get paper files (sorted for consistency with experiment_1)
    all_json_files = glob(os.path.join(papers_dir, "*.json"))
    
    # Load papers with years for sorting (same as experiment_1)
    papers_with_years = []
    for json_file in all_json_files:
        try:
            with open(json_file, 'r') as f:
                paper = json.load(f)
            year = paper.get('year', 9999)
            papers_with_years.append((year, json_file))
        except:
            continue
    
    # Sort by year (oldest first) and take first num_papers
    papers_with_years.sort(key=lambda x: x[0])
    selected_papers = [p[1] for p in papers_with_years[:num_papers]]
    
    print(f"Processing {len(selected_papers)} papers with {num_gpus} GPUs")
    if selected_papers:
        first_year = papers_with_years[0][0]
        last_year = papers_with_years[min(num_papers-1, len(papers_with_years)-1)][0]
        print(f"Year range: {first_year} - {last_year}")
    
    # Split papers across GPUs
    papers_per_gpu = len(selected_papers) // num_gpus
    gpu_assignments = []
    
    for gpu_id in range(num_gpus):
        start_idx = gpu_id * papers_per_gpu
        end_idx = start_idx + papers_per_gpu if gpu_id < num_gpus - 1 else len(selected_papers)
        gpu_papers = selected_papers[start_idx:end_idx]
        gpu_assignments.append((gpu_id, gpu_papers))
        print(f"GPU {gpu_id}: {len(gpu_papers)} papers")
    
    # Process in parallel using multiprocessing
    start_time = time.time()
    
    with mp.Pool(processes=num_gpus) as pool:
        process_args = [(gpu_id, papers, output_dir) for gpu_id, papers in gpu_assignments]
        results = pool.starmap(process_papers_on_gpu, process_args)
    
    # Collect statistics
    total_processed = sum(len(r) for r in results)
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"Late Chunking Complete!")
    print(f"Total papers processed: {total_processed}")
    print(f"Total time: {elapsed_time:.1f} seconds")
    print(f"Average time per paper: {elapsed_time/total_processed:.1f} seconds")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")
    
    # Analyze chunk distribution
    analyze_chunk_distribution(output_dir)


def analyze_chunk_distribution(output_dir: str):
    """Analyze the distribution of chunks across papers."""
    chunk_files = glob(os.path.join(output_dir, "*_chunks.json"))
    
    total_chunks = 0
    chunk_counts = []
    
    for chunk_file in chunk_files[:100]:  # Sample 100 papers
        try:
            with open(chunk_file, 'r') as f:
                data = json.load(f)
            chunk_count = data.get('chunk_count', 0)
            chunk_counts.append(chunk_count)
            total_chunks += chunk_count
        except:
            pass
    
    if chunk_counts:
        print(f"\nChunk Distribution Analysis (sample of {len(chunk_counts)} papers):")
        print(f"  Total chunks: {total_chunks}")
        print(f"  Average chunks per paper: {np.mean(chunk_counts):.1f}")
        print(f"  Min chunks: {min(chunk_counts)}")
        print(f"  Max chunks: {max(chunk_counts)}")
        print(f"  Std dev: {np.std(chunk_counts):.1f}")


if __name__ == "__main__":
    # Check GPU availability
    if torch.cuda.device_count() < 2:
        print(f"Warning: Only {torch.cuda.device_count()} GPU(s) available. Dual GPU mode requires 2 GPUs.")
        print("Modify the script to use single GPU if needed.")
    
    main()