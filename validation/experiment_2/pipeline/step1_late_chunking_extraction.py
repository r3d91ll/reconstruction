#!/usr/bin/env python3
"""
Step 1: Extract papers with late chunking using Jina V4
Processes papers from the Docling-extracted dataset
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run late chunking extraction for experiment 2"""
    
    # Configuration
    papers_dir = os.environ.get("PAPERS_DIR", "/home/todd/olympus/Erebus/unstructured/papers")
    num_papers = int(os.environ.get("EXP2_NUM_PAPERS", "2000"))
    results_dir = os.environ.get("EXP2_RESULTS_DIR", 
                                os.path.join(os.path.dirname(__file__), "..", "results", "current"))
    
    # Create output directory
    output_dir = os.path.join(results_dir, "chunks")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("EXPERIMENT 2 - STEP 1: LATE CHUNKING EXTRACTION")
    logger.info("=" * 60)
    logger.info(f"Papers directory: {papers_dir}")
    logger.info(f"Target papers: {num_papers}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Start time: {datetime.now()}")
    
    # Get list of papers with PDF content
    json_files = []
    for json_path in Path(papers_dir).glob("*.json"):
        # Check if has PDF content
        try:
            with open(json_path, 'r') as f:
                paper = json.load(f)
            if 'pdf_content' in paper and paper['pdf_content'].get('markdown'):
                json_files.append(str(json_path))
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Error reading {json_path}: {e}")
            continue
    
    logger.info(f"Found {len(json_files)} papers with PDF content")
    
    # Limit to requested number
    if len(json_files) > num_papers:
        json_files = json_files[:num_papers]
        logger.info(f"Limited to {num_papers} papers")
    
    # Check GPU availability
    import torch
    num_gpus = torch.cuda.device_count()
    logger.info(f"Available GPUs: {num_gpus}")
    
    if num_gpus == 0:
        logger.error("No GPUs available! Late chunking requires GPU.")
        return 1
    
    # Run processing
    logger.info("\nStarting late chunking extraction...")
    logger.info("This will generate semantic chunks for each paper")
    
    try:
        # Run the late chunking script
        import subprocess
        
        # Set environment variables for the script
        env = os.environ.copy()
        env['PAPERS_DIR'] = papers_dir
        env['OUTPUT_DIR'] = output_dir
        env['LIMIT'] = str(num_papers)
        
        # Run the simplified single-GPU script
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'generate_late_chunked_embeddings_simple.py'))
        result = subprocess.run(
            [sys.executable, script_path],
            env=env,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Late chunking failed: {result.stderr}")
            return 1
            
        # Log output
        if result.stdout:
            for line in result.stdout.split('\n'):
                if line.strip():
                    logger.info(f"  {line}")
        
        # Verify output
        chunk_files = list(Path(output_dir).glob("*_chunks.json"))
        logger.info(f"\n✓ Generated chunk files: {len(chunk_files)}")
        
        # Sample statistics
        if chunk_files:
            total_chunks = 0
            sample_stats = []
            
            for chunk_file in chunk_files[:10]:  # Sample first 10
                try:
                    with open(chunk_file, 'r') as f:
                        data = json.load(f)
                    
                    num_chunks = len(data.get('chunks', []))
                    total_chunks += num_chunks
                    sample_stats.append(num_chunks)
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Error reading chunk file {chunk_file}: {e}")
                    continue
            
            if sample_stats:
                avg_chunks = sum(sample_stats) / len(sample_stats)
                logger.info(f"\nSample statistics (first {len(sample_stats)} papers):")
                logger.info(f"  Average chunks per paper: {avg_chunks:.1f}")
                logger.info(f"  Min chunks: {min(sample_stats)}")
                logger.info(f"  Max chunks: {max(sample_stats)}")
        
        logger.info("\n✓ STEP 1 COMPLETE: Late chunking extraction finished")
        return 0
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main())