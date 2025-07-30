#!/usr/bin/env python3
"""
Adaptive Batch Size Discovery for GPU Embedding Processing
Finds the optimal batch size through binary search, then applies 30% safety margin
"""

import os
import sys
import json
import torch
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModel

# Set environment variable before any imports
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchSizeFinder:
    """Find optimal batch size for Jina embeddings"""
    
    def __init__(self, gpu_id: int = 0, metadata_dir: str = None):
        self.gpu_id = gpu_id
        self.device = f'cuda:{gpu_id}'
        self.model_name = "jinaai/jina-embeddings-v3"
        
        # Set metadata directory from parameter or environment variable
        if metadata_dir:
            self.metadata_dir = Path(metadata_dir)
        else:
            self.metadata_dir = Path(os.environ.get('ARXIV_METADATA_DIR', '/mnt/data-cold/arxiv_data/metadata'))
        
        # Load model
        logger.info(f"Loading Jina model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_name, 
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully")
        
        # Load sample abstracts for testing
        self.test_abstracts = self._load_test_abstracts()
        
    def _load_test_abstracts(self, num_samples: int = 2000) -> List[str]:
        """Load sample abstracts from metadata files"""
        metadata_dir = self.metadata_dir
        abstracts = []
        
        for json_file in list(metadata_dir.glob("*.json"))[:num_samples]:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    abstract = data.get('abstract', '').strip()
                    if len(abstract) > 50:  # Skip very short abstracts
                        abstracts.append(abstract)
            except:
                continue
                
        logger.info(f"Loaded {len(abstracts)} test abstracts")
        return abstracts
        
    def test_batch_size(self, batch_size: int) -> Tuple[bool, float]:
        """Test if a batch size works without OOM"""
        # Clear GPU memory
        torch.cuda.empty_cache()
        
        try:
            # Get batch of abstracts
            test_batch = self.test_abstracts[:batch_size]
            
            # Try to process
            start_mem = torch.cuda.memory_allocated(self.gpu_id) / 1024**3  # GB
            
            inputs = self.tokenizer(
                test_batch,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=8192
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                
            # Force synchronization
            torch.cuda.synchronize()
            
            end_mem = torch.cuda.memory_allocated(self.gpu_id) / 1024**3  # GB
            mem_used = end_mem - start_mem
            
            # Clean up
            del inputs, outputs, embeddings
            torch.cuda.empty_cache()
            
            logger.info(f"Batch size {batch_size}: SUCCESS (used {mem_used:.2f} GB)")
            return True, mem_used
            
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.info(f"Batch size {batch_size}: OOM")
            return False, 0.0
        except Exception as e:
            torch.cuda.empty_cache()
            logger.error(f"Batch size {batch_size}: ERROR - {e}")
            return False, 0.0
            
    def find_optimal_batch_size(self, min_size: int = 1, max_size: int = 1000) -> int:
        """Binary search to find maximum working batch size"""
        logger.info(f"Starting binary search between {min_size} and {max_size}")
        
        # First check if max_size works
        if self.test_batch_size(max_size)[0]:
            return max_size
            
        # Binary search
        best_working = min_size
        left, right = min_size, max_size
        
        while left <= right:
            mid = (left + right) // 2
            
            success, mem_used = self.test_batch_size(mid)
            
            if success:
                best_working = mid
                left = mid + 1  # Try larger
            else:
                right = mid - 1  # Try smaller
                
        logger.info(f"Maximum working batch size: {best_working}")
        
        # Test a few sizes around the boundary for stability
        logger.info("Testing boundary stability...")
        stable_size = best_working
        
        for offset in [-5, -10, -15]:
            test_size = best_working + offset
            if test_size > 0:
                success, _ = self.test_batch_size(test_size)
                if success:
                    stable_size = test_size
                    break
                    
        return stable_size
        
    def recommend_batch_size(self) -> int:
        """Find optimal size and apply 30% safety margin"""
        # Find maximum working size
        max_size = self.find_optimal_batch_size()
        
        # Apply 30% safety margin
        safe_size = int(max_size * 0.7)
        
        logger.info(f"\nResults:")
        logger.info(f"  Maximum batch size: {max_size}")
        logger.info(f"  Recommended size (70%): {safe_size}")
        
        # Verify the safe size works
        logger.info(f"\nVerifying recommended size...")
        success, mem_used = self.test_batch_size(safe_size)
        
        if success:
            logger.info(f"✓ Recommended size {safe_size} verified (uses {mem_used:.2f} GB)")
        else:
            logger.error(f"✗ Recommended size {safe_size} failed! Reducing further...")
            safe_size = int(safe_size * 0.7)
            
        return safe_size


def main():
    """Run batch size discovery"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Find optimal batch size for GPU embedding")
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to test')
    parser.add_argument('--max-test', type=int, default=1000, help='Maximum batch size to test')
    parser.add_argument('--metadata-dir', type=str, default=None, help='Path to metadata directory')
    
    args = parser.parse_args()
    
    # Check GPU availability
    if not torch.cuda.is_available():
        logger.error("No GPU available!")
        sys.exit(1)
        
    logger.info(f"GPU: {torch.cuda.get_device_name(args.gpu)}")
    logger.info(f"Memory: {torch.cuda.get_device_properties(args.gpu).total_memory / 1024**3:.1f} GB")
    
    # Find optimal batch size
    finder = BatchSizeFinder(gpu_id=args.gpu, metadata_dir=args.metadata_dir)
    optimal_size = finder.recommend_batch_size()
    
    # Save recommendation
    config_update = {
        'batch_size': optimal_size,
        'gpu_id': args.gpu,
        'gpu_name': torch.cuda.get_device_name(args.gpu),
        'discovery_method': 'binary_search_70_percent'
    }
    
    with open('optimal_batch_config.json', 'w') as f:
        json.dump(config_update, f, indent=2)
        
    logger.info(f"\nConfiguration saved to optimal_batch_config.json")
    logger.info(f"Update your pipeline config with: batch_size={optimal_size}")
    
    
if __name__ == '__main__':
    main()