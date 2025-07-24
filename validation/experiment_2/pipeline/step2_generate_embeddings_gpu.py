#!/usr/bin/env python3
"""
Step 2: Generate embeddings using Jina-v3 on GPU
Optimized for dual A6000 GPUs with batch processing
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
from torch.cuda.amp import autocast
import gc

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GPUEmbeddingGenerator:
    def __init__(self, model_name='jinaai/jina-embeddings-v2-base-en', device_ids=[0, 1], use_fp16=True):
        """
        Initialize GPU-accelerated embedding generator
        
        Args:
            model_name: Jina model to use
            device_ids: List of GPU IDs to use
            use_fp16: Use half precision for memory efficiency
        """
        self.device_ids = device_ids
        self.primary_device = torch.device(f'cuda:{device_ids[0]}')
        self.use_fp16 = use_fp16
        
        logger.info(f"Loading model: {model_name}")
        logger.info(f"Using GPUs: {device_ids}")
        logger.info(f"Half precision: {use_fp16}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Load model
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if use_fp16 else torch.float32
        )
        
        # Move to GPU and set to eval mode
        self.model = self.model.to(self.primary_device)
        self.model.eval()
        
        # Use DataParallel for multi-GPU if available
        if len(device_ids) > 1:
            self.model = nn.DataParallel(self.model, device_ids=device_ids)
            logger.info(f"Using DataParallel across GPUs: {device_ids}")
        
        # Get embedding dimension
        self.embedding_dim = self.model.module.config.hidden_size if hasattr(self.model, 'module') else self.model.config.hidden_size
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def generate_embeddings(self, texts, batch_size=32, show_progress=True):
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            show_progress: Show progress bar
            
        Returns:
            numpy array of embeddings
        """
        all_embeddings = []
        
        # Create progress bar if requested
        iterator = tqdm(range(0, len(texts), batch_size), desc="Generating embeddings") if show_progress else range(0, len(texts), batch_size)
        
        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize with padding and truncation
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,  # Jina v2 max length
                    return_tensors='pt'
                )
                
                # Move to GPU
                inputs = {k: v.to(self.primary_device) for k, v in inputs.items()}
                
                # Generate embeddings with automatic mixed precision
                if self.use_fp16:
                    with autocast():
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)
                
                # Mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
                
                # Normalize embeddings
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                # Move to CPU and convert to numpy
                embeddings_cpu = embeddings.cpu().numpy()
                all_embeddings.append(embeddings_cpu)
                
                # Clear GPU cache periodically
                if i % (batch_size * 10) == 0:
                    torch.cuda.empty_cache()
        
        # Concatenate all embeddings
        return np.vstack(all_embeddings)

def process_chunks_file(input_file, output_file, embedding_generator, batch_size=64):
    """Process a chunks JSON file and add embeddings"""
    
    logger.info(f"Processing: {input_file}")
    
    # Load chunks
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    chunks = data['chunks']
    logger.info(f"Found {len(chunks)} chunks")
    
    # Extract texts
    texts = [chunk['text'] for chunk in chunks]
    
    # Generate embeddings
    embeddings = embedding_generator.generate_embeddings(texts, batch_size=batch_size)
    
    # Add embeddings to chunks
    for i, chunk in enumerate(chunks):
        chunk['embedding'] = embeddings[i].tolist()
    
    # Update data
    data['embedding_model'] = 'jinaai/jina-embeddings-v2-base-en'
    data['embedding_dim'] = embedding_generator.embedding_dim
    data['gpu_processed'] = True
    data['processing_timestamp'] = datetime.now().isoformat()
    
    # Save with embeddings
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved embeddings to: {output_file}")
    
    return len(chunks)

def main():
    # Configuration
    results_dir = os.environ.get("EXP2_RESULTS_DIR", ".")
    chunks_dir = os.path.join(results_dir, "chunks")
    embeddings_dir = os.path.join(results_dir, "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # GPU configuration
    gpu_ids = [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")]
    use_fp16 = os.environ.get("EXP2_USE_FP16", "true").lower() == "true"
    batch_size = int(os.environ.get("EXP2_EMBEDDING_BATCH_SIZE", "64"))
    
    logger.info("=" * 60)
    logger.info("STEP 2: GPU-ACCELERATED EMBEDDING GENERATION")
    logger.info("=" * 60)
    logger.info(f"Chunks directory: {chunks_dir}")
    logger.info(f"Embeddings directory: {embeddings_dir}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Using FP16: {use_fp16}")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    gpu_count = torch.cuda.device_count()
    logger.info(f"\nGPU Configuration:")
    for i in range(gpu_count):
        name = torch.cuda.get_device_name(i)
        memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        logger.info(f"  GPU {i}: {name} ({memory:.1f} GB)")
    
    # Initialize embedding generator
    logger.info("\nInitializing Jina embeddings model on GPU...")
    embedding_generator = GPUEmbeddingGenerator(
        device_ids=gpu_ids,
        use_fp16=use_fp16
    )
    
    # Process all chunk files
    chunk_files = list(Path(chunks_dir).glob("*.json"))
    logger.info(f"\nFound {len(chunk_files)} chunk files to process")
    
    if not chunk_files:
        logger.error("No chunk files found!")
        return 1
    
    # Process statistics
    total_chunks = 0
    total_time = 0
    start_time = datetime.now()
    
    # Process each file
    for i, chunk_file in enumerate(chunk_files, 1):
        logger.info(f"\nProcessing file {i}/{len(chunk_files)}")
        
        output_file = Path(embeddings_dir) / chunk_file.name
        file_start = datetime.now()
        
        try:
            chunks_processed = process_chunks_file(
                chunk_file,
                output_file,
                embedding_generator,
                batch_size=batch_size
            )
            
            file_duration = (datetime.now() - file_start).total_seconds()
            total_chunks += chunks_processed
            total_time += file_duration
            
            # Log progress
            chunks_per_second = chunks_processed / file_duration if file_duration > 0 else 0
            logger.info(f"  Processed {chunks_processed} chunks in {file_duration:.1f}s ({chunks_per_second:.1f} chunks/s)")
            
            # Log GPU memory usage
            for gpu_id in gpu_ids:
                allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                logger.info(f"  GPU {gpu_id} memory: {allocated:.2f} GB")
            
        except Exception as e:
            logger.error(f"Error processing {chunk_file}: {e}")
            continue
        
        # Clear GPU cache between files
        torch.cuda.empty_cache()
        gc.collect()
    
    # Final statistics
    total_duration = (datetime.now() - start_time).total_seconds()
    
    logger.info("\n" + "=" * 60)
    logger.info("EMBEDDING GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total chunks processed: {total_chunks:,}")
    logger.info(f"Total time: {total_duration:.1f} seconds")
    logger.info(f"Average speed: {total_chunks/total_duration:.1f} chunks/second")
    logger.info(f"Files processed: {len(chunk_files)}")
    
    # Save summary
    summary = {
        'total_chunks': total_chunks,
        'total_files': len(chunk_files),
        'total_duration_seconds': total_duration,
        'chunks_per_second': total_chunks / total_duration,
        'embedding_model': 'jinaai/jina-embeddings-v2-base-en',
        'embedding_dim': embedding_generator.embedding_dim,
        'gpu_count': len(gpu_ids),
        'batch_size': batch_size,
        'use_fp16': use_fp16,
        'timestamp': datetime.now().isoformat()
    }
    
    summary_path = os.path.join(embeddings_dir, "embedding_generation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nSummary saved to: {summary_path}")
    logger.info("\n✓ GPU-ACCELERATED EMBEDDING GENERATION COMPLETE")
    
    return 0

if __name__ == "__main__":
    exit(main())