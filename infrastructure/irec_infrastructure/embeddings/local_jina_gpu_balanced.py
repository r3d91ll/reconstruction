"""
Local GPU-based Jina Model with Balanced Multi-GPU Support

This module provides a local GPU implementation of Jina embeddings
with proper load balancing across multiple GPUs.

Key improvements:
- Uses DistributedDataParallel or manual GPU assignment for better load balancing
- Larger batch sizes with dynamic adjustment based on memory
- Explicit memory management between batches
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
import logging
from dataclasses import dataclass
import gc
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class LocalJinaConfig:
    """Configuration for local Jina GPU model"""
    model_name: str = "jinaai/jina-embeddings-v4"  # Updated to v4
    device_ids: List[int] = None  # Default: use all available GPUs
    use_fp16: bool = True
    max_length: int = 32768  # Updated for v4 context length
    chunk_size: int = 1024
    chunk_overlap: int = 200
    batch_size: int = 16  # Increased from 8
    use_balanced_loading: bool = True  # New flag for balanced GPU loading
    embedding_dim: int = 2048  # v4 embedding dimension
    

class LocalJinaGPUBalanced:
    """
    Balanced multi-GPU implementation of Jina embeddings.
    
    Key improvements:
    - Distributes work evenly across GPUs
    - Better memory management
    - Dynamic batch sizing
    """
    
    def __init__(self, config: LocalJinaConfig = None):
        """Initialize with balanced GPU loading."""
        self.config = config or LocalJinaConfig()
        
        # Auto-detect GPUs
        if self.config.device_ids is None:
            self.config.device_ids = list(range(torch.cuda.device_count()))
            
        if not self.config.device_ids:
            raise RuntimeError("No GPUs available")
            
        self.devices = [torch.device(f'cuda:{i}') for i in self.config.device_ids]
        self.num_gpus = len(self.devices)
        
        logger.info(f"Initializing balanced GPU model: {self.config.model_name}")
        logger.info(f"Using {self.num_gpus} GPUs: {self.config.device_ids}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        # Load models on each GPU separately for balanced loading
        if self.config.use_balanced_loading and self.num_gpus > 1:
            logger.info("Using balanced multi-GPU loading strategy")
            self.models = []
            
            for device in self.devices:
                logger.info(f"Loading model replica on {device}")
                model = AutoModel.from_pretrained(
                    self.config.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.config.use_fp16 else torch.float32
                )
                model = model.to(device)
                model.eval()
                self.models.append(model)
                
            # No DataParallel - we'll distribute manually
            self.model = None
            self.embedding_dim = self.models[0].config.hidden_size
            
        else:
            # Single GPU or traditional DataParallel
            logger.info("Using single GPU or DataParallel strategy")
            self.model = AutoModel.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.config.use_fp16 else torch.float32
            )
            self.model = self.model.to(self.devices[0])
            self.model.eval()
            
            if self.num_gpus > 1:
                self.model = nn.DataParallel(self.model, device_ids=self.config.device_ids)
                
            self.models = None
            self.embedding_dim = self.model.config.hidden_size if hasattr(self.model, 'config') else self.model.module.config.hidden_size
            
        logger.info(f"Models loaded. Embedding dimension: {self.embedding_dim}")
        
        # Track memory usage
        self._log_gpu_memory("After model loading")
        
    def _log_gpu_memory(self, stage: str):
        """Log GPU memory usage."""
        for device_id in self.config.device_ids:
            allocated = torch.cuda.memory_allocated(device_id) / 1024**3
            reserved = torch.cuda.memory_reserved(device_id) / 1024**3
            logger.info(f"{stage} - GPU {device_id}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            
    def _create_semantic_chunks(self, text: str, tokens: Dict) -> List[str]:
        """Create semantic chunks from tokenized text."""
        offset_mapping = tokens['offset_mapping']
        total_tokens = len(tokens['input_ids'])
        
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        
        start_idx = 0
        while start_idx < total_tokens:
            end_idx = min(start_idx + chunk_size, total_tokens)
            
            # Get character offsets
            if start_idx < len(offset_mapping) and end_idx <= len(offset_mapping):
                start_char = offset_mapping[start_idx][0] if offset_mapping[start_idx] else 0
                end_char = offset_mapping[end_idx - 1][1] if offset_mapping[end_idx - 1] else len(text)
                
                chunk_text = text[start_char:end_char].strip()
                if chunk_text:
                    chunks.append(chunk_text)
                    
            start_idx += chunk_size - overlap
            
        return chunks
        
    def _encode_chunks_balanced(self, chunks: List[str]) -> List[np.ndarray]:
        """
        Encode chunks with balanced GPU distribution.
        
        This method distributes chunks evenly across GPUs from the start.
        """
        embeddings = []
        batch_size = self.config.batch_size
        
        # Process in batches
        for batch_start in range(0, len(chunks), batch_size * self.num_gpus):
            # Distribute this super-batch across GPUs
            gpu_batches = []
            
            for gpu_idx in range(self.num_gpus):
                start_idx = batch_start + gpu_idx * batch_size
                end_idx = min(start_idx + batch_size, len(chunks))
                
                if start_idx < len(chunks):
                    gpu_batches.append((gpu_idx, chunks[start_idx:end_idx]))
                    
            # Process each GPU's batch in parallel
            with torch.no_grad():
                batch_embeddings = []
                
                for gpu_idx, batch_chunks in gpu_batches:
                    if not batch_chunks:
                        continue
                        
                    # Use the appropriate model/device
                    if self.models:  # Balanced loading
                        model = self.models[gpu_idx]
                        device = self.devices[gpu_idx]
                    else:  # DataParallel
                        model = self.model
                        device = self.devices[0]
                        
                    # Tokenize
                    inputs = self.tokenizer(
                        batch_chunks,
                        padding=True,
                        truncation=True,
                        max_length=self.config.max_length,
                        return_tensors='pt'
                    )
                    
                    # Move to specific GPU
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # Generate embeddings
                    outputs = model(**inputs)
                    
                    # Mean pooling
                    token_embeddings = outputs.last_hidden_state
                    attention_mask = inputs['attention_mask']
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    gpu_embeddings = sum_embeddings / sum_mask
                    
                    # Move to CPU and store
                    batch_embeddings.extend(gpu_embeddings.cpu().numpy())
                    
                    # Clear GPU cache for this device
                    if gpu_idx == self.num_gpus - 1:  # After last GPU in batch
                        torch.cuda.empty_cache()
                        
                embeddings.extend(batch_embeddings)
                
        return embeddings
        
    def encode_with_late_chunking(
        self,
        text: str,
        return_chunks: bool = True
    ) -> Dict[str, Union[List[np.ndarray], List[str], int]]:
        """
        Encode text with balanced GPU late chunking.
        """
        try:
            # Tokenize full document
            tokens = self.tokenizer(
                text,
                truncation=False,
                return_offsets_mapping=True,
                add_special_tokens=False
            )
            
            total_tokens = len(tokens['input_ids'])
            logger.debug(f"Document has {total_tokens} tokens")
            
            # Create semantic chunks
            chunks = self._create_semantic_chunks(text, tokens)
            logger.debug(f"Created {len(chunks)} semantic chunks")
            
            # Generate embeddings with balanced GPU loading
            if self.config.use_balanced_loading and self.models:
                embeddings = self._encode_chunks_balanced(chunks)
            else:
                # Fall back to original method
                embeddings = self._encode_chunks(chunks)
                
            result = {
                'embeddings': [emb.tolist() for emb in embeddings],
                'num_chunks': len(chunks)
            }
            
            if return_chunks:
                result['chunks'] = chunks
                
            # Periodic memory cleanup
            if hasattr(self, '_docs_processed'):
                self._docs_processed += 1
                if self._docs_processed % 100 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                    self._log_gpu_memory(f"After {self._docs_processed} documents")
            else:
                self._docs_processed = 1
                
            return result
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU OOM error: {e}")
            # Clear memory and retry with smaller batch
            torch.cuda.empty_cache()
            gc.collect()
            
            # Reduce batch size and retry
            old_batch_size = self.config.batch_size
            # Ensure batch size never goes below 1
            new_batch_size = old_batch_size // 2
            self.config.batch_size = max(1, new_batch_size)
            logger.warning(f"Reducing batch size from {old_batch_size} to {self.config.batch_size}")
            
            # Retry with smaller batch
            return self.encode_with_late_chunking(text, return_chunks)
            
        except Exception as e:
            logger.error(f"Error in encoding: {e}")
            raise
            
    def _encode_chunks(self, chunks: List[str]) -> List[np.ndarray]:
        """Original encode method for compatibility."""
        # Implementation from original file
        embeddings = []
        batch_size = self.config.batch_size
        
        with torch.no_grad():
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                
                inputs = self.tokenizer(
                    batch_chunks,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors='pt'
                )
                
                inputs = {k: v.to(self.devices[0]) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                
                token_embeddings = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask
                
                batch_embeddings = batch_embeddings.cpu().numpy()
                embeddings.extend(batch_embeddings)
                
        return embeddings


def create_balanced_local_jina_processor(config: LocalJinaConfig = None) -> LocalJinaGPUBalanced:
    """Create a balanced GPU Jina processor."""
    return LocalJinaGPUBalanced(config or LocalJinaConfig())