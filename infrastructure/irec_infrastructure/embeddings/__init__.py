"""
Embeddings Infrastructure Module

Provides document embedding generation using local Jina V4 models with GPU acceleration
and batch processing support.
"""

from .batch_processor import BatchEmbeddingProcessor
from .local_jina_gpu import LocalJinaGPU, LocalJinaConfig, create_local_jina_processor

__all__ = [
    "BatchEmbeddingProcessor",
    "LocalJinaGPU",
    "LocalJinaConfig",
    "create_local_jina_processor"
]