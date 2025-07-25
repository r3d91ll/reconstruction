"""
Embeddings Infrastructure Module

Provides document embedding generation using Jina V4 with support for
late chunking and batch processing.
"""

from .jina_client import JinaClient, JinaConfig
from .late_chunking import LateChucker
from .batch_processor import BatchEmbeddingProcessor
from .true_late_chunking import TrueLateChucker
from .local_jina_gpu import LocalJinaGPU, LocalJinaConfig, create_local_jina_processor

__all__ = [
    "JinaClient", 
    "JinaConfig",
    "LateChucker", 
    "BatchEmbeddingProcessor",
    "TrueLateChucker",
    "LocalJinaGPU",
    "LocalJinaConfig",
    "create_local_jina_processor"
]