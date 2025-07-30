#!/usr/bin/env python3
"""
Pipeline Configuration Module

Defines configuration classes for different pipeline variants.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum
import os


class PipelineMode(Enum):
    """Pipeline operation modes."""
    ATOMIC = "atomic"  # Full atomic transactions
    BATCH = "batch"    # Batch processing without transactions
    STREAMING = "streaming"  # Stream processing for large datasets


@dataclass
class PipelineConfig:
    """Configuration for document processing pipelines."""
    
    # Operation mode
    mode: PipelineMode = PipelineMode.ATOMIC
    
    # Database settings
    db_host: str = 'localhost'  # Default to localhost
    db_port: int = 8529  # Default ArangoDB port
    db_name: str = 'arxiv_db'  # Default database name
    
    # Processing settings
    use_atomic_transactions: bool = True
    batch_size: int = 32
    max_retries: int = 3
    
    # GPU settings
    use_gpu: bool = True
    device_ids: Optional[list] = None
    use_fp16: bool = True
    
    # Chunking settings
    chunk_size: int = 1024
    chunk_overlap: int = 200
    max_chunk_length: int = 32768
    
    # Embedding settings
    embedding_model: str = "jinaai/jina-embeddings-v4"
    embedding_dim: int = 2048
    
    # Logging settings
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Performance settings
    enable_memory_cleanup: bool = True
    cleanup_interval: int = 100  # Cleanup every N documents
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "mode": self.mode.value,
            "db_host": self.db_host,
            "db_port": self.db_port,
            "db_name": self.db_name,
            "use_atomic_transactions": self.use_atomic_transactions,
            "batch_size": self.batch_size,
            "max_retries": self.max_retries,
            "use_gpu": self.use_gpu,
            "device_ids": self.device_ids,
            "use_fp16": self.use_fp16,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "max_chunk_length": self.max_chunk_length,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "log_level": self.log_level,
            "log_file": self.log_file,
            "enable_memory_cleanup": self.enable_memory_cleanup,
            "cleanup_interval": self.cleanup_interval
        }
    
    @classmethod
    def from_env(cls, **kwargs) -> 'PipelineConfig':
        """Create config from environment variables."""
        config = cls(**kwargs)
        
        # Load database settings from environment
        config.db_host = os.getenv('ARANGO_HOST', 'localhost')
        config.db_port = int(os.getenv('ARANGO_PORT', '8529'))
        config.db_name = os.getenv('ARANGO_DB_NAME', 'irec_atomic')
        
        # Load other settings from environment if available
        if os.getenv('PIPELINE_MODE'):
            config.mode = PipelineMode(os.getenv('PIPELINE_MODE'))
        if os.getenv('PIPELINE_BATCH_SIZE'):
            config.batch_size = int(os.getenv('PIPELINE_BATCH_SIZE'))
        if os.getenv('PIPELINE_USE_GPU'):
            config.use_gpu = os.getenv('PIPELINE_USE_GPU').lower() == 'true'
        
        return config


# Pre-configured pipeline configurations
ATOMIC_CONFIG = PipelineConfig.from_env(
    mode=PipelineMode.ATOMIC,
    use_atomic_transactions=True,
    log_file="process_atomic.log"
)

BATCH_CONFIG = PipelineConfig.from_env(
    mode=PipelineMode.BATCH,
    use_atomic_transactions=False,
    batch_size=64,
    log_file="process_batch.log"
)

STREAMING_CONFIG = PipelineConfig.from_env(
    mode=PipelineMode.STREAMING,
    use_atomic_transactions=False,
    batch_size=1,
    enable_memory_cleanup=True,
    cleanup_interval=50,
    log_file="process_streaming.log"
)


def get_config(mode: str) -> PipelineConfig:
    """Get pre-configured pipeline configuration by mode name."""
    configs = {
        "atomic": ATOMIC_CONFIG,
        "batch": BATCH_CONFIG,
        "streaming": STREAMING_CONFIG
    }
    
    if mode not in configs:
        raise ValueError(f"Unknown pipeline mode: {mode}. Available: {list(configs.keys())}")
    
    return configs[mode]