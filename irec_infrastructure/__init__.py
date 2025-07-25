"""
Information Reconstructionism Infrastructure Package

A collection of validated infrastructure components for processing academic papers,
generating embeddings, and managing large-scale document analysis pipelines.

This package provides reusable components that have been tested and validated
across multiple experiments in the Information Reconstructionism project.

Modules:
    gpu: GPU pipeline orchestration and memory management
    embeddings: Document embedding generation with Jina V4
    data: PDF processing and data loading utilities
    database: ArangoDB interface and schema management
    monitoring: Progress tracking and performance metrics

Example:
    from irec_infrastructure import DocumentProcessor
    from irec_infrastructure.embeddings import JinaClient
    
    processor = DocumentProcessor()
    embeddings = processor.process_documents(
        input_dir="/mnt/data/arxiv_data/pdf",
        num_documents=2000
    )
"""

__version__ = "0.1.0"
__author__ = "Information Reconstructionism Project"

# Convenience imports
from .data.document_processor import DocumentProcessor
from .embeddings.jina_client import JinaClient

__all__ = [
    "DocumentProcessor",
    "JinaClient",
]