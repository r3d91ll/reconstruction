"""
Data Infrastructure Module

Provides data loading, PDF processing, and chunk management utilities
for processing academic papers from arXiv and other sources.
"""

from .arxiv_loader import ArxivLoader
from .document_processor import DocumentProcessor

__all__ = ["ArxivLoader", "DocumentProcessor"]