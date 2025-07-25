"""
TRUE Late Chunking Implementation

This module implements late chunking correctly:
1. Docling extracts FULL documents (no chunking)
2. Jina API receives FULL documents and creates semantic chunks

This is fundamentally different from traditional RAG chunking.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat

from .jina_client import JinaClient, JinaConfig
from .local_jina_gpu import LocalJinaGPU, LocalJinaConfig, create_local_jina_processor


logger = logging.getLogger(__name__)


class TrueLateChucker:
    """
    Implements TRUE late chunking where Jina does ALL the chunking.
    
    Key principle: We NEVER chunk documents ourselves. Jina sees the 
    full document and decides optimal chunk boundaries based on semantic
    understanding.
    
    This preserves context safety - the model understands the entire
    document before creating chunks.
    """
    
    def __init__(self, jina_config: JinaConfig = None, use_local_gpu: bool = True):
        """
        Initialize with Jina configuration.
        
        Args:
            jina_config: Jina API configuration with api_key (for cloud API)
            use_local_gpu: Use local GPU implementation instead of API
        """
        self.use_local_gpu = use_local_gpu
        
        if use_local_gpu:
            # Use local GPU implementation
            logger.info("Using LOCAL GPU implementation with dual A6000s")
            self.local_processor = create_local_jina_processor()
        else:
            # Use cloud API
            if not jina_config:
                raise ValueError("jina_config required for cloud API mode")
            self.jina_client = JinaClient(jina_config)
        
        # Docling for FULL document extraction
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    use_ocr=False  # Disable OCR for speed on academic papers
                )
            }
        )
        
        logger.info("Initialized TrueLateChucker - Jina will do ALL chunking")
    
    def process_document(
        self, 
        pdf_path: Union[str, Path]
    ) -> Dict[str, Union[List[Dict], List[float], Dict]]:
        """
        Process document with TRUE late chunking.
        
        Args:
            pdf_path: Path to PDF document
            
        Returns:
            Dictionary containing:
                - full_text: Complete document text
                - chunks: List of semantic chunks created by Jina
                - embeddings: Embeddings for each chunk
                - metadata: Document metadata
        """
        pdf_path = Path(pdf_path)
        
        try:
            # Step 1: Extract FULL document - NO CHUNKING!
            logger.info(f"Extracting full document: {pdf_path.name}")
            doc_result = self.converter.convert(pdf_path)
            
            # Get complete document text
            full_text = doc_result.document.export_to_markdown()
            
            # Document metadata
            metadata = {
                'filename': pdf_path.name,
                'pdf_path': str(pdf_path),
                'extraction_timestamp': datetime.now().isoformat(),
                'full_text_length': len(full_text),
                'full_text_tokens': len(full_text.split())  # Rough estimate
            }
            
            logger.info(f"Extracted {len(full_text)} characters from {pdf_path.name}")
            
            # Step 2: Process with late chunking (local GPU or API)
            if self.use_local_gpu:
                logger.info(f"Processing with LOCAL GPU late chunking...")
                result = self.local_processor.encode_with_late_chunking(full_text)
            else:
                logger.info(f"Sending full document to Jina API for late chunking...")
                result = self.jina_client.encode_with_late_chunking(full_text)
            
            # Step 3: Process results
            chunks = result['chunks']
            embeddings = result['embeddings']
            
            logger.info(f"Jina created {len(chunks)} semantic chunks")
            
            # Format chunks with metadata
            formatted_chunks = []
            for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
                formatted_chunks.append({
                    'chunk_id': i,
                    'text': chunk_text,
                    'embedding': embedding,
                    'tokens': len(chunk_text.split())  # Rough estimate
                })
            
            return {
                'success': True,
                'document_id': pdf_path.stem,
                'metadata': metadata,
                'full_text': full_text,
                'chunks': formatted_chunks,
                'num_chunks': len(chunks),
                'embeddings': embeddings
            }
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return {
                'success': False,
                'document_id': pdf_path.stem,
                'error': str(e),
                'metadata': {},
                'chunks': [],
                'embeddings': []
            }
    
    def process_documents_batch(
        self,
        pdf_paths: List[Union[str, Path]],
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Process multiple documents with true late chunking.
        
        Args:
            pdf_paths: List of PDF paths
            show_progress: Whether to show progress bar
            
        Returns:
            List of processing results
        """
        results = []
        
        if show_progress:
            from tqdm import tqdm
            pdf_paths = tqdm(pdf_paths, desc="Processing with TRUE late chunking")
        
        for pdf_path in pdf_paths:
            result = self.process_document(pdf_path)
            results.append(result)
        
        # Summary statistics
        successful = sum(1 for r in results if r['success'])
        total_chunks = sum(r.get('num_chunks', 0) for r in results)
        
        logger.info(f"Processed {successful}/{len(results)} documents successfully")
        logger.info(f"Total chunks created by Jina: {total_chunks}")
        logger.info(f"Average chunks per document: {total_chunks/successful if successful > 0 else 0:.1f}")
        
        return results


# IMPORTANT: This replaces the old LateChucker class
# The old implementation was doing chunking wrong!
class LateChucker:
    """
    DEPRECATED: Use TrueLateChucker instead.
    
    This class name is kept for compatibility but now implements
    TRUE late chunking.
    """
    def __init__(self, *args, **kwargs):
        logger.warning("LateChucker is deprecated. Use TrueLateChucker for TRUE late chunking.")
        # Remove incompatible parameters
        kwargs.pop('use_gpu', None)
        kwargs.pop('target_chunk_size', None)
        kwargs.pop('overlap_size', None)
        kwargs.pop('gpu_id', None)
        
        # Create Jina config if not provided
        if 'jina_config' not in kwargs:
            raise ValueError("TrueLateChucker requires jina_config with API key")
        
        self._true_chunker = TrueLateChucker(kwargs['jina_config'])
    
    def chunk_document(self, pdf_path):
        """Redirect to true implementation."""
        return self._true_chunker.process_document(pdf_path)
    
    def batch_chunk_documents(self, pdf_paths, show_progress=True):
        """Redirect to true implementation."""
        return self._true_chunker.process_documents_batch(pdf_paths, show_progress)