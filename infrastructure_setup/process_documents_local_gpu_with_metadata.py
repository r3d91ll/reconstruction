#!/usr/bin/env python3
"""
Process Documents with TRUE Late Chunking using LOCAL GPU
INCLUDING arXiv metadata integration

This script:
1. Uses Docling to extract FULL document text (NO chunking)
2. Uses LOCAL Jina model on dual A6000 GPUs for late chunking
3. Integrates arXiv metadata for each document
4. Saves combined results for database loading

This runs entirely locally - no cloud API calls needed!
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from tqdm import tqdm

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat

# Import our local GPU implementation
import sys
sys.path.append(str(Path(__file__).parent.parent))
from irec_infrastructure.embeddings.local_jina_gpu import create_local_jina_processor


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetadataAwareProcessor:
    """
    Implements TRUE late chunking with metadata integration.
    
    Key principle: Docling extracts full documents, local Jina model does ALL chunking,
    and we integrate arXiv metadata for rich context.
    """
    
    def __init__(self, metadata_dir: Path):
        """
        Initialize processor with local GPU model and metadata access.
        
        Args:
            metadata_dir: Directory containing arXiv metadata JSON files
        """
        logger.info("Initializing Metadata-Aware LOCAL GPU processor...")
        
        self.metadata_dir = metadata_dir
        
        # Local GPU Jina processor
        self.jina_processor = create_local_jina_processor()
        
        # Docling for FULL document extraction only
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    use_ocr=False  # Disable OCR for speed
                )
            }
        )
        
        logger.info("Processor ready with dual A6000 GPUs and metadata integration!")
        
    def load_metadata(self, pdf_path: Path) -> Dict:
        """
        Load arXiv metadata for a given PDF.
        
        Args:
            pdf_path: Path to PDF document
            
        Returns:
            Dictionary with metadata or empty dict if not found
        """
        # Get corresponding metadata file
        metadata_file = self.metadata_dir / f"{pdf_path.stem}.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading metadata for {pdf_path.name}: {e}")
                return {}
        else:
            logger.warning(f"No metadata found for {pdf_path.name}")
            return {}
        
    def process_document(self, pdf_path: Path) -> Dict:
        """
        Process a single document with true late chunking and metadata.
        
        Args:
            pdf_path: Path to PDF document
            
        Returns:
            Dictionary with document text, chunks, embeddings, and metadata
        """
        try:
            # Step 1: Load arXiv metadata
            metadata = self.load_metadata(pdf_path)
            arxiv_id = metadata.get('arxiv_id', pdf_path.stem)
            
            logger.info(f"Processing {pdf_path.name} (arXiv ID: {arxiv_id})")
            
            # Step 2: Extract FULL document text with Docling
            doc_result = self.converter.convert(pdf_path)
            
            # Get complete document text - NO CHUNKING HERE!
            full_text = doc_result.document.export_to_markdown()
            
            # Step 3: Process with LOCAL GPU late chunking
            logger.info(f"Processing with local GPU ({len(full_text)} chars)...")
            
            jina_result = self.jina_processor.encode_with_late_chunking(
                full_text,
                return_chunks=True
            )
            
            # Step 4: Format results with metadata
            chunks = jina_result['chunks']
            embeddings = jina_result['embeddings']
            
            logger.info(f"Created {len(chunks)} semantic chunks for {arxiv_id}")
            
            # Format chunks with metadata
            formatted_chunks = []
            for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
                formatted_chunks.append({
                    'chunk_id': f"{arxiv_id}_chunk_{i}",
                    'chunk_index': i,
                    'text': chunk_text,
                    'embedding': embedding,  # Already a list
                    'tokens': len(chunk_text.split()),  # Rough estimate
                    'arxiv_id': arxiv_id  # Link back to document
                })
            
            # Combine all information
            return {
                'success': True,
                'document_id': arxiv_id,
                'pdf_path': str(pdf_path),
                'metadata': {
                    'arxiv_id': metadata.get('arxiv_id', arxiv_id),
                    'title': metadata.get('title', 'Unknown'),
                    'authors': metadata.get('authors', []),
                    'abstract': metadata.get('abstract', ''),
                    'categories': metadata.get('categories', []),
                    'published': metadata.get('published', ''),
                    'updated': metadata.get('updated', ''),
                    'pdf_url': metadata.get('pdf_url', ''),
                    'abs_url': metadata.get('abs_url', '')
                },
                'processing': {
                    'extraction_timestamp': datetime.now().isoformat(),
                    'text_length': len(full_text),
                    'processing_mode': 'local_gpu_with_metadata'
                },
                'full_text': full_text,
                'chunks': formatted_chunks,
                'embeddings': embeddings,
                'num_chunks': len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return {
                'success': False,
                'document_id': pdf_path.stem,
                'error': str(e)
            }
    
    def process_documents_batch(
        self,
        pdf_paths: List[Path],
        output_dir: Path,
        resume: bool = True
    ) -> Dict:
        """
        Process multiple documents with metadata integration.
        
        Args:
            pdf_paths: List of PDF paths
            output_dir: Directory to save results
            resume: Whether to resume from previous run
            
        Returns:
            Processing statistics
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for existing progress
        processed = set()
        if resume:
            for json_file in output_dir.glob("*.json"):
                processed.add(json_file.stem)
        
        # Filter out already processed
        remaining = [p for p in pdf_paths if p.stem not in processed]
        
        logger.info(f"Processing {len(remaining)} documents "
                   f"({len(processed)} already completed)")
        
        stats = {
            'total': len(pdf_paths),
            'processed': len(processed),
            'failed': 0,
            'total_chunks': 0,
            'with_metadata': 0,
            'start_time': datetime.now().isoformat(),
            'processing_mode': 'local_gpu_with_metadata',
            'gpu_count': 2,
            'gpu_type': 'A6000'
        }
        
        # Process each document
        for pdf_path in tqdm(remaining, desc="Processing with metadata"):
            result = self.process_document(pdf_path)
            
            if result['success']:
                # Save result
                output_file = output_dir / f"{result['document_id']}.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                stats['processed'] += 1
                stats['total_chunks'] += result['num_chunks']
                
                # Check if metadata was found
                if result['metadata'].get('title') != 'Unknown':
                    stats['with_metadata'] += 1
            else:
                logger.error(f"Failed: {result.get('error')}")
                stats['failed'] += 1
        
        stats['end_time'] = datetime.now().isoformat()
        
        # Save statistics
        stats_file = output_dir / "processing_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Completed: {stats['processed']} successful, "
                   f"{stats['failed']} failed, "
                   f"{stats['total_chunks']} total chunks, "
                   f"{stats['with_metadata']} with metadata")
        
        return stats
    
    def create_collection_views(self, output_dir: Path) -> Dict[str, List[str]]:
        """
        Create collection views based on metadata categories.
        
        Args:
            output_dir: Directory with processed documents
            
        Returns:
            Dictionary mapping categories to document lists
        """
        collections = {}
        
        for json_file in output_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                if data.get('success') and data.get('metadata'):
                    categories = data['metadata'].get('categories', [])
                    arxiv_id = data['document_id']
                    
                    # Add to each category collection
                    for category in categories:
                        if category not in collections:
                            collections[category] = []
                        collections[category].append(arxiv_id)
                        
            except Exception as e:
                logger.error(f"Error reading {json_file}: {e}")
        
        # Save collection mapping
        collections_file = output_dir / "collections_mapping.json"
        with open(collections_file, 'w') as f:
            json.dump(collections, f, indent=2)
        
        logger.info(f"Created {len(collections)} collection views")
        return collections


def main():
    """Run local GPU late chunking processing with metadata."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process documents with metadata integration")
    parser.add_argument("--input-dir", required=True, help="Directory containing PDFs")
    parser.add_argument("--metadata-dir", required=True, help="Directory containing metadata JSONs")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--num-docs", type=int, help="Number of documents to process")
    parser.add_argument("--create-collections", action="store_true", help="Create collection views")
    
    args = parser.parse_args()
    
    # Get PDF files
    input_dir = Path(args.input_dir)
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if args.num_docs:
        pdf_files = pdf_files[:args.num_docs]
    
    logger.info(f"Found {len(pdf_files)} PDFs to process")
    
    # Initialize processor with metadata
    processor = MetadataAwareProcessor(Path(args.metadata_dir))
    
    # Process with metadata integration
    stats = processor.process_documents_batch(
        pdf_paths=pdf_files,
        output_dir=Path(args.output_dir)
    )
    
    # Create collection views if requested
    if args.create_collections:
        collections = processor.create_collection_views(Path(args.output_dir))
        print(f"\nCreated {len(collections)} collection views based on categories")
    
    print(f"\nProcessing complete!")
    print(f"Mode: LOCAL GPU with Metadata Integration")
    print(f"Processed: {stats['processed']}")
    print(f"Failed: {stats['failed']}")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Documents with metadata: {stats['with_metadata']}")


if __name__ == "__main__":
    main()