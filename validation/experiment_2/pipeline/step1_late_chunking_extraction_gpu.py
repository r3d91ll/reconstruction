#!/usr/bin/env python3
"""
Step 1: GPU-Accelerated Late Chunking Extraction
Uses Docling with GPU acceleration for PDF processing
"""

import os
import sys
import json
import torch
from pathlib import Path
from datetime import datetime
import logging
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
from tqdm import tqdm
import gc

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GPUAcceleratedExtractor:
    def __init__(self, use_gpu=True, gpu_id=0):
        """
        Initialize GPU-accelerated document extractor
        
        Args:
            use_gpu: Whether to use GPU acceleration
            gpu_id: Which GPU to use for this process
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.gpu_id = gpu_id
        
        if self.use_gpu:
            # Since we set CUDA_VISIBLE_DEVICES in the worker process,
            # we always use device 0 (which maps to the assigned physical GPU)
            torch.cuda.set_device(0)
            actual_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', 'unknown')
            logger.info(f"Worker process using GPU {actual_gpu} (logical device 0)")
        else:
            logger.info("GPU not available, using CPU")
        
        # Configure Docling for optimal performance
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    use_ocr=False,  # Disable OCR for speed
                )
            }
        )
    
    def extract_document(self, pdf_path):
        """Extract and chunk a single document"""
        try:
            # Convert document
            result = self.converter.convert(pdf_path)
            
            # Extract metadata
            metadata = {
                'filename': os.path.basename(pdf_path),
                'path': str(pdf_path),
                'extraction_timestamp': datetime.now().isoformat()
            }
            
            # Extract text and create semantic chunks
            chunks = self.create_semantic_chunks(result)
            
            return {
                'success': True,
                'metadata': metadata,
                'chunks': chunks,
                'num_chunks': len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'path': str(pdf_path)
            }
    
    def create_semantic_chunks(self, result: ConversionResult, 
                             target_chunk_size=1000, 
                             overlap=200):
        """Create semantic chunks from document"""
        chunks = []
        
        # Get full text
        full_text = result.document.export_to_markdown()
        
        # Simple semantic chunking based on paragraphs and sections
        sections = full_text.split('\n\n')
        
        current_chunk = []
        current_size = 0
        current_position = 0  # Track current character position
        
        for section in sections:
            section_size = len(section)
            
            # If section is too large, split it
            if section_size > target_chunk_size:
                # Split by sentences
                sentences = section.split('. ')
                for sentence in sentences:
                    if current_size + len(sentence) > target_chunk_size and current_chunk:
                        # Save current chunk
                        chunk_text = ' '.join(current_chunk)
                        start_pos = current_position
                        end_pos = current_position + len(chunk_text)
                        chunks.append({
                            'text': chunk_text,
                            'start_char': start_pos,
                            'end_char': end_pos,
                            'chunk_id': len(chunks)
                        })
                        current_position = end_pos
                        
                        # Start new chunk with overlap
                        overlap_text = ' '.join(current_chunk[-2:]) if len(current_chunk) > 2 else current_chunk[-1] if current_chunk else ''
                        current_chunk = [overlap_text, sentence] if overlap_text else [sentence]
                        current_size = len(overlap_text) + len(sentence)
                    else:
                        current_chunk.append(sentence)
                        current_size += len(sentence)
            else:
                # Add section to current chunk
                if current_size + section_size > target_chunk_size and current_chunk:
                    # Save current chunk
                    chunk_text = ' '.join(current_chunk)
                    start_pos = current_position
                    end_pos = current_position + len(chunk_text)
                    chunks.append({
                        'text': chunk_text,
                        'start_char': start_pos,
                        'end_char': end_pos,
                        'chunk_id': len(chunks)
                    })
                    current_position = end_pos
                    
                    # Start new chunk
                    current_chunk = [section]
                    current_size = section_size
                else:
                    current_chunk.append(section)
                    current_size += section_size
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            start_pos = current_position
            end_pos = current_position + len(chunk_text)
            chunks.append({
                'text': chunk_text,
                'start_char': start_pos,
                'end_char': end_pos,
                'chunk_id': len(chunks)
            })
        
        return chunks

def process_pdf_batch(pdf_paths, gpu_id, results_dir):
    """Process a batch of PDFs on a specific GPU"""
    # Set CUDA_VISIBLE_DEVICES for this process to use only the assigned GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Force PyTorch to use the specified GPU
    torch.cuda.set_device(0)  # Device 0 in this process maps to the physical GPU we set
    
    # Log GPU assignment
    logger.info(f"Worker process started on GPU {gpu_id} (PID: {os.getpid()})")
    
    # Now when we create the extractor, it will only see one GPU (as device 0)
    extractor = GPUAcceleratedExtractor(use_gpu=True, gpu_id=0)
    results = []
    
    for idx, pdf_path in enumerate(pdf_paths):
        if idx % 5 == 0:
            logger.info(f"GPU {gpu_id}: Processing {idx+1}/{len(pdf_paths)} - {pdf_path.name}")
        result = extractor.extract_document(pdf_path)
        
        if result['success']:
            # Save chunks to file
            paper_id = Path(pdf_path).stem
            output_file = os.path.join(results_dir, f"{paper_id}.json")
            
            output_data = {
                'paper_id': paper_id,
                'metadata': result['metadata'],
                'chunks': result['chunks'],
                'num_chunks': result['num_chunks'],
                'gpu_processed': True,
                'gpu_id': gpu_id
            }
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            results.append({
                'paper_id': paper_id,
                'success': True,
                'num_chunks': result['num_chunks']
            })
        else:
            results.append({
                'paper_id': Path(pdf_path).stem,
                'success': False,
                'error': result['error']
            })
    
    return results

def main():
    # Configuration
    num_papers = int(os.environ.get("EXP2_NUM_PAPERS", "2000"))
    results_dir = os.environ.get("EXP2_RESULTS_DIR", ".")
    chunks_dir = os.path.join(results_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    
    # GPU configuration
    gpu_ids = [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")]
    num_workers = len(gpu_ids)
    
    logger.info("=" * 60)
    logger.info("STEP 1: GPU-ACCELERATED LATE CHUNKING EXTRACTION")
    logger.info("=" * 60)
    logger.info(f"Target papers: {num_papers}")
    logger.info(f"Output directory: {chunks_dir}")
    logger.info(f"Using {num_workers} GPUs: {gpu_ids}")
    
    # Find PDF files
    pdf_dir = Path("/home/todd/olympus/Erebus/unstructured/papers/")
    if not pdf_dir.exists():
        # Try alternative paths
        alt_paths = [
            Path("./validation/data/raw_pdfs"),
            Path("../data/raw_pdfs"),
            Path("../../data/raw_pdfs"),
            Path("validation/experiment_2/data/raw_pdfs")
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                pdf_dir = alt_path
                break
    
    pdf_files = list(pdf_dir.glob("*.pdf"))[:num_papers]
    
    if not pdf_files:
        logger.error(f"No PDF files found in {pdf_dir}")
        return 1
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Split PDFs across GPUs
    batch_size = len(pdf_files) // num_workers
    pdf_batches = []
    
    for i in range(num_workers):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size if i < num_workers - 1 else len(pdf_files)
        pdf_batches.append(pdf_files[start_idx:end_idx])
    
    logger.info(f"Processing in {num_workers} batches of ~{batch_size} PDFs each")
    
    # Process in parallel across GPUs
    start_time = datetime.now()
    all_results = []
    
    # Important: Save the parent process CUDA_VISIBLE_DEVICES
    parent_cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1')
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit batch processing tasks
        futures = []
        for i, (gpu_id, pdf_batch) in enumerate(zip(gpu_ids, pdf_batches)):
            logger.info(f"Submitting batch {i+1} ({len(pdf_batch)} PDFs) to GPU {gpu_id}")
            logger.info(f"  Batch {i+1} will process: {[p.name for p in pdf_batch[:3]]}..." if len(pdf_batch) > 3 else f"  Files: {[p.name for p in pdf_batch]}")
            future = executor.submit(process_pdf_batch, pdf_batch, gpu_id, chunks_dir)
            futures.append((future, i, gpu_id))
        
        # Collect results
        for future, batch_idx, gpu_id in futures:
            batch_results = future.result()
            all_results.extend(batch_results)
            logger.info(f"Batch {batch_idx+1} (GPU {gpu_id}) complete: {len(batch_results)} papers processed")
    
    # Calculate statistics
    duration = (datetime.now() - start_time).total_seconds()
    successful = sum(1 for r in all_results if r['success'])
    failed = len(all_results) - successful
    total_chunks = sum(r.get('num_chunks', 0) for r in all_results if r['success'])
    
    logger.info("\n" + "=" * 60)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total papers processed: {len(all_results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total chunks created: {total_chunks:,}")
    logger.info(f"Average chunks per paper: {total_chunks/successful:.1f}")
    logger.info(f"Total time: {duration:.1f} seconds")
    logger.info(f"Processing rate: {len(all_results)/duration:.1f} papers/second")
    
    # Save summary
    summary = {
        'total_papers': len(all_results),
        'successful': successful,
        'failed': failed,
        'total_chunks': total_chunks,
        'avg_chunks_per_paper': total_chunks / successful if successful > 0 else 0,
        'duration_seconds': duration,
        'papers_per_second': len(all_results) / duration,
        'gpu_count': num_workers,
        'timestamp': datetime.now().isoformat(),
        'results': all_results
    }
    
    summary_path = os.path.join(results_dir, "extraction_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nSummary saved to: {summary_path}")
    logger.info("\n✓ GPU-ACCELERATED EXTRACTION COMPLETE")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    exit(main())