#!/usr/bin/env python3
"""
CPU-only PDF extraction script that runs in complete isolation from GPU processes.
This script extracts text from PDFs and saves to intermediate files for GPU processing.
"""

import os
import sys

# CRITICAL: Set environment variables BEFORE any imports
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TORCH_CUDA_ARCH_LIST'] = ''
os.environ['EASYOCR_CUDA'] = '0'
os.environ['CUDA_HOME'] = ''
os.environ['FORCE_CUDA'] = '0'

import json
import logging
from pathlib import Path
from datetime import datetime
import time
from typing import Dict, List, Optional
import hashlib
import re

# Now safe to import after environment is set
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler('pdf_extraction_cpu.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CPUPDFExtractor:
    """PDF extractor that runs purely on CPU"""
    
    def __init__(self, output_dir: str = "./extracted_texts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify CPU mode
        try:
            import torch
            logger.info(f"PyTorch CUDA available: {torch.cuda.is_available()}")
            logger.info(f"PyTorch device count: {torch.cuda.device_count()}")
        except ImportError:
            logger.info("PyTorch not available, continuing with Docling only")
        
        # Initialize Docling
        self.converter = self._init_docling()
        
    def _init_docling(self):
        """Initialize Docling converter"""
        try:
            converter = DocumentConverter()
            
            # Configure for CPU-only operation
            if hasattr(converter, 'format_options'):
                try:
                    converter.format_options[InputFormat.PDF].use_ocr = False  # Disable OCR by default
                    converter.format_options[InputFormat.PDF].extract_tables = True
                    converter.format_options[InputFormat.PDF].extract_figures = False
                    logger.info("Configured Docling for CPU-only operation")
                except Exception as e:
                    logger.warning(f"Could not configure format_options: {e}")
                    
            logger.info("Docling initialized successfully in CPU mode")
            return converter
            
        except Exception as e:
            logger.error(f"Failed to initialize Docling: {e}")
            raise
            
    def extract_pdf(self, pdf_path: Path) -> Optional[Dict]:
        """Extract text and metadata from PDF"""
        try:
            # Convert PDF
            start_time = time.time()
            result = self.converter.convert(str(pdf_path))
            extraction_time = time.time() - start_time
            
            if not result or not hasattr(result, 'document'):
                logger.warning(f"No content extracted from {pdf_path}")
                return None
                
            # Export to markdown
            full_text = result.document.export_to_markdown()
            
            if not full_text or len(full_text.strip()) < 100:
                logger.warning(f"Insufficient text from {pdf_path}: {len(full_text)} chars")
                return None
                
            # Extract metadata
            metadata = self._extract_metadata(result)
            
            # Get arXiv ID from filename
            arxiv_id = pdf_path.stem
            
            # Extract structure
            structure = self._extract_structure(full_text)
            
            return {
                'arxiv_id': arxiv_id,
                'pdf_path': str(pdf_path),
                'full_text': full_text,
                'metadata': metadata,
                'extraction_time': extraction_time,
                'doc_structure': structure,
                'extraction_metadata': {
                    'page_count': len(result.document.pages) if hasattr(result.document, 'pages') else 0,
                    'has_tables': bool(result.document.tables) if hasattr(result.document, 'tables') else False,
                    'char_count': len(full_text),
                    'extraction_method': 'docling_cpu',
                    'pdf_size_mb': pdf_path.stat().st_size / (1024 * 1024),
                    'extracted_at': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting {pdf_path}: {e}")
            return None
            
    def _extract_metadata(self, result) -> Dict:
        """Extract document metadata from Docling result"""
        metadata = {}
        
        try:
            if hasattr(result.document, 'metadata'):
                doc_meta = result.document.metadata
                if hasattr(doc_meta, 'title'):
                    metadata['title'] = doc_meta.title
                if hasattr(doc_meta, 'authors'):
                    metadata['authors'] = doc_meta.authors
                if hasattr(doc_meta, 'date'):
                    metadata['date'] = str(doc_meta.date)
        except Exception as e:
            logger.debug(f"Error extracting metadata: {e}")
            
        return metadata
        
    def _extract_structure(self, text: str) -> Dict:
        """Extract document structure"""
        structure = {
            'sections': [],
            'subsections': [],
            'has_abstract': False,
            'has_introduction': False,
            'has_conclusion': False,
            'has_references': False
        }
        
        try:
            # Find section headers
            sections = re.findall(r'^#+\s+(.+)$', text, re.MULTILINE)
            structure['sections'] = sections[:30]  # Limit to first 30
            
            # Find subsections
            subsections = re.findall(r'^###\s+(.+)$', text, re.MULTILINE)
            structure['subsections'] = subsections[:50]
            
            # Check for common sections
            text_lower = text.lower()
            structure['has_abstract'] = bool(re.search(r'\babstract\b', text_lower[:5000]))
            structure['has_introduction'] = bool(re.search(r'\bintroduction\b', text_lower[:10000]))
            structure['has_conclusion'] = bool(re.search(r'\bconclusion\b', text_lower[-10000:]))
            structure['has_references'] = bool(re.search(r'\breferences\b', text_lower[-15000:]))
            
        except Exception as e:
            logger.debug(f"Error extracting structure: {e}")
            
        return structure
        
    def save_extraction(self, extraction_data: Dict):
        """Save extraction to JSON file"""
        arxiv_id = extraction_data['arxiv_id']
        output_file = self.output_dir / f"{arxiv_id}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(extraction_data, f, ensure_ascii=False, indent=2)
            
        logger.debug(f"Saved extraction to {output_file}")
        
    def process_pdfs(self, pdf_files: List[Path], resume: bool = True):
        """Process multiple PDF files"""
        # Filter already processed if resuming
        if resume:
            processed = set()
            for json_file in self.output_dir.glob("*.json"):
                processed.add(json_file.stem)
                
            pdf_files = [pdf for pdf in pdf_files if pdf.stem not in processed]
            logger.info(f"Resuming: {len(processed)} PDFs already processed")
            
        logger.info(f"Processing {len(pdf_files)} PDF files")
        
        successful = 0
        failed = 0
        
        for pdf_path in tqdm(pdf_files, desc="Extracting PDFs"):
            try:
                # Check PDF size
                pdf_size_mb = pdf_path.stat().st_size / (1024 * 1024)
                if pdf_size_mb > 100:  # 100MB limit
                    logger.warning(f"Skipping large PDF {pdf_path} ({pdf_size_mb:.1f}MB)")
                    failed += 1
                    continue
                    
                # Extract PDF
                extraction = self.extract_pdf(pdf_path)
                
                if extraction:
                    self.save_extraction(extraction)
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
                failed += 1
                
        logger.info(f"Extraction complete: {successful} successful, {failed} failed")
        return successful, failed


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract text from PDFs using CPU only"
    )
    
    parser.add_argument(
        '--pdf-dir',
        type=str,
        default='/mnt/data/arxiv_data/pdf',
        help='Directory containing PDF files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./extracted_texts',
        help='Directory to save extracted texts'
    )
    parser.add_argument(
        '--max-pdfs',
        type=int,
        help='Maximum number of PDFs to process'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from previous run'
    )
    
    args = parser.parse_args()
    
    # Verify CPU mode
    print("Verifying CPU-only mode...")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    
    # Get PDF files
    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        print(f"ERROR: PDF directory not found: {pdf_dir}")
        sys.exit(1)
        
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if args.max_pdfs:
        pdf_files = pdf_files[:args.max_pdfs]
        
    print(f"\nFound {len(pdf_files)} PDF files")
    
    if not pdf_files:
        print("No PDF files found!")
        sys.exit(1)
        
    # Run extraction
    try:
        extractor = CPUPDFExtractor(output_dir=args.output_dir)
        extractor.process_pdfs(pdf_files, resume=args.resume)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Extraction error: {e}", exc_info=True)
        
    print("\n✅ PDF extraction complete!")


if __name__ == "__main__":
    main()