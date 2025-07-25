#!/usr/bin/env python3
"""
Extract PDF content using Docling for experiment_1.
Adapted to use local Jina V4 model and our pipeline structure.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from tqdm import tqdm
import torch
from transformers import AutoModel
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except ImportError:
    logger.error("Docling not installed. Install with: pip install docling")
    DOCLING_AVAILABLE = False


class LocalJinaEmbedder:
    """Use local Jina V4 model instead of API."""
    
    def __init__(self):
        logger.info("Loading local Jina V4 model...")
        self.model = AutoModel.from_pretrained(
            'jinaai/jina-embeddings-v4',
            trust_remote_code=True
        )
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded on {self.device}")
        logger.info(f"Model config: {self.model.config.hidden_size} dimensions")
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding using local Jina V4."""
        # Truncate if needed (128K token limit)
        max_chars = 128000 * 4  # ~512K characters
        if len(text) > max_chars:
            logger.warning(f"Truncating text from {len(text)} to {max_chars} characters")
            text = text[:max_chars]
        
        # Generate embedding
        embeddings = self.model.encode_text(
            texts=[text],
            task="retrieval",
            prompt_name="passage",
        )
        
        return embeddings[0].tolist()


class DoclingPDFExtractor:
    """
    Extract PDF content using Docling for experiment_1.
    Preserves document structure and relationships.
    """
    
    def __init__(self, papers_dir: str):
        """Initialize the Docling PDF extractor."""
        self.papers_dir = Path(papers_dir)
        self.images_dir = self.papers_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        
        if not DOCLING_AVAILABLE:
            raise RuntimeError("Docling is required but not installed")
        
        # Initialize converter
        self.converter = DocumentConverter()
        
        # Initialize local embedder
        self.embedder = LocalJinaEmbedder()
    
    def extract_pdf_content(self, pdf_path: Path) -> Optional[Dict[str, Any]]:
        """Extract content from PDF preserving structure and relationships."""
        try:
            logger.info(f"Extracting content from {pdf_path.name} with Docling")
            
            # Convert PDF
            result = self.converter.convert(pdf_path)
            
            if not result or not result.document:
                logger.error(f"Failed to convert {pdf_path}")
                return None
            
            # Export to markdown
            markdown_with_images = result.document.export_to_markdown()
            
            # Get structured representation
            doc_dict = result.document.export_to_dict()
            
            # Extract sections with proper hierarchy
            sections = self._extract_sections(doc_dict)
            
            # Extract images with context
            images = self._extract_images_with_context(result, pdf_path)
            
            # Extract tables
            tables = self._extract_tables(doc_dict)
            
            # Extract equations
            equations = self._extract_equations(doc_dict)
            
            # Extract code blocks
            code_blocks = self._extract_code_blocks(markdown_with_images)
            
            # Build content structure matching our requirements
            content = {
                "markdown": markdown_with_images,
                "sections": sections,
                "images": images,
                "tables": tables,
                "equations": equations,
                "code_blocks": code_blocks,
                "references": self._extract_references(doc_dict),
                "metadata": {
                    "num_pages": len(result.pages) if hasattr(result, 'pages') else 0,
                    "num_figures": len(images),
                    "num_tables": len(tables),
                    "num_equations": len(equations),
                    "num_code_blocks": len(code_blocks),
                    "extraction_timestamp": datetime.now().isoformat() + "Z",
                    "extraction_method": "docling"
                }
            }
            
            return content
            
        except Exception as e:
            logger.error(f"Failed to extract content from {pdf_path}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _extract_sections(self, doc_dict: Dict) -> List[Dict[str, Any]]:
        """Extract document sections with hierarchy preserved."""
        sections = []
        
        if 'sections' in doc_dict:
            for section in doc_dict['sections']:
                sections.append({
                    'title': section.get('title', ''),
                    'level': section.get('level', 1),
                    'content': section.get('text', ''),
                    'subsections': []
                })
        
        return sections
    
    def _extract_images_with_context(self, result, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract images with contextual information."""
        images = []
        
        # Extract image references from markdown
        import re
        markdown = result.document.export_to_markdown()
        image_pattern = r'!\[([^\]]*)\]\(([^\)]+)\)'
        image_matches = re.findall(image_pattern, markdown)
        
        for idx, (caption, path) in enumerate(image_matches):
            images.append({
                "figure_id": f"fig{idx+1}",
                "caption": caption or f"Figure {idx+1}",
                "path": path,
                "context": "embedded in document",
                "type": "figure"
            })
        
        return images
    
    def _extract_tables(self, doc_dict: Dict) -> List[Dict[str, Any]]:
        """Extract tables with proper formatting."""
        tables = []
        
        try:
            if 'content' in doc_dict and isinstance(doc_dict['content'], list):
                table_count = 0
                for item in doc_dict['content']:
                    if isinstance(item, dict) and item.get('type') == 'table':
                        table_count += 1
                        tables.append({
                            'table_id': f'tab{table_count}',
                            'caption': item.get('caption', f'Table {table_count}'),
                            'markdown': str(item.get('data', '')),
                            'context': ''
                        })
        except Exception as e:
            logger.warning(f"Could not extract tables: {e}")
        
        return tables
    
    def _extract_equations(self, doc_dict: Dict) -> List[Dict[str, Any]]:
        """Extract equations with LaTeX representation."""
        equations = []
        
        if 'equations' in doc_dict:
            for idx, eq in enumerate(doc_dict['equations']):
                equations.append({
                    'eq_id': f'eq{idx+1}',
                    'latex': eq.get('latex', ''),
                    'context': eq.get('context', ''),
                    'type': 'display' if eq.get('display', False) else 'inline'
                })
        
        return equations
    
    def _extract_code_blocks(self, markdown: str) -> List[Dict[str, Any]]:
        """Extract code blocks from markdown."""
        import re
        code_blocks = []
        
        pattern = r'```(\w*)\n(.*?)```'
        matches = re.findall(pattern, markdown, re.DOTALL)
        
        for lang, code in matches:
            if code.strip():
                code_blocks.append({
                    'language': lang or 'unknown',
                    'content': code.strip(),
                    'section': 'extracted from markdown'
                })
        
        return code_blocks
    
    def _extract_references(self, doc_dict: Dict) -> List[Dict[str, Any]]:
        """Extract bibliography/references."""
        references = []
        
        if 'references' in doc_dict:
            for idx, ref in enumerate(doc_dict['references']):
                references.append({
                    'id': f'ref{idx+1}',
                    'text': ref.get('text', ''),
                    'type': ref.get('type', 'paper')
                })
        
        return references
    
    def generate_full_text_with_context(self, content: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """Generate comprehensive text representation for embedding."""
        text_parts = []
        
        # Add metadata
        if 'title' in metadata:
            text_parts.append(f"Title: {metadata['title']}")
        if 'abstract' in metadata:
            text_parts.append(f"Abstract: {metadata['abstract']}")
        if 'categories' in metadata:
            text_parts.append(f"Categories: {', '.join(metadata['categories'])}")
        
        # Add the full markdown content (includes images in context)
        if content and 'markdown' in content:
            text_parts.append("\n# Full Paper Content\n")
            text_parts.append(content['markdown'])
        
        return "\n\n".join(text_parts)
    
    def process_paper(self, pdf_path: Path) -> bool:
        """Process a single paper with Docling extraction."""
        json_path = pdf_path.with_suffix('.json')
        
        if not json_path.exists():
            logger.warning(f"No JSON file for {pdf_path.name}")
            return False
        
        # Load existing metadata
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON {json_path}: {e}")
            return False
        
        # Check if already processed
        if 'pdf_content' in metadata and metadata['pdf_content'].get('markdown'):
            logger.info(f"Already processed: {pdf_path.name}")
            return True
        
        # Extract PDF content with Docling
        content = self.extract_pdf_content(pdf_path)
        
        if not content:
            logger.error(f"Failed to extract content from {pdf_path.name}")
            return False
        
        # Generate full text for embedding
        full_text = self.generate_full_text_with_context(content, metadata)
        logger.info(f"Generated {len(full_text)} characters of text")
        
        # Generate embedding with local model
        embedding = self.embedder.embed(full_text)
        logger.info(f"Generated embedding with {len(embedding)} dimensions")
        
        # Add content and embedding to metadata
        metadata['pdf_content'] = content
        
        # Add embeddings in our expected format
        if 'dimensions' not in metadata:
            metadata['dimensions'] = {}
        if 'WHAT' not in metadata['dimensions']:
            metadata['dimensions']['WHAT'] = {}
        
        metadata['dimensions']['WHAT']['embeddings'] = embedding
        metadata['dimensions']['WHAT']['embedding_dim'] = len(embedding)
        metadata['dimensions']['WHAT']['embedding_method'] = 'jina-v4-docling'
        metadata['dimensions']['WHAT']['context_length'] = len(full_text)
        metadata['dimensions']['WHAT']['has_full_content'] = True
        
        # Also add at root level for compatibility
        metadata['embeddings'] = embedding
        
        # Save enhanced JSON
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Successfully processed {pdf_path.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to save JSON {json_path}: {e}")
            return False
    
    def process_all_papers(self, limit: Optional[int] = None):
        """Process all papers in the directory."""
        pdf_files = list(self.papers_dir.glob("*.pdf"))
        
        if limit:
            pdf_files = pdf_files[:limit]
        
        logger.info(f"Found {len(pdf_files)} PDFs to process")
        
        success_count = 0
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            if self.process_paper(pdf_path):
                success_count += 1
            
            # Save progress every 10 papers
            if success_count % 10 == 0 and success_count > 0:
                logger.info(f"Progress: {success_count} papers processed")
        
        logger.info(f"Successfully processed {success_count}/{len(pdf_files)} papers")
        
        # Print summary
        if success_count > 0:
            self.print_summary_stats()
    
    def print_summary_stats(self):
        """Print summary statistics."""
        stats = {
            'total_papers': 0,
            'with_content': 0,
            'with_embeddings': 0,
            'total_images': 0,
            'total_tables': 0,
            'total_equations': 0,
            'total_code_blocks': 0,
            'total_references': 0,
            'avg_content_length': 0
        }
        
        content_lengths = []
        
        for json_path in self.papers_dir.glob("*.json"):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                stats['total_papers'] += 1
                
                if 'pdf_content' in data:
                    stats['with_content'] += 1
                    content = data['pdf_content']
                    if 'metadata' in content:
                        stats['total_images'] += content['metadata'].get('num_figures', 0)
                        stats['total_tables'] += content['metadata'].get('num_tables', 0)
                        stats['total_equations'] += content['metadata'].get('num_equations', 0)
                        stats['total_code_blocks'] += content['metadata'].get('num_code_blocks', 0)
                    if 'references' in content:
                        stats['total_references'] += len(content['references'])
                
                if 'embeddings' in data or ('dimensions' in data and 'WHAT' in data['dimensions']):
                    stats['with_embeddings'] += 1
                    
                if 'dimensions' in data and 'WHAT' in data['dimensions']:
                    context_len = data['dimensions']['WHAT'].get('context_length', 0)
                    if context_len > 0:
                        content_lengths.append(context_len)
                    
            except:
                pass
        
        if content_lengths:
            stats['avg_content_length'] = sum(content_lengths) / len(content_lengths)
        
        logger.info("\n=== Processing Summary ===")
        logger.info(f"Total papers: {stats['total_papers']}")
        logger.info(f"With extracted content: {stats['with_content']}")
        logger.info(f"With embeddings: {stats['with_embeddings']}")
        logger.info(f"Total extracted images: {stats['total_images']}")
        logger.info(f"Total extracted tables: {stats['total_tables']}")
        logger.info(f"Total extracted equations: {stats['total_equations']}")
        logger.info(f"Total extracted code blocks: {stats['total_code_blocks']}")
        logger.info(f"Total extracted references: {stats['total_references']}")
        logger.info(f"Average content length: {stats['avg_content_length']:.0f} characters")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract PDF content using Docling for experiment_1"
    )
    parser.add_argument(
        "--papers-dir",
        type=str,
        default="/home/todd/olympus/Erebus/unstructured/papers",
        help="Directory containing PDFs and JSON files"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of papers to process (for testing)"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only print statistics of existing extractions"
    )
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = DoclingPDFExtractor(args.papers_dir)
    
    if args.stats_only:
        extractor.print_summary_stats()
    else:
        # Process papers
        extractor.process_all_papers(limit=args.limit)


if __name__ == "__main__":
    main()