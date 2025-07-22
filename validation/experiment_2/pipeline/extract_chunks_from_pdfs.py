#!/usr/bin/env python3
"""
Extract chunks from papers that already have Docling PDF content.
This reuses the PDF extraction from experiment_1.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import os
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChunkExtractor:
    """Extract semantic chunks from Docling-processed papers."""
    
    def __init__(self, papers_dir: str, output_dir: str, strategy: str = "sections"):
        self.papers_dir = Path(papers_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.strategy = strategy
        
    def extract_section_chunks(self, paper: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract chunks based on document sections."""
        chunks = []
        
        if 'pdf_content' not in paper:
            return chunks
            
        pdf_content = paper['pdf_content']
        paper_id = paper.get('id', 'unknown')
        
        # Extract introduction if exists
        if 'markdown' in pdf_content:
            # This is simplified - in reality we'd parse the markdown structure
            sections = self._split_markdown_sections(pdf_content['markdown'])
            
            for idx, section in enumerate(sections):
                chunk = {
                    '_key': f"{paper_id}_chunk_{idx}",
                    'paper_id': paper_id,
                    'chunk_index': idx,
                    'chunk_type': section['type'],
                    'title': section['title'],
                    'content': section['content'],
                    'metadata': {
                        'extraction_method': 'section-based',
                        'has_figures': len(section.get('figures', [])) > 0,
                        'has_tables': len(section.get('tables', [])) > 0,
                        'has_equations': len(section.get('equations', [])) > 0,
                        'word_count': len(section['content'].split())
                    }
                }
                chunks.append(chunk)
        
        # Add figure chunks
        if 'images' in pdf_content:
            for fig_idx, figure in enumerate(pdf_content['images']):
                chunk = {
                    '_key': f"{paper_id}_fig_{fig_idx}",
                    'paper_id': paper_id,
                    'chunk_index': 1000 + fig_idx,  # High index for figures
                    'chunk_type': 'figure',
                    'title': figure.get('caption', f'Figure {fig_idx+1}'),
                    'content': figure.get('caption', '') + '\n' + figure.get('context', ''),
                    'metadata': {
                        'extraction_method': 'figure',
                        'figure_id': figure.get('figure_id'),
                        'figure_type': figure.get('type', 'unknown')
                    }
                }
                chunks.append(chunk)
        
        # Add table chunks
        if 'tables' in pdf_content:
            for tab_idx, table in enumerate(pdf_content['tables']):
                chunk = {
                    '_key': f"{paper_id}_tab_{tab_idx}",
                    'paper_id': paper_id,
                    'chunk_index': 2000 + tab_idx,  # High index for tables
                    'chunk_type': 'table',
                    'title': table.get('caption', f'Table {tab_idx+1}'),
                    'content': table.get('caption', '') + '\n' + table.get('markdown', ''),
                    'metadata': {
                        'extraction_method': 'table',
                        'table_id': table.get('table_id')
                    }
                }
                chunks.append(chunk)
                
        # Add code chunks
        if 'code_blocks' in pdf_content:
            for code_idx, code in enumerate(pdf_content['code_blocks']):
                chunk = {
                    '_key': f"{paper_id}_code_{code_idx}",
                    'paper_id': paper_id,
                    'chunk_index': 3000 + code_idx,  # High index for code
                    'chunk_type': 'code',
                    'title': f"Code Block {code_idx+1} ({code.get('language', 'unknown')})",
                    'content': code.get('content', ''),
                    'metadata': {
                        'extraction_method': 'code',
                        'language': code.get('language', 'unknown'),
                        'section': code.get('section', 'unknown')
                    }
                }
                chunks.append(chunk)
        
        return chunks
    
    def _split_markdown_sections(self, markdown: str) -> List[Dict[str, Any]]:
        """Split markdown into semantic sections."""
        # Simple section detection based on headers
        lines = markdown.split('\n')
        sections = []
        current_section = None
        
        section_keywords = {
            'abstract': 'abstract',
            'introduction': 'introduction',
            'related work': 'related_work',
            'background': 'background',
            'method': 'methodology',
            'approach': 'methodology',
            'experiment': 'experiments',
            'result': 'results',
            'evaluation': 'results',
            'discussion': 'discussion',
            'conclusion': 'conclusion',
            'future work': 'future_work',
            'reference': 'references',
            'acknowledge': 'acknowledgments'
        }
        
        for line in lines:
            # Check if it's a header
            if line.startswith('#'):
                header_text = line.strip('#').strip().lower()
                
                # Save previous section
                if current_section:
                    sections.append(current_section)
                
                # Determine section type
                section_type = 'other'
                for keyword, stype in section_keywords.items():
                    if keyword in header_text:
                        section_type = stype
                        break
                
                # Start new section
                current_section = {
                    'title': line.strip('#').strip(),
                    'type': section_type,
                    'content': '',
                    'figures': [],
                    'tables': [],
                    'equations': []
                }
            elif current_section:
                current_section['content'] += line + '\n'
        
        # Don't forget the last section
        if current_section:
            sections.append(current_section)
        
        # If no sections found, treat whole document as one chunk
        if not sections:
            sections.append({
                'title': 'Full Document',
                'type': 'full_document',
                'content': markdown,
                'figures': [],
                'tables': [],
                'equations': []
            })
        
        return sections
    
    def process_papers(self, limit: int = 1000):
        """Process papers and extract chunks."""
        json_files = list(self.papers_dir.glob("*.json"))[:limit]
        
        logger.info(f"Processing {len(json_files)} papers for chunk extraction")
        
        total_chunks = 0
        papers_with_content = 0
        
        for json_file in tqdm(json_files, desc="Extracting chunks"):
            try:
                # Load paper
                with open(json_file, 'r', encoding='utf-8') as f:
                    paper = json.load(f)
                
                # Skip if no PDF content
                if 'pdf_content' not in paper:
                    continue
                    
                papers_with_content += 1
                
                # Extract chunks based on strategy
                if self.strategy == "sections":
                    chunks = self.extract_section_chunks(paper)
                else:
                    raise ValueError(f"Unknown strategy: {self.strategy}")
                
                # Save chunks
                if chunks:
                    paper_id = paper.get('id', json_file.stem)
                    output_file = self.output_dir / f"{paper_id}_chunks.json"
                    
                    chunk_data = {
                        'paper_id': paper_id,
                        'paper_title': paper.get('title', 'Unknown'),
                        'paper_year': paper.get('year', None),
                        'chunk_count': len(chunks),
                        'chunks': chunks,
                        'extraction_timestamp': datetime.now().isoformat() + 'Z'
                    }
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(chunk_data, f, indent=2)
                    
                    total_chunks += len(chunks)
                    
            except Exception as e:
                logger.error(f"Failed to process {json_file}: {e}")
        
        logger.info(f"\nExtraction complete:")
        logger.info(f"  Papers with PDF content: {papers_with_content}")
        logger.info(f"  Total chunks extracted: {total_chunks}")
        logger.info(f"  Average chunks per paper: {total_chunks/papers_with_content:.1f}")
        logger.info(f"  Output directory: {self.output_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract chunks from Docling-processed papers")
    parser.add_argument("--papers-dir", default="/home/todd/olympus/Erebus/unstructured/papers",
                       help="Directory with JSON papers")
    parser.add_argument("--output-dir", default="/home/todd/reconstructionism/validation/experiment_2/data/chunks",
                       help="Output directory for chunks")
    parser.add_argument("--strategy", default="sections", choices=["sections", "semantic", "fixed"],
                       help="Chunking strategy")
    parser.add_argument("--limit", type=int, default=1000,
                       help="Number of papers to process")
    
    args = parser.parse_args()
    
    extractor = ChunkExtractor(args.papers_dir, args.output_dir, args.strategy)
    extractor.process_papers(args.limit)


if __name__ == "__main__":
    main()