#!/usr/bin/env python3
"""
Base Pipeline Class for Document Processing

This abstract base class provides common functionality for all document processing
pipelines, reducing code duplication and ensuring consistent behavior.
"""

import os
import sys
import json
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from arango import ArangoClient

# Add infrastructure directory to path
infrastructure_dir = Path(__file__).parent.parent
sys.path.append(str(infrastructure_dir))

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from irec_infrastructure.models.metadata import ArxivMetadata, EnrichedDocument

logger = logging.getLogger(__name__)


class BaseDocumentProcessor(ABC):
    """Abstract base class for document processors."""
    
    def __init__(self, metadata_dir: Path):
        """Initialize with metadata directory and processors."""
        self.metadata_dir = metadata_dir
        
        # Docling for text extraction
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(use_ocr=False)
            }
        )
    
    def load_metadata(self, pdf_path: Path) -> Optional[ArxivMetadata]:
        """Load and parse arXiv metadata for a document."""
        metadata_file = self.metadata_dir / f"{pdf_path.stem}.json"
        
        if not metadata_file.exists():
            logger.warning(f"No metadata found for {pdf_path.name}")
            return None
            
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
                
            # Parse dates properly
            if 'published' in data and data['published']:
                try:
                    data['published'] = datetime.fromisoformat(
                        data['published'].replace('Z', '+00:00')
                    )
                except ValueError as e:
                    logger.warning(f"Failed to parse published date: {e}")
                    data['published'] = None
            if 'updated' in data and data['updated']:
                try:
                    data['updated'] = datetime.fromisoformat(
                        data['updated'].replace('Z', '+00:00')
                    )
                except ValueError as e:
                    logger.warning(f"Failed to parse updated date: {e}")
                    data['updated'] = None
                    
            return ArxivMetadata(**data)
            
        except Exception as e:
            logger.error(f"Error loading metadata for {pdf_path.name}: {e}")
            return None
    
    def create_enriched_document(self, pdf_path: Path) -> Optional[EnrichedDocument]:
        """Create enriched document by combining metadata and content."""
        try:
            # Load metadata
            metadata = self.load_metadata(pdf_path)
            if not metadata:
                # Create minimal metadata if not found
                metadata = ArxivMetadata(
                    arxiv_id=pdf_path.stem,
                    title="Unknown Title",
                    authors=["Unknown Author"],
                    abstract="No abstract available",
                    categories=[]
                )
                
            # Extract content with Docling
            logger.info(f"Extracting content from {pdf_path.name}")
            doc_result = self.converter.convert(pdf_path)
            
            if not doc_result or not hasattr(doc_result, 'document'):
                logger.error(f"Failed to extract content from {pdf_path.name}")
                return None
                
            # Get markdown content
            content = doc_result.document.export_to_markdown()
            
            if not content or len(content.strip()) < 100:
                logger.error(f"Insufficient content extracted from {pdf_path.name}")
                return None
                
            # Create enriched document
            enriched_doc = EnrichedDocument(
                arxiv_id=metadata.arxiv_id,
                metadata=metadata,
                content=content
            )
            
            # Generate enriched text for embedding
            enriched_doc.create_enriched_text()
            
            return enriched_doc
            
        except Exception as e:
            logger.error(f"Error creating enriched document for {pdf_path}: {e}")
            return None
    
    def _detect_math_content(self, chunk_text: str) -> Tuple[bool, str]:
        """Detect mathematical content and determine if chunk is primarily math."""
        math_patterns = [
            '\\begin{equation}', '\\end{equation}',
            '\\begin{align}', '\\end{align}',
            '\\begin{gather}', '\\end{gather}',
            '\\[', '\\]',  # Display math
            '\\(', '\\)',  # Inline math
            '$$',  # Display math delimiter
            '$'    # Inline math delimiter
        ]
        
        has_math = any(pattern in chunk_text for pattern in math_patterns)
        chunk_type = 'text'
        
        if has_math and chunk_text:
            # Check if predominantly math
            math_density = sum(chunk_text.count(p) for p in math_patterns) / len(chunk_text)
            if math_density > 0.1:  # More than 10% math markers
                chunk_type = 'equation'
                
        return has_math, chunk_type
    
    def _detect_code_content(self, chunk_text: str) -> Tuple[bool, str]:
        """Detect code content and determine if chunk is primarily code."""
        code_patterns = [
            '```',  # Markdown code blocks
            'def ', 'class ', 'lambda ',  # Python
            'function ', 'const ', 'var ', 'let ', '=>',  # JavaScript/TypeScript
            'public ', 'private ', 'void ', 'static ',  # Java/C#
            'import ', 'from ', 'require(',  # Imports
            '#include', 'using namespace',  # C/C++
            'package ', 'module ',  # Various languages
            'struct ', 'impl ', 'fn ',  # Rust
            'func ', 'type ', 'interface '  # Go
        ]
        
        has_code = any(pattern in chunk_text for pattern in code_patterns)
        chunk_type = 'text'
        
        if has_code:
            # Determine if primarily code
            if '```' in chunk_text:
                chunk_type = 'code'
            elif sum(chunk_text.count(p) for p in code_patterns) > 5:
                chunk_type = 'code'
            else:
                chunk_type = 'mixed'
                
        return has_code, chunk_type
    
    def _detect_figures_and_tables(self, chunk_text: str) -> Tuple[bool, bool, str]:
        """Detect figures and tables, return has_figures, has_tables, and potential chunk_type."""
        figure_patterns = [
            'Figure', 'figure', 'Fig.', 'fig.',
            '\\begin{figure}', '\\includegraphics',
            'plot', 'graph', 'diagram', 'illustration'
        ]
        
        table_patterns = [
            'Table', 'table', 'Tab.',
            '\\begin{table}', '\\begin{tabular}',
            '|---|', '┌', '├', '└'  # ASCII table markers
        ]
        
        has_figures = any(pattern in chunk_text for pattern in figure_patterns)
        has_tables = any(pattern in chunk_text for pattern in table_patterns)
        chunk_type = 'text'
        
        if has_figures and ('\\begin{figure}' in chunk_text or chunk_text.count('Fig') > 2):
            chunk_type = 'figure_caption'
        elif has_tables and ('\\begin{table}' in chunk_text or '\\begin{tabular}' in chunk_text):
            chunk_type = 'table'
            
        return has_figures, has_tables, chunk_type
    
    def _extract_section_header(self, chunk_text: str) -> Optional[str]:
        """Extract section header from chunk text."""
        lines = chunk_text.split('\n')
        
        for line in lines[:10]:  # Check first 10 lines
            line_stripped = line.strip()
            if not line_stripped or len(line_stripped) > 100:
                continue
                
            # Markdown headers
            if line_stripped.startswith('#'):
                return line_stripped.strip('#').strip()
            
            # LaTeX sections
            if any(line_stripped.startswith(cmd) for cmd in [
                '\\section{', '\\subsection{', '\\subsubsection{',
                '\\chapter{', '\\paragraph{'
            ]):
                # Extract section title from LaTeX command
                start = line_stripped.find('{') + 1
                end = line_stripped.rfind('}')
                if start > 0 and end > start:
                    return line_stripped[start:end]
            
            # Numbered sections (e.g., "1. Introduction", "2.3 Methods")
            if re.match(r'^\d+\.?\d*\s+[A-Z][a-zA-Z\s]+$', line_stripped):
                return line_stripped
            
            # All caps headers (but not single words like "USA" or "API")
            if (line_stripped.isupper() and 
                len(line_stripped.split()) > 1 and 
                not any(char.isdigit() for char in line_stripped)):
                return line_stripped.title()
                
        return None
    
    def _determine_chunk_type(self, metadata: Dict) -> str:
        """Determine final chunk type based on all detected content."""
        # Count how many content types were detected
        type_indicators = sum([
            metadata['has_equations'],
            metadata['has_code'],
            metadata['has_figures'],
            metadata['has_tables']
        ])
        
        # If multiple types detected and current type is still 'text', mark as mixed
        if type_indicators > 1 and metadata['chunk_type'] == 'text':
            return 'mixed'
            
        return metadata['chunk_type']
    
    def extract_chunk_metadata(self, chunk_text: str, chunk_index: int) -> Dict:
        """Extract chunk-specific metadata with enhanced detection."""
        metadata = {
            'section': None,
            'has_equations': False,
            'has_code': False,
            'has_figures': False,
            'has_tables': False,
            'chunk_type': 'text'
        }
        
        # Detect different content types
        has_math, math_type = self._detect_math_content(chunk_text)
        metadata['has_equations'] = has_math
        if math_type != 'text':
            metadata['chunk_type'] = math_type
        
        has_code, code_type = self._detect_code_content(chunk_text)
        metadata['has_code'] = has_code
        if code_type != 'text' and metadata['chunk_type'] == 'text':
            metadata['chunk_type'] = code_type
        
        has_figs, has_tabs, fig_tab_type = self._detect_figures_and_tables(chunk_text)
        metadata['has_figures'] = has_figs
        metadata['has_tables'] = has_tabs
        if fig_tab_type != 'text' and metadata['chunk_type'] == 'text':
            metadata['chunk_type'] = fig_tab_type
        
        # Extract section header
        section = self._extract_section_header(chunk_text)
        if section:
            metadata['section'] = section
        
        # Determine final chunk type
        metadata['chunk_type'] = self._determine_chunk_type(metadata)
        
        return metadata
    
    @abstractmethod
    def process_enriched_document(self, enriched_doc: EnrichedDocument) -> Dict:
        """Process enriched document - must be implemented by subclasses."""
        pass


class BasePipeline(ABC):
    """Abstract base class for processing pipelines."""
    
    def __init__(self, db_host: str, db_name: str, metadata_dir: Optional[Path] = None):
        """Initialize pipeline with database connection."""
        self.db_host = db_host
        self.db_port = int(os.environ.get('ARANGO_PORT', '8529'))
        self.db_name = db_name
        self.username = os.environ.get('ARANGO_USERNAME', 'root')
        self.password = os.environ.get('ARANGO_PASSWORD', '')
        
        if not self.password:
            raise ValueError("ARANGO_PASSWORD environment variable not set!")
            
        # Connect to database
        self.client = ArangoClient(hosts=f'http://{self.db_host}:{self.db_port}')
        self.sys_db = self.client.db('_system', username=self.username, password=self.password)
        
        # Default metadata directory
        self.metadata_dir = metadata_dir or Path("/mnt/data/arxiv_data/metadata")
        
        logger.info(f"Initialized pipeline")
        logger.info(f"Database: {self.db_name} at {self.db_host}")
    
    def setup_database(self, clean_start=False):
        """Setup database with metadata and chunks collections."""
        if self.sys_db.has_database(self.db_name):
            if clean_start:
                logger.warning(f"Dropping existing database: {self.db_name}")
                self.sys_db.delete_database(self.db_name)
            else:
                logger.info(f"Using existing database: {self.db_name}")
                self.db = self.client.db(self.db_name, username=self.username, password=self.password)
                return
                
        # Create database
        self.sys_db.create_database(self.db_name)
        self.db = self.client.db(self.db_name, username=self.username, password=self.password)
        
        # Create collections with proper schema
        collections = {
            'metadata': [  # Document metadata and chunk inventory
                {'type': 'hash', 'fields': ['arxiv_id'], 'unique': True},
                {'type': 'persistent', 'fields': ['categories[*]']},
                {'type': 'persistent', 'fields': ['published']},
                {'type': 'fulltext', 'fields': ['title']},
                {'type': 'fulltext', 'fields': ['abstract']}
            ],
            'chunks': [  # Semantic chunks with embeddings
                {'type': 'hash', 'fields': ['chunk_id'], 'unique': True},
                {'type': 'hash', 'fields': ['arxiv_id']},  # Foreign key
                {'type': 'persistent', 'fields': ['chunk_index']},
                {'type': 'persistent', 'fields': ['chunk_metadata.section']},
                {'type': 'persistent', 'fields': ['chunk_metadata.chunk_type']},
                {'type': 'fulltext', 'fields': ['text']}
            ],
            'processing_log': [
                {'type': 'persistent', 'fields': ['batch_id']},
                {'type': 'persistent', 'fields': ['timestamp']},
                {'type': 'persistent', 'fields': ['status']}
            ]
        }
        
        for coll_name, indexes in collections.items():
            collection = self.db.create_collection(coll_name)
            logger.info(f"Created collection: {coll_name}")
            
            for index in indexes:
                if index['type'] == 'hash':
                    collection.add_hash_index(
                        fields=index['fields'],
                        unique=index.get('unique', False)
                    )
                elif index['type'] == 'fulltext':
                    collection.add_fulltext_index(fields=index['fields'])
                elif index['type'] == 'persistent':
                    collection.add_persistent_index(fields=index['fields'])
                    
        logger.info("Database schema created with metadata/chunks architecture")
    
    def reconstruct_document(self, arxiv_id: str) -> Optional[Dict]:
        """
        Reconstruct a document from metadata and chunks.
        
        Uses the efficient chunk ID generation (no need to store all IDs).
        """
        try:
            # Get metadata
            metadata = self.db.collection('metadata').get(arxiv_id)
            if not metadata:
                logger.warning(f"No metadata found for {arxiv_id}")
                return None
                
            # Get all chunks using predictable naming
            chunks = []
            missing_chunks = []
            
            for i in range(metadata['num_chunks']):
                chunk_id = f"{arxiv_id}_chunk_{i}"
                chunk = self.db.collection('chunks').get(chunk_id)
                
                if chunk:
                    chunks.append(chunk)
                else:
                    missing_chunks.append(i)
                    
            # Verify integrity
            if missing_chunks:
                logger.warning(
                    f"Missing chunks for {arxiv_id}: "
                    f"expected {metadata['num_chunks']}, "
                    f"missing indices: {missing_chunks}"
                )
                
            # Sort chunks by index to ensure correct order
            chunks.sort(key=lambda x: x['chunk_index'])
            
            # Reconstruct full text
            full_text = '\n'.join(chunk['text'] for chunk in chunks)
            
            return {
                'arxiv_id': arxiv_id,
                'metadata': metadata,
                'chunks': chunks,
                'num_chunks': len(chunks),
                'expected_chunks': metadata['num_chunks'],
                'full_text': full_text,
                'complete': len(chunks) == metadata['num_chunks']
            }
            
        except Exception as e:
            logger.error(f"Error reconstructing document {arxiv_id}: {e}")
            return None
    
    def verify_document_integrity(self, arxiv_id: str) -> Dict[str, Any]:
        """Verify that a document has all expected chunks."""
        metadata = self.db.collection('metadata').get(arxiv_id)
        if not metadata:
            return {'exists': False, 'error': 'No metadata found'}
            
        # Count actual chunks
        cursor = self.db.aql.execute(
            'FOR c IN chunks FILTER c.arxiv_id == @arxiv_id RETURN c.chunk_id',
            bind_vars={'arxiv_id': arxiv_id}
        )
        actual_chunks = set(cursor)
        
        # Generate expected chunk IDs
        expected_chunks = set(
            f"{arxiv_id}_chunk_{i}" for i in range(metadata['num_chunks'])
        )
        
        # Find missing chunks
        missing = expected_chunks - actual_chunks
        extra = actual_chunks - expected_chunks
        
        return {
            'exists': True,
            'complete': len(missing) == 0 and len(extra) == 0,
            'expected': metadata['num_chunks'],
            'actual': len(actual_chunks),
            'missing': list(missing),
            'extra': list(extra)
        }
    
    def verify_database(self):
        """Verify the database structure and contents."""
        print("\nDatabase Verification:")
        print("="*60)
        
        # Collection statistics
        metadata_count = self.db.collection('metadata').count()
        chunk_count = self.db.collection('chunks').count()
        
        print(f"Metadata records: {metadata_count}")
        print(f"Chunks: {chunk_count}")
        
        if metadata_count > 0:
            # Calculate average chunks per document
            cursor = self.db.aql.execute(
                'FOR m IN metadata RETURN m.num_chunks'
            )
            chunk_counts = list(cursor)
            avg_chunks = sum(chunk_counts) / len(chunk_counts)
            expected_total = sum(chunk_counts)
            
            print(f"Average chunks per document: {avg_chunks:.1f}")
            print(f"Expected total chunks: {expected_total}")
            print(f"Actual total chunks: {chunk_count}")
            print(f"Integrity: {'✓' if expected_total == chunk_count else '✗'}")
            
            # Sample metadata record
            cursor = self.db.aql.execute("""
                FOR m IN metadata
                    LIMIT 1
                    RETURN {
                        arxiv_id: m.arxiv_id,
                        title: m.title,
                        authors_count: LENGTH(m.authors),
                        categories: m.categories,
                        num_chunks: m.num_chunks
                    }
            """)
            
            sample = list(cursor)[0]
            print(f"\nSample metadata record:")
            print(f"  ID: {sample['arxiv_id']}")
            print(f"  Title: {sample['title'][:80]}...")
            print(f"  Authors: {sample['authors_count']}")
            print(f"  Categories: {', '.join(sample['categories'])}")
            print(f"  Chunks: {sample['num_chunks']}")
            
            # Verify sample document integrity
            integrity = self.verify_document_integrity(sample['arxiv_id'])
            print(f"  Integrity check: {'✓' if integrity['complete'] else '✗'}")
            
        # Chunk type distribution
        cursor = self.db.aql.execute("""
            FOR c IN chunks
                COLLECT type = c.chunk_metadata.chunk_type WITH COUNT INTO count
                RETURN {type: type, count: count}
        """)
        
        print("\nChunk type distribution:")
        for item in cursor:
            print(f"  {item['type']}: {item['count']}")
    
    @abstractmethod
    def process_and_store_document(self, pdf_path: Path) -> Dict:
        """Process and store document - must be implemented by subclasses."""
        pass