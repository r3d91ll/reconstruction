# Three-Collection Architecture Implementation Plan

## Overview

Transform the current 2-collection architecture (metadata + chunks) into a 3-collection architecture (metadata + documents + chunks) to preserve Docling's full markdown output and enable flexible re-chunking.

## Architecture Comparison

### Current (2 collections)
```
metadata: arXiv metadata + num_chunks
chunks: Semantic chunks with embeddings
```

### New (3 collections)
```
metadata: Pure arXiv metadata
documents: Full Docling markdown + structure
chunks: Semantic chunks with embeddings
```

## Implementation Steps

### 1. Update Data Models (`irec_infrastructure/models/metadata.py`)

Add new model:
```python
class DocumentRecord(BaseModel):
    """Full document storage from Docling."""
    arxiv_id: str
    full_text_markdown: str
    document_structure: Dict[str, List[str]]  # sections, headers, etc.
    extraction_metadata: Dict[str, Any]  # pages, figures, tables, etc.
    extraction_timestamp: datetime
    docling_version: str = "0.1.0"
```

### 2. Update Base Pipeline (`base_pipeline.py`)

Add documents collection to schema:
```python
'documents': [
    {'type': 'hash', 'fields': ['arxiv_id'], 'unique': True},
    {'type': 'fulltext', 'fields': ['full_text_markdown']},
    {'type': 'persistent', 'fields': ['extraction_metadata.pages']},
    {'type': 'persistent', 'fields': ['document_structure.sections[*]']}
]
```

### 3. Update Atomic Pipeline (`process_documents_atomic.py`)

#### Modify Processing Flow

Current:
```python
# Process enriched document → chunks → store (metadata + chunks)
```

New:
```python
# Extract with Docling → store document → create enriched → chunks → store all
```

#### Update Transaction

```python
transaction = db.begin_transaction(
    write_collections=['metadata', 'documents', 'chunks'],
    read_collections=[]
)
```

#### Split Storage Logic

```python
# 1. Store pure metadata
metadata_record = {
    '_key': arxiv_id,
    'arxiv_id': arxiv_id,
    'title': metadata.title,
    'authors': metadata.authors,
    'categories': metadata.categories,
    'published': metadata.published,
    # NO num_chunks here anymore
}

# 2. Store full document
document_record = {
    '_key': arxiv_id,
    'arxiv_id': arxiv_id,
    'full_text_markdown': docling_result.document.export_to_markdown(),
    'document_structure': extract_structure(docling_result),
    'extraction_metadata': {
        'pages': docling_result.pages,
        'figures': len(docling_result.figures),
        'tables': len(docling_result.tables),
        'extraction_timestamp': datetime.now()
    }
}

# 3. Store chunks (unchanged)
chunk_records = [...]
```

### 4. Add Document Structure Extraction

```python
def extract_document_structure(docling_result) -> Dict[str, List[str]]:
    """Extract document structure from Docling output."""
    return {
        'sections': [s.title for s in docling_result.sections],
        'headers': extract_headers(docling_result),
        'figure_captions': [f.caption for f in docling_result.figures],
        'table_captions': [t.caption for t in docling_result.tables]
    }
```

### 5. Update Document Reconstruction

Add options for reconstruction:
```python
def reconstruct_document(self, arxiv_id: str, method='full') -> Optional[Dict]:
    """
    Reconstruct document.
    
    Args:
        method: 'full' (from documents), 'chunks' (from chunks), 'hybrid'
    """
    if method == 'full':
        # Fast: just return stored document
        return self.db.collection('documents').get(arxiv_id)
    elif method == 'chunks':
        # Current method: reconstruct from chunks
        return self._reconstruct_from_chunks(arxiv_id)
    elif method == 'hybrid':
        # Use structure from documents + content from chunks
        return self._hybrid_reconstruction(arxiv_id)
```

### 6. Enable Re-chunking Capability

```python
def rechunk_document(self, arxiv_id: str, chunk_size: int = 1024) -> List[Dict]:
    """
    Re-chunk a document with different parameters without re-processing PDF.
    """
    # Get stored document
    doc = self.db.collection('documents').get(arxiv_id)
    if not doc:
        return []
    
    # Apply new chunking strategy
    new_chunks = self.chunk_with_params(
        doc['full_text_markdown'], 
        chunk_size=chunk_size
    )
    
    return new_chunks
```

## Benefits Realized

1. **PDF Processing Once**: Extract with Docling once, experiment many times
2. **Preserve Structure**: Section headers, formatting, all preserved
3. **Flexible Analysis**: Can do document-level OR chunk-level analysis
4. **Re-chunking**: Try different chunk sizes, overlap strategies
5. **Debugging**: Can see exactly what Docling extracted
6. **Future-proof**: Ready for document-level embeddings

## Migration Strategy

Since we're bootstrapping:
1. Drop existing databases (as confirmed by user)
2. Implement new 3-collection architecture
3. Process documents fresh with new pipeline
4. No migration needed - clean start

## Testing Plan

1. Process 10 documents with new architecture
2. Verify all 3 collections populated correctly
3. Test reconstruction methods (full, chunks, hybrid)
4. Verify atomic transactions work with 3 collections
5. Test re-chunking capability

## Success Metrics

- All 3 collections update atomically
- Document structure preserved perfectly
- Can reconstruct documents 3 ways
- Re-chunking works without PDF access
- No partial states possible