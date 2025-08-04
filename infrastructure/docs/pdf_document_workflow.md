# ArXiv PDF Document Management Workflow

## Overview
This document outlines the workflow for managing ArXiv PDFs, tracking their locations, and enabling on-demand embedding.

## Directory Structure
```
/bulk-store/arxiv-data/
├── pdfs/                    # Final location for extracted PDFs
│   ├── 2301.00001.pdf
│   ├── 2301.00002.pdf
│   └── ...
├── metadata/               # ArXiv metadata JSON
└── tars/                   # Original tar files
    └── extracted/          # Temporary extraction location
```

## Database Schema Extension

### Current Schema (abstracts only)
```javascript
{
  "_key": "2301.00001",
  "id": "2301.00001",
  "title": "...",
  "abstract": "...",
  "authors": [...],
  "categories": [...],
  "embedding": [...],  // Abstract embedding
  "created": "...",
  "updated": "..."
}
```

### Extended Schema (with document tracking)
```javascript
{
  "_key": "2301.00001",
  "id": "2301.00001",
  "title": "...",
  "abstract": "...",
  "authors": [...],
  "categories": [...],
  "embedding": [...],  // Abstract embedding
  "created": "...",
  "updated": "...",
  
  // New fields for document management
  "document": {
    "status": "available|not_found|pending",
    "local_path": "/bulk-store/arxiv-data/pdfs/2301.00001.pdf",
    "arxiv_url": "https://arxiv.org/pdf/2301.00001.pdf",
    "file_size": 1234567,
    "last_checked": "2025-08-02T20:00:00Z",
    "embedded": false,
    "embedding_date": null,
    "chunks": []  // Will contain chunk embeddings when processed
  }
}
```

## Workflow Steps

### Phase 1: PDF Inventory and Indexing
1. **Complete PDF extraction** from tars to `/bulk-store/arxiv-data/tars/extracted/`
2. **Move PDFs** to final location `/bulk-store/arxiv-data/pdfs/`
3. **Index local PDFs**:
   ```python
   # Scan directory and update database
   for pdf in /bulk-store/arxiv-data/pdfs/:
       arxiv_id = extract_arxiv_id(pdf)
       update_document(arxiv_id, {
           "document.status": "available",
           "document.local_path": pdf_path,
           "document.file_size": file_size
       })
   ```

### Phase 2: On-Demand Document Processing
When a document is requested for embedding:

1. **Check local availability**:
   ```python
   doc = db.arxiv_documents.get(arxiv_id)
   if doc.document.status == "available":
       pdf_path = doc.document.local_path
   else:
       # Download from ArXiv
       pdf_path = download_from_arxiv(doc.document.arxiv_url)
   ```

2. **Process PDF** (only when needed):
   - Extract text using Docling
   - Generate chunks (late chunking strategy)
   - Create embeddings with Jina
   - Store in database

3. **Update document status**:
   ```python
   update_document(arxiv_id, {
       "document.embedded": true,
       "document.embedding_date": now(),
       "document.chunks": chunk_embeddings
   })
   ```

## Implementation Components

### 1. PDF Indexer Script (`index_local_pdfs.py`)
- Scans `/bulk-store/arxiv-data/pdfs/`
- Updates database with local file availability
- Runs periodically to catch new additions

### 2. Document Fetcher (`fetch_document.py`)
- Checks local availability first
- Falls back to ArXiv download
- Caches downloaded files locally

### 3. Embedding Pipeline (`embed_on_demand.py`)
- Triggered by user/system request
- Processes single documents or batches
- Updates database with embeddings

### 4. Query Interface
```python
# Find papers about "neural networks" with local PDFs
db.arxiv_documents.find({
    "abstract": {"$regex": "neural network"},
    "document.status": "available"
})

# Find papers that need embedding
db.arxiv_documents.find({
    "document.status": "available",
    "document.embedded": false
})
```

## Benefits
1. **Storage efficient**: Only embed PDFs when needed
2. **Flexible**: Can work with partial local collection
3. **Scalable**: Can process millions of papers incrementally
4. **Fault tolerant**: Can resume from any point

## Next Steps
1. Complete PDF extraction and organization
2. Create PDF indexing script
3. Implement on-demand embedding pipeline
4. Add API endpoints for document requests