# Multi-Source Academic Database MVP Requirements

## Version 5.0 - PDF Processing Pipeline Update

### Executive Summary

Phase 1 (Abstract Processing) has been successfully completed with 99.8 docs/sec performance, exceeding the 25-30 target by 3.3x. This document updates the requirements for Phase 2: PDF Processing Pipeline.

---

## Phase 1: Collection Consolidation (COMPLETED ✅)

### Achieved Performance
- **Processing rate**: 99.8 documents/second (vs 18 baseline)
- **Time**: ~7.7 hours for 2.7M documents (vs 36.5 hours)
- **Architecture**: Single collection with smart batching
- **Batch size**: 100 documents
- **Queue depth**: 7 batches
- **Zero OOM errors**

---

## Phase 2: PDF Processing Pipeline

### Architecture Overview

Directory-based PDF processing pipeline:
1. **Input**: PDFs placed in `/mnt/data-cold/arxiv_data/pdf`
2. **GPU 0**: Docling PDF → Markdown conversion
3. **GPU 1**: Jina embedding of markdown chunks
4. **Process**: Check metadata → Process → Delete PDF
5. **Output**: Orphaned PDFs (no metadata) remain for investigation

### Updated Document Schema

```javascript
// Collection: arxiv_documents (existing)
{
  "_key": "2301.00001",  // arXiv ID
  "title": "...",
  "authors": [...],
  "categories": ["cs.AI", "cs.LG"],
  "abstract": "...",
  "abstract_embedding": [...],  // 1024-dim vector
  "submitted_date": "2023-01-01",
  "updated_date": "2023-01-15",
  
  // PDF tracking (updated)
  "pdf_status": {
    "state": "unprocessed",  // State machine below
    "tar_source": "arXiv_pdf_2301_001.tar",
    "last_updated": null,
    "retry_count": 0,
    "error_message": null,
    "processing_time_seconds": null
  },
  
  // NEW: PDF content fields
  "pdf_content": {
    "markdown": "...",  // Full markdown text
    "chunks": [
      {
        "chunk_id": 0,
        "text": "...",
        "embedding": [...],  // 1024-dim vector
        "metadata": {
          "section": "Introduction",
          "page_start": 1,
          "page_end": 2,
          "char_start": 0,
          "char_end": 2048
        }
      }
      // ... more chunks
    ],
    "extraction_metadata": {
      "docling_version": "...",
      "total_pages": 15,
      "extraction_time": "2025-07-30T20:00:00Z",
      "chunk_count": 25,
      "chunking_strategy": "semantic_2048"
    }
  }
}
```

### PDF Processing Pipeline Design

```python
Pipeline Components:
1. PDFExtractor (GPU 0)
   - Extract PDFs from tar archives
   - Convert to markdown using Docling
   - Clean up PDF files after processing

2. ChunkProcessor
   - Semantic chunking (2048 chars with overlap)
   - Preserve section headers and context
   - Generate chunk metadata

3. EmbeddingWorker (GPU 1)
   - Batch embed chunks using Jina
   - Same optimizations as abstract pipeline

4. DocumentUpdater
   - Update existing documents
   - Add pdf_content field
   - Update pdf_status tracking
```

### Processing Strategy

1. **Directory-Based Queue**
   - PDFs placed in `/mnt/data-cold/arxiv_data/pdf` for processing
   - Each PDF checked against database for metadata
   - Three outcomes:
     - Has metadata + not processed → Process and delete
     - Has metadata + already processed → Delete duplicate
     - No metadata → Leave in place (orphaned)

2. **Resource Management**
   - Process PDFs directly from directory
   - Delete immediately after successful processing
   - Keep only markdown and embeddings in memory

3. **Chunking Strategy**
   - Target chunk size: 2048 characters
   - Overlap: 200 characters
   - Preserve paragraph boundaries
   - Include section context in metadata

### PDF Lifecycle State Machine

```
State Transitions:
- unprocessed → extracting (tar extraction started)
- extracting → extracted (successful PDF extraction)
- extracted → converting (Docling processing)
- converting → chunking (text segmentation)
- chunking → embedding (Jina processing)
- embedding → updating (database update in progress)
- updating → completed (successful)
- any → failed (error with retry_count++)
- failed → unprocessed (retry_count < 3)
- failed → abandoned (retry_count >= 3)

Timeouts:
- extracting → failed (60 seconds)
- converting → failed (300 seconds)
- embedding → failed (120 seconds)
- updating → failed (60 seconds)
```

### Performance Targets

Based on Phase 1 learnings:

| Component | Target | Notes |
|-----------|--------|-------|
| PDF extraction | 10 PDFs/sec | From tar archives |
| Docling conversion | 2 PDFs/sec | GPU 0 bottleneck |
| Chunking | 50 docs/sec | CPU-based |
| Embedding | 100 chunks/sec | Based on abstract performance |
| Overall pipeline | 2 PDFs/sec | Limited by Docling |

Expected processing time: ~385 hours (16 days) for 2.7M PDFs

### Resource Allocation

```yaml
GPU Usage:
  GPU 0: Docling PDF→Markdown (100% during processing)
  GPU 1: Jina embeddings (intermittent based on chunk flow)

Memory:
  System RAM: 64GB active
    - Tar extraction buffer: 10GB
    - Markdown storage: 20GB (50 docs @ 400KB average)
    - Chunk processing: 10GB
    - Pipeline overhead: 24GB

Storage:
  Working space: 50GB (50 PDFs @ 1MB average)
  No permanent PDF storage
```

### Queue Configuration

Learning from Phase 1:
- PDF extraction queue: 3 batches
- Docling queue: 2 batches (GPU 0 memory limited)
- Chunk queue: 10 batches
- Embedding queue: 7 batches (proven in Phase 1)
- Update queue: 5 batches

### Error Handling

1. **Extraction failures**: Skip to next PDF in tar
2. **Conversion failures**: Mark as failed, continue batch
3. **Embedding failures**: Use Phase 1 retry strategy
4. **Update failures**: Implement transaction rollback

### Monitoring Metrics

```python
import threading

class PDFPipelineMetrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.pdfs_extracted = 0
        self.pdfs_converted = 0
        self.chunks_created = 0
        self.chunks_embedded = 0
        self.documents_updated = 0
        self.extraction_errors = {}
        self.conversion_errors = {}
        self.gpu0_utilization = []
        self.gpu1_utilization = []
    
    def increment_counter(self, counter_name, value=1):
        with self._lock:
            setattr(self, counter_name, getattr(self, counter_name) + value)
    
    def add_error(self, error_dict_name, key, error):
        with self._lock:
            error_dict = getattr(self, error_dict_name)
            error_dict[key] = error
    
    def append_utilization(self, gpu_list_name, value):
        with self._lock:
            gpu_list = getattr(self, gpu_list_name)
            gpu_list.append(value)
```

### Implementation Priority

1. **Week 1**: Basic pipeline structure, tar extraction
2. **Week 2**: Docling integration, GPU 0 optimization
3. **Week 3**: Chunking strategy, embedding pipeline
4. **Week 4**: Full integration, error handling
5. **Week 5**: Performance tuning, production deployment

---

## Hardware Utilization Update

### Phase 1 + Phase 2 Combined

When both pipelines run:
- **GPU 0**: Docling (Phase 2)
- **GPU 1**: Jina embeddings (both phases)
- **RAM**: 96GB total (32GB Phase 1 + 64GB Phase 2)
- **CPU**: 20 threads (83% utilization)

### Storage Requirements

- **Database**: 200GB estimated after PDF content
- **Working space**: 50GB for PDF processing
- **Tar archives**: 4TB (existing, read-only)

---

## Success Metrics

### Phase 2 Targets

1. **Processing rate**: 2+ PDFs/second consistently
2. **Chunk quality**: Semantic coherence score > 0.8
3. **Storage efficiency**: < 2x size increase per document
4. **Error rate**: < 1% abandonment rate

### End-to-End Goals

- Complete PDF processing in < 20 days
- Maintain 99%+ document coverage
- Enable full-text semantic search
- Support context-aware retrieval