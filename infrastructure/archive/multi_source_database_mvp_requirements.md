# Multi-Source Academic Database MVP Requirements

## Version 3.0 - Single Collection Architecture with PDF Lifecycle Management

### Executive Summary

Build a scalable, multi-source academic database using a knowledge silo architecture with single collections per source. This design eliminates redundancy, optimizes write throughput by 3x, and implements intelligent PDF lifecycle management. The unified catalog approach enables fast cross-source similarity search while maintaining data locality for efficient processing.

---

## Architecture Overview

### Three-Layer Architecture

1. **Knowledge Silos**: Single collection per source (arxiv_documents, pubmed_documents)
2. **Unified Catalog**: RAM-cached Library of Congress organized abstracts
3. **Analytical Graphs**: On-demand relationship networks for focused research

### Container Strategy

**Phase 1 - Knowledge Silos**: Single collection per source
```
- arxiv_documents      # All arXiv data in one collection
- pubmed_documents     # All PubMed data in one collection
- github_repositories  # All GitHub data in one collection
```

**Phase 3 - Analytical Layer**: Purpose-built collections
```
- unified_catalog      # Library of Congress organized catalog
- citation_network     # Cross-source citation graph
- semantic_bridges     # CONVEYANCE relationships
```

---

## Phase 1: MVP Requirements (arXiv Base Containers)

### 1.1 Core Infrastructure

#### Processing Pipeline

- **GPU 0**: Embeddings generation for abstracts
- **GPU 1**: Reserved for future PDF processing
- **Current bottleneck**: Database I/O (13.5 docs/sec)
- **Target**: Optimize I/O to achieve 100+ docs/sec throughput

#### Database Schema - Single Collection Architecture

```javascript
// Collection: arxiv_documents (knowledge silo)
{
  "_key": "2301.00001",  // arXiv ID
  "source": "arxiv",
  "title": "...",
  "authors": [...],
  "categories": ["cs.AI", "cs.LG"],
  "abstract": "...",
  "abstract_embedding": [...],  // 1024-dim Jina embedding
  "submitted_date": "2023-01-01",
  "updated_date": "2023-01-15",
  "pdf_url": "https://arxiv.org/pdf/2301.00001.pdf",
  
  // PDF lifecycle tracking
  "pdf_status": {
    "state": "archived",  // archived|extracted|embedded|deleted
    "tar_file": "arXiv_pdf_2301_001.tar",
    "tar_offset": 12345678,
    "extracted_path": null,
    "embedded_date": null
  },
  
  // PDF content (when processed)
  "pdf_content": {
    "markdown": null,
    "embedding": null,
    "chunk_count": null,
    "sections": null,
    "citations": null
  }
}
```

### 1.2 Processing Pipeline v2

#### Key Improvements

1. **I/O Optimized Architecture**

   ```python
   Abstracts → Parser → GPU 0 (Embeddings) → Write Buffer → Batch Writer → DB
   ```

   **Key I/O Optimizations:**
   - Large in-memory write buffers (1GB+)
   - Batch aggregation (10K+ documents)
   - Sequential write patterns
   - Async write pipeline
   - Single collection focus

2. **Checkpoint System**
   - LMDB-based progress tracking
   - Resume capability at any point
   - Batch-level granularity

3. **Memory Management**
   - Streaming processing (no full dataset in memory)
   - Configurable batch sizes
   - GPU memory monitoring

4. **Performance Metrics**
   - Real-time throughput monitoring
   - GPU utilization tracking
   - Queue depth monitoring
   - ETA calculation

---

## Phase 2: PDF Processing and Lifecycle Management

### Storage Architecture

```
/mnt/data-cold/arxiv_data/  # Original tar archives (RAID1)
/mnt/new-raid1/pdfs/        # Temporary extracted PDFs (6TB RAID1)
/mnt/raid0/processing/      # Active PDF→Markdown conversion
```

### PDF Lifecycle Workflow

1. **Extract from tar** → Update document with `extracted_path`
2. **Convert to Markdown** → Process on RAID0 scratch space  
3. **Generate embeddings** → Add to `pdf_content` in document
4. **Delete extracted PDF** → Update status to `embedded`
5. **Re-extraction** → Use tar reference if needed again

### Processing Requirements

- On-demand processing based on research needs
- Batch extraction from tar files
- Lifecycle state tracking
- Automatic cleanup after embedding
- Tar source reference preservation

---

## Phase 3: Analytical Containers

### 3.1 Unified Catalog (Library of Congress Organization)

```javascript
// Collection: unified_catalog
{
  "_key": "LOC:QA76.9.D3:arxiv:2301.00001",
  "classification": "QA76.9.D3",  // Library of Congress
  "title": "...",
  "abstract": "...",
  "abstract_embedding": [...],  // For fast similarity search
  "authors": [
    {"first": "John", "last": "Doe", "affiliation": "..."}
  ],
  "publication_date": "2023-01-01",
  "subjects": ["Machine Learning", "Neural Networks"],
  "sources": [
    {"type": "arxiv", "id": "2301.00001"},
    {"type": "github", "id": "tensorflow/models"}
  ],
  "citations_count": 15000,
  "impact_score": 0.95
}
```

**Key Benefits:**
- Entire catalog fits in RAM (15-20GB)
- Fast similarity search across all abstracts
- Cross-source deduplication
- Standardized classification

### 3.2 Citation Network

```javascript
// Collection: citation_edges
{
  "_from": "unified_papers_metadata/arxiv_2301.00001",
  "_to": "unified_papers_metadata/arxiv_2201.00123",
  "citation_contexts": [...],
  "citation_type": "methodology",
  "strength": 0.85
}
```

### 3.3 Semantic Bridges (CONVEYANCE)

```javascript
// Collection: semantic_bridges
{
  "_from": "unified_catalog/LOC:QA76.9.D3:arxiv:2301.00001",
  "_to": "unified_catalog/LOC:QA76.9.D3:github:transformers",
  "similarity": 0.87,
  "bridge_type": "algorithm_implementation",
  "conveyance_score": 0.92,
  "evidence": [
    "Shared attention mechanism implementation",
    "Referenced in code comments"
  ]
}
```

---

## Implementation Tasks

### Sprint 1: Core Pipeline (Week 1)

- [ ] Create `process_abstracts_io_optimized.py` with I/O focus
- [ ] Implement large write buffer system
- [ ] Add I/O throughput monitoring
- [ ] Create checkpoint system with LMDB
- [ ] Optimize ArangoDB write settings

### Sprint 2: Database Schema (Week 1)

- [ ] Design ArangoDB collections with proper naming
- [ ] Create indexes for efficient queries
- [ ] Implement bulk insert optimization
- [ ] Add collection versioning support

### Sprint 3: Testing & Optimization (Week 2)

- [ ] Benchmark current vs optimized I/O patterns
- [ ] Test various batch sizes (1K, 10K, 100K docs)
- [ ] Measure write amplification reduction
- [ ] Profile RocksDB compaction impact

### Sprint 4: Monitoring & Operations (Week 2)

- [ ] Create monitoring dashboard
- [ ] Add alerting for failures
- [ ] Document operational procedures
- [ ] Create backup/restore procedures

---

## Success Metrics

### Performance

- Process 2.5M abstracts with optimized I/O
- Achieve 100+ docs/sec sustained write throughput
- Reduce write amplification by 10x
- Maintain single GPU at optimal utilization
- <1% error rate
- Full resume capability

### Data Quality

- 100% metadata preservation
- Accurate embeddings generation
- Proper linking between collections
- Traceable data lineage

### Scalability

- Easy addition of new sources (PubMed, JSTOR)
- Container-based isolation
- Incremental processing support
- Clear upgrade path

---

## Technical Specifications

### Hardware Requirements

- 2x NVIDIA A6000 GPUs (48GB each)
- 256GB System RAM (for catalog caching)
- 4TB NVMe RAID0 (processing scratch)
- 4TB SATA RAID1 (cold tar storage)
- 6TB SATA RAID1 (PDF lifecycle storage)
- 10TB+ for database

### Software Stack

- Python 3.10+
- ArangoDB 3.11+
- CUDA 12.0+
- Jina Embeddings v3
- Docling (for PDF processing)

### Performance Targets

- Abstract embedding: 500 docs/second (single GPU)
- Database insertion: 100+ docs/second (3x improvement)
- Single collection writes: 33% reduction in I/O
- Catalog search: <2 seconds for 2.5M documents
- PDF extraction: On-demand, ~5 seconds per file
- Memory usage: 15-20GB for full catalog cache

---

## Risk Mitigation

### Technical Risks

1. **GPU Memory Overflow**
   - Solution: Adaptive batch sizing
   - Monitoring: Real-time memory tracking

2. **Database Bottlenecks**
   - Solution: Bulk insert optimization
   - Monitoring: Insert queue depth

3. **Network Interruptions**
   - Solution: Checkpoint system
   - Recovery: Automatic resume

### Data Risks

1. **Schema Evolution**
   - Solution: Versioned containers
   - Migration: Scripted upgrades

2. **Source Format Changes**
   - Solution: Adapter pattern
   - Validation: Schema validation

---

## Future Roadmap

### Q3 2024

- PubMed integration
- GitHub code repository processing
- Initial CONVEYANCE analysis

### Q4 2024

- JSTOR integration
- Full citation network analysis
- Production deployment

### Q1 2025

- Additional sources (ACM, IEEE)
- Advanced analytics
- API development

---

## Appendix A: Code Structure

### Current Development Structure (Single File)

```
infrastructure/setup/
├── process_abstracts_io_optimized.py  # Single-file I/O optimized pipeline
├── configs/
│   ├── arxiv_schema.yaml             # arXiv container schemas
│   ├── pipeline_config.yaml          # Processing configuration
│   └── io_optimization.yaml          # I/O tuning parameters
└── scripts/
    ├── migrate_v1_to_v2.py           # Migration script
    ├── monitor_pipeline.py           # Real-time monitoring
    └── validate_data.py              # Data integrity checks
```

### Class Structure within Single File

```python
# process_abstracts_io_optimized.py

class WriteBufferManager:
    """Manages large in-memory write buffers for batch operations"""
    
class BatchWriter:
    """Handles optimized batch writing to ArangoDB"""
    
class CheckpointManager:
    """LMDB-based checkpoint and resume functionality"""
    
class IOMetricsCollector:
    """Tracks I/O performance metrics and throughput"""
    
class DatabaseOptimizer:
    """ArangoDB connection and write optimization settings"""
    
class EmbeddingProcessor:
    """Handles Jina embeddings generation on GPU"""
    
class PDFLifecycleManager:
    """Manages PDF extraction, processing, and cleanup"""
    
class ArxivProcessor:
    """Main orchestrator class for the pipeline"""
```

### Future Modular Structure

When ready to break out into modules:

```
infrastructure/setup/
├── process_abstracts_io_optimized.py  # Main orchestrator only
├── lib/
│   ├── write_buffer_manager.py        # WriteBufferManager class
│   ├── batch_writer.py                # BatchWriter class
│   ├── checkpoint_manager.py          # CheckpointManager class
│   ├── io_metrics_collector.py        # IOMetricsCollector class
│   ├── database_optimizer.py          # DatabaseOptimizer class
│   └── embedding_processor.py         # EmbeddingProcessor class
```

**Development Philosophy:**

- Keep everything in one file for rapid iteration
- Use proper classes and methods for clean separation
- Design with future modularization in mind
- Each class should have clear responsibilities
- Minimal inter-class dependencies

---

## Appendix B: Migration Plan

### From v1 to v2

1. Export current data with proper arxiv_ prefix
2. Recreate collections with new schema
3. Bulk import with validation
4. Verify data integrity
5. Update application code

### Estimated Timeline

- Export: 2 hours
- Schema creation: 30 minutes
- Import abstracts with I/O optimization: <24 hours
- PDF processing: On-demand basis
- Catalog generation: 2-4 hours
- Validation: 2 hours
- Total: ~25 hours

---

## Sign-off

**Prepared by**: Claude Assistant  
**Date**: July 30, 2025  
**Version**: 3.0  
**Status**: Draft for Review

### Approval

- [ ] Technical Lead: _________________
- [ ] Database Architect: _________________
- [ ] Project Manager: _________________

---

## Notes

This MVP focuses on establishing the foundation for a multi-source academic database. The container-based architecture ensures clean separation of concerns while the I/O optimized pipeline addresses the real bottleneck in processing millions of documents. The design prioritizes database write performance, data integrity, and reserves GPU resources for future PDF processing needs.
