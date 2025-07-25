# Information Reconstructionism Infrastructure

This module contains validated, reusable infrastructure components extracted from successful experiments. All code here has been tested on large-scale document processing tasks.

## Overview

The `irec_infrastructure` package provides battle-tested components for:

- **GPU Management**: Multi-GPU coordination, memory optimization
- **Embeddings**: Jina V4 integration with late chunking support  
- **Data Processing**: PDF extraction, chunk management, arXiv loading
- **Database**: ArangoDB interfaces and bulk loading
- **Monitoring**: Progress tracking, metrics, and checkpointing

## Installation

```bash
# From the reconstructionism root directory
pip install -e .
```

## Quick Start

```python
from irec_infrastructure import DocumentProcessor
from irec_infrastructure.embeddings import JinaClient

# Process documents with validated pipeline
processor = DocumentProcessor(
    gpu_count=2,
    batch_size=64
)

results = processor.process_documents(
    input_dir="/mnt/data/arxiv_data/pdf",
    num_documents=2000,
    chunking_strategy="late"
)
```

## Module Structure

### GPU (`irec_infrastructure.gpu`)
- `GPUPipeline`: Orchestrates multi-GPU processing
- `GPUMemoryManager`: Optimizes memory usage
- `MultiGPUCoordinator`: Handles work distribution

### Embeddings (`irec_infrastructure.embeddings`)
- `JinaClient`: Jina V4 API interface
- `LateChucker`: Semantic late chunking implementation
- `BatchEmbeddingProcessor`: Efficient batch processing

### Data (`irec_infrastructure.data`)
- `PDFProcessor`: Extract text from PDFs using Docling
- `ChunkManager`: Create and manage document chunks
- `ArxivLoader`: Load papers from arXiv dataset
- `DocumentProcessor`: High-level document processing

### Database (`irec_infrastructure.database`)
- `ArangoClient`: Database connection management
- `DatabaseSchema`: Schema definitions
- `BulkLoader`: Efficient bulk data loading

### Monitoring (`irec_infrastructure.monitoring`)
- `ProgressTracker`: Track processing progress
- `MetricsCollector`: Collect performance metrics
- `CheckpointManager`: Enable resume capability

## Validated Performance

Based on processing 4000+ arXiv documents:

- **PDF Processing**: ~100 documents/minute with 2 GPUs
- **Embedding Generation**: ~50-200 chunks per document
- **Memory Usage**: ~4GB GPU memory per 1000 chunks
- **Error Recovery**: Automatic retry on transient failures

## Best Practices

1. **Always use batch processing** for efficiency
2. **Monitor GPU memory** to avoid OOM errors
3. **Enable checkpointing** for long runs
4. **Use validated batch sizes** (64 for embeddings)

## Testing

```bash
# Run infrastructure tests
pytest irec_infrastructure/tests/
```

## Contributing

When adding new infrastructure:
1. Ensure it's experiment-agnostic
2. Add comprehensive docstrings
3. Include performance metrics
4. Write unit tests