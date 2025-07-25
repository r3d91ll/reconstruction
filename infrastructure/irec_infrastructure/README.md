# Information Reconstructionism Infrastructure

Core infrastructure package for processing arXiv documents with the three-collection architecture.

## Components

### Data Processing (`data/`)
- `arxiv_loader.py` - Loads arXiv papers from local directories
- `document_processor.py` - Orchestrates document processing workflow

### Embeddings (`embeddings/`)
- `local_jina_gpu.py` - GPU-accelerated local Jina v4 embeddings
- `true_late_chunking.py` - TRUE late chunking implementation
- `batch_processor.py` - Efficient batch processing utilities

### Database (`database/`)
- `arango_client.py` - ArangoDB interface
- `experiment_base.py` - Base class for experiments

### Models (`models/`)
- `metadata.py` - Data models for documents and chunks

### Monitoring (`monitoring/`)
- `progress.py` - Progress tracking utilities

## Installation

```bash
pip install -e .
```

## Three-Collection Architecture

This infrastructure supports the three-collection architecture:
1. **metadata** - Document metadata from all sources
2. **documents** - Full Docling markdown content
3. **chunks** - Semantic chunks with embeddings

## Usage

This package is used by the pipeline scripts in `setup/`. See the setup README for pipeline usage.