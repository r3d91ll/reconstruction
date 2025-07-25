# Infrastructure Setup

This directory contains the three-collection pipeline for processing arXiv documents.

## Main Pipeline

- `process_documents_three_collections.py` - Main pipeline implementing the three-collection architecture
- `base_pipeline.py` - Base classes for document processing
- `pipeline_config.py` - Configuration management
- `setup_irec_database_schema.py` - Database schema setup

## Three-Collection Architecture

1. **metadata** - Document-level metadata from Docling and arXiv
2. **documents** - Full Docling markdown content  
3. **chunks** - Semantic chunks with embeddings from Jina v4

## Usage

1. **Setup Database** (one time):
```bash
python3 setup_irec_database_schema.py --db-name irec_three_collections
```

2. **Process Documents**:
```bash
python3 process_documents_three_collections.py --count 10 --clean-start
```

## Environment Variables

Required (set in .bashrc):
- `ARANGO_USERNAME` - ArangoDB username
- `ARANGO_PASSWORD` - ArangoDB password

## Key Features

- Atomic transactions across all three collections
- Full Docling markdown preservation
- Complete JSON reconstruction capability
- Jina v4 embeddings with TRUE late chunking
- GPU-accelerated processing