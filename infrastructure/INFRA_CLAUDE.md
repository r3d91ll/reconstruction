# INFRA_CLAUDE.md

This file provides guidance to Claude Code when working with the infrastructure implementation for Information Reconstructionism.

## Infrastructure Overview

The infrastructure provides a robust, scalable system for processing academic documents (particularly arXiv papers) to support the Information Reconstructionism theoretical framework validation.

## Document Processing Pipeline

### Three-Collection Architecture

The pipeline uses a 3-collection architecture for maximum flexibility and data preservation:

1. **metadata** - Pure arXiv metadata
   ```json
   {
     "arxiv_id": "2310.08560",
     "title": "MemGPT: Towards LLMs as Operating Systems",
     "authors": ["Charles Packer", "..."],
     "categories": ["cs.AI"],
     "published": "2023-10-12"
   }
   ```

2. **documents** - Full Docling-extracted markdown with structure
   ```json
   {
     "arxiv_id": "2310.08560",
     "full_text_markdown": "# MemGPT: Towards LLMs...\n\n## Abstract\n\nLarge language models...",
     "document_structure": {"sections": ["Abstract", "Introduction", ...]},
     "extraction_metadata": {"pages": 12, "figures": 3, "tables": 2}
   }
   ```

3. **chunks** - Semantic chunks with embeddings
   ```json
   {
     "chunk_id": "2310.08560_chunk_001",
     "arxiv_id": "2310.08560",
     "text": "Large language models have revolutionized...",
     "embedding": [0.123, 0.456, ...],
     "chunk_index": 1,
     "chunk_metadata": {
       "section": "Introduction",
       "has_equations": false,
       "chunk_type": "text"
     }
   }
   ```

### Processing Flow

```
1. Load PDF + arXiv metadata
   ↓
2. Extract markdown with Docling (preserves structure, formatting)
   ↓
3. Store in ATOMIC TRANSACTION:
   - metadata → metadata collection
   - full markdown → documents collection  
   - enriched text → Jina v4 → chunks collection
   ↓
4. All three collections updated atomically or rollback
```

### Key Design Decisions

1. **Preserve Docling Output**: Full markdown stored exactly as extracted
2. **Atomic Transactions**: All-or-nothing storage prevents partial states
3. **No Duplication**: Each piece of data stored only once
4. **Flexible Chunking**: Can re-chunk from stored documents without re-processing PDFs
5. **Predictable IDs**: Chunks use pattern `{arxiv_id}_chunk_{index}`

### Benefits

- **Re-chunking Flexibility**: Experiment with different chunking strategies
- **Document Structure**: Headers, sections, formatting preserved
- **Multiple Analysis Types**: Document-level vs chunk-level analysis
- **Source of Truth**: Full markdown as canonical version
- **Efficient Storage**: No metadata duplication in chunks

## Core Infrastructure (`irec_infrastructure/`)

The infrastructure is designed as a reusable package that provides:

1. **Data Processing Pipeline**
   - `data/arxiv_loader.py`: Loads arXiv papers from local directories
   - `data/document_processor.py`: Orchestrates document processing workflow
   - Handles PDF extraction, text processing, and metadata management

2. **Embedding Generation**
   - `embeddings/local_jina_gpu.py`: GPU-accelerated local Jina embeddings
   - `embeddings/true_late_chunking.py`: Implements TRUE late chunking strategy
   - `embeddings/batch_processor.py`: Efficient batch processing utilities
   - Produces 2048-dimensional embeddings for documents and chunks

3. **Database Layer**
   - `database/arango_client.py`: ArangoDB interface for graph storage
   - `database/experiment_base.py`: Base class for all experiments
   - Stores documents, chunks, embeddings, and similarity graphs

4. **GPU Pipeline**
   - `gpu/pipeline.py`: Complete GPU-accelerated processing pipeline
   - Handles memory management and batch optimization

### Key Design Principles

1. **Pre-computed Infrastructure**: All heavy computation (embeddings, similarities) is done once and stored
2. **Experiment Isolation**: Each experiment focuses purely on hypothesis testing
3. **GPU Optimization**: Batched processing with memory management for large-scale data
4. **Graph-Based Storage**: ArangoDB enables efficient similarity queries and graph traversal

## Database Schema

### Primary Collections

- `metadata`: Pure arXiv metadata (title, authors, categories, dates)
- `documents`: Full Docling markdown with structure and extraction metadata
- `chunks`: Semantic chunks with embeddings and chunk-specific metadata

### Analysis Collections (Future)

- `paper_embeddings`: Document-level embeddings (2048D)
- `paper_similarities`: Document similarity edges
- `chunk_similarities`: Chunk similarity edges
- `implementations`: GitHub/implementation links
- `citation_network`: Academic citation graph

## Common Development Tasks

### Setting Up the Environment

```bash
# Navigate to infrastructure directory
cd infrastructure/

# Create and activate virtual environment
./setup_venv.sh
source venv/bin/activate
```

### Installing Dependencies

```bash
# Core infrastructure dependencies
pip install -r setup/requirements.txt

# Install irec_infrastructure package in development mode
pip install -e .

# For development with testing/linting
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests with pytest
pytest tests/

# Run tests with coverage
pytest --cov=irec_infrastructure tests/

# Test infrastructure components
python tests/test_infrastructure.py
```

### Code Quality Checks

```bash
# Format code with black
black irec_infrastructure/

# Type checking with mypy
mypy irec_infrastructure/
```

### Processing Documents

```bash
# Process arXiv documents with 3-collection pipeline
python setup/process_documents_three_collections.py \
    --count 10 \
    --db-name irec_three_collections \
    --clean-start

# Process larger batch
python setup/process_documents_three_collections.py \
    --count 1000 \
    --source-dir /mnt/data/arxiv_data/pdf \
    --db-name irec_production \
    --db-host 192.168.1.69
```

**Note**: The three-collection pipeline ensures all collections (metadata, documents, chunks) are updated atomically, preventing partial states.

## Environment Variables

Create a `.env` file:

```bash
# Database
ARANGO_HOST=localhost
ARANGO_PORT=8529
ARANGO_USERNAME=root
ARANGO_PASSWORD=your_password
ARANGO_DATABASE=information_reconstructionism

# GPU Settings
USE_GPU=true
GPU_DEVICES=0,1

# Processing
BATCH_SIZE=32
CHUNK_SIZE=1024
```

## Infrastructure Commands

### Launch a test run (10 documents)
```bash
python setup/process_documents_three_collections.py --count 10 --clean-start
```

### Verify database integrity
```bash
# The pipeline includes built-in verification after processing
python setup/process_documents_three_collections.py --count 0
```

## Important Notes

- Always use atomic transactions to prevent partial states
- The 3-collection architecture enables flexible experimentation
- GPU processing uses both A6000s with NVLink for maximum performance
- All chunk IDs follow predictable pattern for easy reconstruction