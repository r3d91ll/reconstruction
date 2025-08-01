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

5. **Dual-GPU Processing**
   - `setup/process_abstracts_dual_gpu.py`: Enhanced dual-GPU pipeline with NVLink
   - `setup/process_abstracts_production.py`: Production pipeline with advanced features
   - `launch_dual_gpu_pipeline.py`: Unified launcher with monitoring
   - `gpu_monitor_dashboard.py`: Real-time GPU performance dashboard

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

#### Full Document Pipeline
Process complete PDFs with text extraction and semantic chunking:

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

#### Abstract-Only Pipeline
Rapidly process metadata and abstract embeddings:

```bash
# Process abstracts only (167,000x faster!)
python setup/process_abstracts_only.py \
    --count 1000 \
    --metadata-dir /mnt/data/arxiv_data/metadata \
    --db-name arxiv_abstracts \
    --clean-start

# Process full metadata collection
python setup/process_abstracts_only.py \
    --count 61463 \
    --db-name arxiv_abstracts_full
```

**Note**: The abstract pipeline automatically uses GPU 1, allowing parallel processing with the full document pipeline on GPU 0.

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

### Full Document Processing
```bash
# Test run (10 documents)
python setup/process_documents_three_collections.py --count 10 --clean-start

# Production run
python setup/process_documents_three_collections.py --count 2200 --db-name irec_production
```

### Abstract-Only Processing
```bash
# Test run (100 abstracts)
python setup/process_abstracts_only.py --count 100 --clean-start

# Full metadata collection
python setup/process_abstracts_only.py --count 61463 --db-name arxiv_abstracts_full
```

### Dual-GPU Processing Pipeline

The infrastructure includes an advanced dual-GPU processing pipeline that leverages NVLink for optimal performance:

#### Basic Dual-GPU Processing
```bash
# Launch with monitoring dashboard
python launch_dual_gpu_pipeline.py \
    --db-name arxiv_abstracts_enhanced \
    --metadata-dir /mnt/data/arxiv_data/metadata \
    --batch-size 200 \
    --clean-start

# Resume from checkpoint
python launch_dual_gpu_pipeline.py \
    --db-name arxiv_abstracts_enhanced \
    --resume \
    --checkpoint-dir ./checkpoints/arxiv_abstracts_enhanced
```

#### Production Pipeline (Advanced Features)
```bash
# Use production pipeline with smart batching and predictive load balancing
python launch_dual_gpu_pipeline.py \
    --db-name arxiv_abstracts_production \
    --metadata-dir /mnt/data/arxiv_data/metadata \
    --batch-size 200 \
    --production \
    --checkpoint-dir ./checkpoints/production
```

#### Key Features:
- **NVLink Optimization**: Automatic detection and use of NVLink for faster GPU communication
- **Dynamic Load Balancing**: Intelligent work distribution based on GPU performance
- **Memory Management**: Adaptive batch sizing to prevent OOM errors
- **Checkpoint/Resume**: Robust recovery from interruptions
- **Real-time Monitoring**: GPU metrics dashboard for performance tracking
- **Error Recovery**: Automatic retry with exponential backoff
- **Production Mode**: Advanced memory prediction and smart batching

#### Performance:
- Combined throughput: 400-600 documents/second
- Parallel efficiency: 85-95% with NVLink
- Automatic memory management prevents OOM
- Load balancing ensures optimal GPU utilization

### Parallel Processing (Both GPUs)
```bash
# Terminal 1: Full documents on GPU 0
python setup/process_documents_three_collections.py --count 2200

# Terminal 2: Abstracts on GPU 1
python setup/process_abstracts_only.py --count 61463
```

## Important Notes

- Always use atomic transactions to prevent partial states
- The 3-collection architecture enables flexible experimentation
- GPU processing uses both A6000s with NVLink for maximum performance
- All chunk IDs follow predictable pattern for easy reconstruction
- The dual-GPU pipeline automatically handles load balancing and memory management
- Production pipeline includes advanced features for large-scale processing
- Checkpoint system ensures no work is lost on interruption