# Information Reconstructionism Infrastructure Pipelines

This directory contains two specialized pipelines for processing arXiv documents, each optimized for different use cases.

## Pipeline Architecture

### 1. Full Document Pipeline (`process_documents_three_collections.py`)
Processes complete PDF documents with Docling extraction and semantic chunking.

**Features:**
- Extracts full text from PDFs using Docling
- Creates semantic chunks with Jina v4 embeddings (2048D)
- Three-collection architecture: `metadata`, `documents`, `chunks`
- Atomic transactions ensure data consistency
- Processing rate: ~0.06 documents/second

**Usage:**
```bash
python process_documents_three_collections.py \
    --count 100 \
    --source-dir /mnt/data/arxiv_data/pdf \
    --db-name irec_three_collections \
    --clean-start
```

**Options:**
- `--count`: Number of documents to process (default: all)
- `--source-dir`: Directory containing PDF files (default: /mnt/data/arxiv_data/pdf)
- `--db-name`: Database name (default: irec_three_collections)
- `--db-host`: Database host (default: 192.168.1.69)
- `--clean-start`: Drop existing database and start fresh

### 2. Abstract-Only Pipeline (`process_abstracts_only.py`)
Rapidly processes metadata and abstract embeddings for large-scale discovery.

**Features:**
- Loads arXiv metadata from JSON files
- Embeds abstracts using Jina v4 (2048D)
- Single collection architecture for simplicity
- Processing rate: ~10,000 abstracts/second
- GPU selection support (automatically uses GPU 1)

**Usage:**
```bash
python process_abstracts_only.py \
    --count 1000 \
    --metadata-dir /mnt/data/arxiv_data/metadata \
    --db-name arxiv_abstracts \
    --clean-start
```

**Options:**
- `--count`: Number of abstracts to process (default: all)
- `--metadata-dir`: Directory containing metadata JSON files (default: /mnt/data/arxiv_data/metadata)
- `--db-name`: Database name (default: arxiv_abstracts)
- `--db-host`: Database host (default: 192.168.1.69)
- `--clean-start`: Drop existing database and start fresh
- `--batch-size`: Batch size for embedding generation (default: 100)

## Parallel Processing Example

Run both pipelines simultaneously on different GPUs:

**Terminal 1 (GPU 0):**
```bash
# Process 2,200 full PDFs on GPU 0 (default)
python process_documents_three_collections.py --count 2200 --db-name irec_full_docs
```

**Terminal 2 (GPU 1):**
```bash
# Process 61,463 abstracts on GPU 1 (automatically configured)
python process_abstracts_only.py --count 61463 --db-name irec_abstracts
```

## Database Schemas

### Three-Collection Schema (Full Documents)
- **metadata**: Combined metadata from arXiv and Docling
- **documents**: Full Docling markdown content
- **chunks**: Semantic chunks with embeddings and metadata

### Single-Collection Schema (Abstracts)
- **abstract_metadata**: arXiv metadata with abstract embeddings

## Environment Variables

Required (set in .bashrc or .env):
- `ARANGO_USERNAME` - ArangoDB username
- `ARANGO_PASSWORD` - ArangoDB password
- `ARANGO_HOST` - Database host (optional, defaults to configured value)
- `ARANGO_PORT` - Database port (optional, defaults to 8529)

## Performance Comparison

| Pipeline | Processing Rate | Use Case |
|----------|----------------|----------|
| Full Documents | ~0.06 docs/sec | Deep analysis, chunk-level search |
| Abstracts Only | ~10,000 docs/sec | Large-scale discovery, rapid indexing |

## Supporting Files

- `base_pipeline.py` - Base classes for document processing
- `pipeline_config.py` - Configuration management
- `requirements.txt` - Python dependencies