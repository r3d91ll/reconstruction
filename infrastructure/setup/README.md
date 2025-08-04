# Information Reconstructionism Infrastructure

Production infrastructure for processing and storing academic paper embeddings at scale.

## Production Pipeline

### ArXiv Processing (`processing_scripts/process_arxiv_production.py`)
High-performance pipeline for processing ArXiv abstracts with GPU-accelerated embeddings.

**Features:**
- Processes ~2.8M ArXiv abstracts from Kaggle dataset
- GPU-optimized with Jina v3 embeddings (1024D)
- Database: `academy_store` / Collection: `base_arxiv`
- Processing rate: 400-500 docs/second on RTX A6000
- Full checkpoint/resume capability
- Comprehensive error handling and retry logic

**Usage:**
```bash
cd processing_scripts
ARANGO_PASSWORD='your_password' python3 process_arxiv_production.py
```

**Configuration:**
- Input: `/fastpool/temp/arxiv-metadata-oai-snapshot.json`
- Database Host: 192.168.1.69
- GPU Batch Size: 1024 documents
- DB Write Batch: 5000 documents
- Checkpoint Interval: 50,000 documents

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