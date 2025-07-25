# Information Reconstructionism Infrastructure Setup

This directory contains the production pipeline for processing arXiv documents with TRUE late chunking.

## Production Files

### Core Pipeline
- **`process_final_1960_documents.py`** - Main production script that processes all 1960 documents
- **`process_documents_local_gpu_with_metadata.py`** - Document processor with metadata integration
- **`setup_irec_database_schema.py`** - Creates the ArangoDB schema for Information Reconstructionism

### Documentation
- **`TRUE_LATE_CHUNKING_SETUP.md`** - Explains the TRUE late chunking approach
- **`setup_plan.md`** - Infrastructure setup plan
- **`requirements.txt`** - Python dependencies

### Logs
- **`process_1960_documents.log`** - Processing log (created during run)

## Usage

1. **Setup Database Schema** (one time):
```bash
python3 setup_irec_database_schema.py --db-name irec_production
```

2. **Process All Documents**:
```bash
python3 process_final_1960_documents.py
```

## Environment Variables

The pipeline uses these environment variables (from .bashrc):
- `ARANGO_USERNAME` - ArangoDB username
- `ARANGO_PASSWORD` - ArangoDB password

## TRUE Late Chunking

This pipeline implements TRUE late chunking:
1. **Docling** extracts FULL documents (NO chunking)
2. **Local Jina GPU** creates semantic chunks from full documents
3. **Metadata** from arXiv is fully integrated
4. **ArangoDB** stores everything with proper indexes

## Expected Results

- **Documents**: 1,960
- **Chunks**: ~100,000 (50-100 per document)
- **Processing Time**: 6-8 hours
- **GPU Usage**: Dual A6000s with NVLink

## Archive

Test scripts and old implementations have been moved to `archive_tests/` for reference.