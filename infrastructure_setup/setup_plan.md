# Infrastructure Setup Plan

## Overview

This setup creates a persistent infrastructure that all experiments can use without rebuilding. The key insight is to do ALL heavy computation upfront, including ISNE model training, so experiments can focus purely on hypothesis testing.

## Phase 0: Complete Infrastructure Initialization

### 0.1 Database Schema Creation (30 minutes)
```bash
python setup_database_schema.py \
    --db-name information_reconstructionism_base \
    --clean-start
```

**Creates:**
- Database: `information_reconstructionism_base`
- Collections:
  - `papers`: 2000 document metadata
  - `paper_embeddings`: 2000 × 2048 dimensions
  - `chunks`: ~100,000 semantic units  
  - `chunk_embeddings`: ~100,000 × 2048 dimensions
  - `paper_similarities`: ~2M edges
  - `chunk_similarities`: ~10M edges
  - `implementations`: GitHub/citation links
  - `citation_network`: Paper-to-paper citations

### 0.2 Document Processing (6 hours)
```bash
python process_documents.py \
    --input-dir /mnt/data/arxiv_data/pdf \
    --num-documents 2000 \
    --chunk-size 512 \
    --chunking-strategy semantic
```

**Output:**
- Extracted text from 2000 PDFs
- ~50 chunks per document (semantic boundaries)
- Total: ~100,000 chunks
- Stored in database with metadata

### 0.3 Embedding Generation (9 hours)
```bash
python generate_all_embeddings.py \
    --use-gpu \
    --batch-size 32 \
    --model jina-v4
```

**Generates:**
- Document embeddings: 2000 × 2048
- Chunk embeddings: 100,000 × 2048
- Total: 102,000 embeddings
- GPU utilization: 2 × A6000

### 0.4 Similarity Matrix Computation (3 hours)
```bash
python compute_similarity_matrices.py \
    --document-threshold 0.7 \
    --chunk-threshold 0.8 \
    --output-format sparse
```

**Computes:**
- Document-level: ~2M edges (threshold > 0.7)
- Chunk-level: ~10M edges (threshold > 0.8)
- Stores as sparse matrices in database

### 0.5 ISNE Model Training (4 hours)
```bash
python train_isne_model.py \
    --eigenvectors 100 \
    --output-path /models/isne_base.pkl
```

**Trains:**
- ISNE on complete similarity graph
- k=100 eigenvectors for spectral embedding
- Serialized model for instant loading
- Supports incremental updates

### 0.6 ISNE Validation (1 hour)
```bash
python validate_isne_incremental.py \
    --test-papers 200 \
    --max-update-time 300
```

**Validates:**
- Incremental addition of new papers
- Update time < 5 minutes per paper
- Graph integrity maintained
- Spectral gap preserved

## Total Setup Time: 23.5 hours

## Post-Setup Assets

### Persistent Database
- Complete document and chunk data
- All embeddings pre-computed
- Similarity matrices cached
- Ready for any experiment

### Trained Models
- `/models/isne_base.pkl` - ISNE model
- `/models/jina_cache.pkl` - Embedding cache
- `/models/similarity_index.pkl` - Fast lookup

### Infrastructure State
```json
{
  "database": "information_reconstructionism_base",
  "documents": 2000,
  "chunks": 100000,
  "embeddings": 102000,
  "similarities": {
    "document_edges": 2000000,
    "chunk_edges": 10000000
  },
  "models": {
    "isne": "trained",
    "incremental_capable": true
  }
}
```

## Benefits

1. **No Infrastructure in Experiments**: Experiments become pure hypothesis tests
2. **Instant Startup**: Everything pre-computed, experiments run immediately
3. **Reproducible**: Same base data for all experiments
4. **Scalable**: ISNE allows incremental additions without full rebuild
5. **Version Controlled**: Database snapshots for exact reproduction

## Usage in Experiments

```python
# Experiment code becomes trivial
from irec_infrastructure import ExperimentBase

class Experiment1(ExperimentBase):
    def run(self):
        # Everything is already computed!
        papers = self.db.get_papers()
        similarities = self.db.get_similarities()
        
        # Focus purely on hypothesis testing
        results = self.test_multiplicative_model(papers, similarities)
```

## Maintenance

- Weekly incremental updates with new papers
- Monthly full validation
- Quarterly model retraining if needed