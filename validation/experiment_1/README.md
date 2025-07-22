# Experiment 1: WHERE × WHAT × Context(top-k)

## Purpose

Build and validate a semantic similarity graph as the foundation for Information Reconstructionism theory.

## Directory Structure

```
experiment_1/
├── README.md              # This file
├── pipeline/              # Core pipeline scripts
│   ├── step1_*.py        # Embedding generation
│   ├── step2_*.py        # ArangoDB loading
│   ├── step3_*.py        # Similarity computation
│   ├── step4_*.py        # Context amplification
│   ├── test_pipeline_10.py
│   └── run_batch.py
├── wolfram/               # Wolfram validation scripts
│   ├── *.wl              # Wolfram Language scripts
│   └── *_validation.py   # Python validation
├── data/                  # Generated data
│   ├── graph_data.json
│   └── graph_data.wl
├── docs/                  # Documentation
│   └── *.md              # Reports and specs
└── archive/               # Old/experimental scripts
```

## Theory Being Tested

```
Information(i→j|S-O) = WHERE × WHAT × CONVEYANCE × TIME × FRAME
```

This experiment focuses on the **WHERE × WHAT × Context** slice, where:
- **WHERE**: Paper location (arxiv_id, categories)
- **WHAT**: Semantic content (2048-dim Jina embeddings)
- **Context**: Semantic similarity (cosine similarity > 0.5)

## Pipeline Components (in order)

### Phase 1: Embedding Generation
- `step1_generate_embeddings_jina.py` - Generate Jina V4 embeddings (WHERE × WHAT)
- `step1_test_10.py` - Test version for 10 papers

### Phase 2: Graph Construction  
- `step2_load_arangodb.py` - Load papers as nodes into ArangoDB
- `step3_compute_similarity.py` - Compute semantic similarity (Context)
- `step4_context_amplification.py` - Apply Context^1.5 amplification

### Phase 3: Validation & Analysis
- `export_for_wolfram.py` - Export graph data for analysis
- `wolfram_validation.py` - Mathematical validation of principles
- `wolfram_api_validation.py` - Wolfram Alpha API validation
- `generate_wolfram_report.py` - Generate validation reports

### Wolfram Scripts
- `run_validation.wl` - Core validation tests
- `context_amplification_test.wl` - Context^α analysis
- `zero_propagation_test.wl` - Zero propagation proof
- `validate_with_real_data.wl` - Real data validation

### Utilities
- `test_pipeline_10.py` - Test entire pipeline with 10 papers
- `run_batch.py` - Run pipeline on N papers
- `verify_graph.py` - Verify graph structure and queries

## Usage

```bash
# Set environment variables
export ARANGO_HOST="http://192.168.1.69:8529"
export ARANGO_USERNAME="root"
export ARANGO_PASSWORD="your_password"
export WOLFRAM_APP_ID="your_app_id"

# Test pipeline
python test_pipeline_10.py

# Run on 1000 papers
python run_batch.py 1000

# Verify results
python verify_graph.py
```

## What This Pipeline Measures

**Semantic Context Mapping**: WHERE × WHAT × Context(top-k)
- WHERE: Paper location (arxiv_id, categories)  
- WHAT: Semantic content (2048-dim embeddings)
- Context: Similarity score (filtered > 0.5, amplified by ^1.5)

## Key Findings

1. **Zero Propagation**: Validated ✓
2. **Context Amplification**: Context^1.5 creates clustering ✓
3. **Graph Structure**: Full connectivity at threshold 0.5 ✓

## What We Need Next

**Conveyance Mapping**: WHERE × WHAT × Conveyance
- Measure transformation potential
- Track implementation success
- Analyze practical utility

## Key Insight

If Context^α hypothesis is correct, papers with high semantic similarity (Context) should also show high transformation potential (Conveyance) when properly amplified.