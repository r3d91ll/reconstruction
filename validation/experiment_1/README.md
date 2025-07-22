# Experiment 1: WHERE × WHAT × Context(top-k)

## Purpose

Build and validate a semantic similarity graph as the foundation for Information Reconstructionism theory.

## Directory Structure

```
experiment_1/
├── README.md              # This file
├── pipeline/              # Core pipeline scripts
│   ├── extract_pdfs_docling.py    # PDF content extraction with Docling
│   ├── step1_*.py                 # Embedding generation
│   ├── step2_*.py                 # ArangoDB loading
│   ├── step3_*.py                 # Similarity computation
│   ├── step4_*.py                 # Context amplification
│   ├── run_pipeline.py            # Main pipeline runner
│   ├── generate_graph_report.py   # Final analysis report
│   └── check_*.py                 # Utility scripts
├── wolfram/               # Wolfram validation scripts
│   ├── *.wl              # Wolfram Language scripts
│   └── *_validation.py   # Python validation
├── data/                  # Generated data
│   ├── graph_data.json
│   └── graph_data.wl
├── docs/                  # Documentation
│   ├── PDF_EXTRACTION_REQUIREMENTS.md
│   └── *.md              # Reports and specs
├── results/               # Pipeline run results
│   └── run_N_timestamp/  # Each run's output
└── archive/               # Old/experimental scripts
```

## Theory Being Tested

```
Information(i→j|S-O) = WHERE × WHAT × CONVEYANCE × TIME × FRAME
```

This experiment focuses on the **WHERE × WHAT × Context** slice, where:
- **WHERE**: Paper location (arxiv_id, categories) - stored as metadata
- **WHAT**: Semantic content (ALL 2048 dimensions from Jina embeddings)
- **Context**: Semantic similarity (cosine similarity > 0.5)

### Important: Dimensional Allocation

**Current Implementation (Experiment 1):**
- Using **ALL 2048 dimensions** for semantic content (WHAT)
- WHERE, TIME stored as metadata fields (not embedded)
- CONVEYANCE not yet implemented
- FRAME implicit (single observer perspective)

**Future Implementation (Full Theory):**
- WHERE: 64 dimensions
- WHAT: 1024 dimensions  
- CONVEYANCE: 936 dimensions
- TIME: 24 dimensions
- Total: 2048 dimensions

This experiment validates pure semantic similarity first before introducing dimensional allocation.

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

## Full Pipeline Process (With PDF Content)

### Prerequisites
1. Papers with JSON metadata in `/home/todd/olympus/Erebus/unstructured/papers/`
2. Corresponding PDF files in the same directory
3. Docling installed: `pip install docling`

### Step 1: Extract Full PDF Content
```bash
cd pipeline
python3 extract_pdfs_docling.py --papers-dir /home/todd/olympus/Erebus/unstructured/papers --limit 4000
```
- Extracts full PDF content using IBM's Docling
- Preserves document structure (sections, figures, tables, equations)
- Adds `pdf_content` field to each JSON
- Generates embeddings from complete papers (not just abstracts)
- **Processing time**: ~30-60 seconds per paper

### Step 2: Run Complete Pipeline
```bash
python3 run_pipeline.py 4000
```
This executes:
1. **step1_generate_embeddings_jina.py**: Uses full PDF content for embeddings
2. **step2_load_arangodb.py**: Loads papers as nodes
3. **step3_compute_similarity.py**: GPU-accelerated similarity computation
4. **step4_context_amplification_batch.py**: Applies Context^1.5

### Step 3: Generate Analysis Report
```bash
python3 generate_graph_report.py
```
- Analyzes graph structure
- Identifies milestone papers
- Finds theory-practice bridges
- Creates visualizations

## Usage (Quick Test)

```bash
# Set environment variables
export ARANGO_HOST="http://192.168.1.69:8529"
export ARANGO_USERNAME="root"
export ARANGO_PASSWORD="your_password"

# Quick test with abstracts only
python3 test_pipeline_10.py

# Check year distribution
python3 check_year_distribution.py

# Verify results
python3 verify_graph.py
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

## Abstract vs Full Content Comparison

### Abstract-Only Embeddings
- **Content**: ~1KB per paper (title + abstract + categories)
- **Processing**: ~0.1 seconds per paper
- **Use case**: Quick semantic overview
- **Limitations**: Misses implementation details

### Full PDF Embeddings (with Docling)
- **Content**: ~50-100KB per paper (complete text + figures + tables)
- **Processing**: ~30-60 seconds per paper
- **Use case**: Deep theory-practice bridge discovery
- **Advantages**: 
  - Captures implementation details
  - Includes code examples
  - Preserves figure context
  - Enables true conveyance measurement

## What We Need Next

**Conveyance Mapping**: WHERE × WHAT × Conveyance
- Measure transformation potential
- Track implementation success
- Analyze practical utility

## Key Insight

If Context^α hypothesis is correct, papers with high semantic similarity (Context) should also show high transformation potential (Conveyance) when properly amplified.