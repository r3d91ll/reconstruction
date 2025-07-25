# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains the theoretical framework and mathematical validation for Information Reconstructionism - a new theory of observer-dependent information existence.

## Project Goal

**Prove that information exists as a multiplicative function of dimensional prerequisites, manifesting only through observer interaction.**

## Core Equation

```
Information(i→j|S-O) = WHERE × WHAT × CONVEYANCE × TIME × FRAME
```

If ANY dimension = 0, then Information = 0

## Repository Structure

```
reconstructionism/
├── theory/                      # Core theoretical papers
│   ├── main_theory.md          # Primary theoretical framework
│   ├── mathematical_proofs.md   # Formal mathematical validation
│   └── empirical_validation.md  # Evidence and test results
├── validation/                  # Proof-of-concept code
│   ├── wolfram/                # Mathematica validation scripts
│   ├── python/                 # Python demonstrations
│   └── data/                   # Test datasets
├── presentations/              # Academic presentation materials
│   ├── slides/                 # Conference presentations
│   └── abstracts/              # Paper abstracts
└── evidence/                   # Empirical evidence
    ├── zero_propagation/       # Tests showing dimension = 0 → info = 0
    ├── context_amplification/  # Measuring Context^α effects
    └── bridge_discovery/       # Theory-practice connections
```

## Key Claims to Prove

### 1. Multiplicative Model
- Information requires ALL dimensions > 0
- Not additive: you can't compensate for missing dimensions
- Empirically testable through zero propagation

### 2. Context Amplification
- Context acts as exponential amplifier: Context^α
- Different domains have different α values (1.5-2.0)
- Measurable through conveyance analysis

### 3. Observer Dependency
- Different observers create different information realities
- FRAME function determines what information exists
- A-Observer vs S-Observer perspectives

### 4. Theory-Practice Bridges
- High-conveyance connections between abstract and concrete
- Discoverable through dimensional analysis
- Key to knowledge transformation

## Validation Strategy

### Mathematical Validation
1. Prove multiplicative model mathematically
2. Validate dimensional bounds (Johnson-Lindenstrauss)
3. Show convergence properties
4. Demonstrate stability of Context^α

### Empirical Validation
1. Test zero propagation (any dimension = 0)
2. Measure context amplification values
3. Find real theory-practice bridges
4. Compare observer perspectives

### Computational Validation
1. Implement minimal proof-of-concept
2. Process test corpus (100-1000 documents)
3. Visualize dimensional relationships
4. Demonstrate bridge discovery

## Success Criteria

### For Academic Presentation
- [ ] Mathematical proofs complete and verified
- [ ] Empirical evidence from test corpus
- [ ] Working demonstration (minimal viable)
- [ ] Clear differentiation from existing theories

### For Publication
- [ ] Formal mathematical framework
- [ ] Reproducible experiments
- [ ] Statistical validation
- [ ] Peer review responses ready

## Current Status

### Theory: [In Progress]
- Core framework documented
- Mathematical foundations laid
- Critical responses addressed

### Validation: [Pending]
- Wolfram validation script ready
- Python implementation needed
- Test corpus selection required

### Evidence: [Partial]
- Zero propagation conceptually proven
- Context amplification theoretical
- Bridge discovery methodology defined

## Next Steps

1. **Immediate**: Create minimal Python validation showing zero propagation
2. **Short-term**: Gather 100-document test corpus for empirical validation
3. **Medium-term**: Build interactive visualization of theory-practice bridges
4. **Long-term**: Full implementation proving 10M document scalability

## Key Differentiators

### From Traditional Information Theory
- Observer-dependent (not universal)
- Multiplicative (not additive)
- Dimensional (not scalar)

### From Semantic Similarity
- Includes actionability (CONVEYANCE)
- Context amplifies exponentially
- Bridges have special properties

### From Standard RAG/Embeddings
- Theory-grounded dimensions
- Observer-relative retrieval
- Bridge-based navigation

## Remember

This repository focuses ONLY on proving the theoretical framework. Implementation details, infrastructure, and applications come later. The goal is academic validation and publication of the core theory.

**Focus areas**:
1. Mathematical rigor
2. Empirical validation
3. Clear presentation
4. Reproducible results

**Avoid**:
1. Implementation complexity
2. Infrastructure details
3. Mythological allegories
4. Scope creep

The theory stands on its own mathematical and empirical merits.

## Common Development Tasks

### Setting Up the Environment

```bash
# Create and activate virtual environment
./setup_venv.sh
source venv/bin/activate
```

### Installing Dependencies

```bash
# Core infrastructure dependencies
pip install -r infrastructure_setup/requirements.txt

# Install irec_infrastructure package in development mode
pip install -e .

# For development with testing/linting
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests with pytest
pytest

# Run tests with coverage
pytest --cov=irec_infrastructure

# Test infrastructure components
python test_infrastructure.py
```

### Code Quality Checks

```bash
# Format code with black
black irec_infrastructure/ validation/

# Type checking with mypy
mypy irec_infrastructure/
```

### Running Experiments

```bash
# Experiment 1: Multiplicative Model Validation
cd validation/experiment_1
python run_experiment.py

# Experiment 2: GPU-accelerated processing
cd validation/experiment_2
./pipeline/launch_gpu_pipeline.sh
```

### Processing Documents

```bash
# Process arXiv documents with GPU acceleration
python infrastructure_setup/process_documents_local_gpu_with_metadata.py \
    --input-dir /mnt/data/arxiv_data/pdf \
    --output-dir ./processed_documents_local \
    --num-docs 1960

# Process final 1960 documents
python infrastructure_setup/process_final_1960_documents.py
```

## High-Level Architecture

### Core Infrastructure (`irec_infrastructure/`)

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

### Validation Framework (`validation/`)

Experiments are structured to test specific hypotheses:

- **Experiment 1**: Tests multiplicative model and zero propagation
- **Experiment 2**: Large-scale GPU processing and multi-scale analysis
- Each experiment inherits from `ExperimentBase` for consistent infrastructure access

### Key Design Principles

1. **Pre-computed Infrastructure**: All heavy computation (embeddings, similarities) is done once and stored
2. **Experiment Isolation**: Each experiment focuses purely on hypothesis testing
3. **GPU Optimization**: Batched processing with memory management for large-scale data
4. **Graph-Based Storage**: ArangoDB enables efficient similarity queries and graph traversal

## Database Schema

- `papers`: Document metadata and full text
- `paper_embeddings`: Document-level embeddings (2048D)
- `chunks`: Semantic text chunks with metadata  
- `chunk_embeddings`: Chunk-level embeddings (2048D)
- `paper_similarities`: Document similarity edges
- `chunk_similarities`: Chunk similarity edges
- `implementations`: GitHub/implementation links
- `citation_network`: Academic citation graph

## Environment Variables

Create a `.env` file with:

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