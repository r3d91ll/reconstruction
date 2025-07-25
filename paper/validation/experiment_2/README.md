# Experiment 2: Chunk-Level Embeddings

## Purpose

Test whether chunk-level embeddings provide better fidelity or introduce noise compared to document-level embeddings from Experiment 1.

## Key Differences from Experiment 1

### Experiment 1 (Document-Level)
- **Unit**: Entire paper as single embedding
- **Embeddings**: 1 per paper (2048-dim)
- **Connections**: Paper-to-paper
- **Graph**: ~1000 nodes, ~500K edges

### Experiment 2 (Chunk-Level)
- **Unit**: Paper sections/chunks
- **Embeddings**: ~10-50 per paper (2048-dim each)
- **Connections**: Chunk-to-chunk across papers
- **Graph**: ~20K nodes, ~10M potential edges

## Chunking Strategy

### Option 1: Section-Based Chunking
- Introduction
- Related Work
- Methodology
- Results
- Conclusion
- Each figure/table as separate chunk

### Option 2: Semantic Chunking (using Docling)
- Use Docling's natural document segments
- Preserve figure-text relationships
- Keep tables with their context

### Option 3: Fixed-Size Chunking
- 1000-token chunks with 200-token overlap
- Simple but may break semantic units

## Comparison Metrics

### 1. Fidelity Improvements
- Can we find finer-grained theory-practice bridges?
- Do methodology chunks connect better to implementation chunks?
- Are figure-to-figure connections meaningful?

### 2. Noise Analysis
- How many chunk connections are spurious?
- Do boilerplate sections create false connections?
- Is there a "chunk size sweet spot"?

### 3. Computational Efficiency
- Memory usage (20x more embeddings)
- Query performance
- Storage requirements

## Implementation Plan

### Phase 1: Chunk Extraction
```python
# Extract chunks from existing Docling PDF content
python3 extract_chunks_from_pdfs.py --strategy sections --papers 1000
```

### Phase 2: Chunk Embeddings
```python
# Generate embeddings for each chunk
python3 generate_chunk_embeddings.py
```

### Phase 3: Graph Construction
```python
# Build chunk-to-chunk similarity graph
python3 build_chunk_graph.py --threshold 0.6  # Higher threshold for chunks
```

### Phase 4: Analysis
```python
# Compare with experiment_1 results
python3 compare_experiments.py
```

## Expected Outcomes

### If Chunks Add Fidelity:
- Specific methodology sections connect to implementations
- Figure captions link to related figures
- Code blocks cluster by algorithm type
- Theory sections connect to proof sections

### If Chunks Add Noise:
- References sections create false connections
- Boilerplate text dominates connections
- Loss of document-level context
- Computational overhead not justified

## Database Design

### Collections
- `paper_chunks`: Individual chunks with metadata
- `chunk_similarity`: Chunk-to-chunk edges
- `chunk_to_paper`: Mapping chunks to source papers

### Chunk Schema
```json
{
  "_key": "2301.12345_chunk_3",
  "paper_id": "papers/2301.12345",
  "chunk_index": 3,
  "chunk_type": "methodology",
  "title": "Section 3: Proposed Approach",
  "content": "...",
  "embeddings": [...],
  "metadata": {
    "start_page": 4,
    "end_page": 6,
    "has_figures": true,
    "has_code": true
  }
}
```

## Success Criteria

1. **Precision**: Find connections invisible at document level
2. **Recall**: Don't lose important document-level connections
3. **Interpretability**: Chunk connections are explainable
4. **Efficiency**: Query time remains practical

## Next Steps

1. Implement chunk extraction from Docling output
2. Generate chunk embeddings with position encoding
3. Build chunk graph with appropriate thresholds
4. Compare results with experiment_1
5. Analyze fidelity vs noise tradeoffs