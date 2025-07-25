# FRAME Discovery Method: From Citation Networks to Compatibility Functions

## Overview

FRAME emerges as a directional compatibility function between information sources and receivers, discovered empirically through citation patterns and semantic chunk propagation.

## Mathematical Definition

```
FRAME(i→j) = f(semantic_overlap, temporal_distance, citation_strength, chunk_propagation)
```

Where:
- `i` is the source (cited paper)
- `j` is the receiver (citing paper)
- FRAME(i→j) ≠ FRAME(j→i) (directional/asymmetric)

## Discovery Algorithm

### Step 1: Build Citation Network
```python
# From arXiv metadata
citation_graph = {
    paper_id: {
        'cites': [cited_paper_ids],
        'cited_by': [citing_paper_ids],
        'chunks': [semantic_chunks],
        'timestamp': publication_date
    }
}
```

### Step 2: Analyze Chunk Propagation
```python
def analyze_propagation(source_paper, target_paper):
    source_chunks = source_paper['chunks']
    target_chunks = target_paper['chunks']
    
    # Which chunks appear (transformed) in target?
    propagated = find_semantic_matches(source_chunks, target_chunks)
    
    # How much did they transform?
    transformation_score = measure_semantic_drift(propagated)
    
    return {
        'propagation_rate': len(propagated) / len(source_chunks),
        'transformation_score': transformation_score
    }
```

### Step 3: Measure Citation Strength
```python
def citation_strength(source, target):
    # Direct citation
    direct = 1 if source in target['cites'] else 0
    
    # Co-citation (cited together by others)
    co_cited = len(set(source['cited_by']) & set(target['cited_by']))
    
    # Citation proximity (how central to target's work)
    proximity = citation_proximity_in_text(source, target)
    
    return weighted_sum(direct, co_cited, proximity)
```

### Step 4: Compute Directional FRAME
```python
def compute_frame(i, j):
    # Temporal component (can only flow forward in time)
    if i['timestamp'] > j['timestamp']:
        return 0  # No backward information flow
    
    # Semantic compatibility
    semantic = semantic_overlap(i['chunks'], j['chunks'])
    
    # Propagation success
    propagation = analyze_propagation(i, j)
    
    # Citation relationship
    citation = citation_strength(i, j)
    
    # Asymmetric weighting
    FRAME_ij = (
        semantic * α +
        propagation['rate'] * β +
        citation * γ +
        temporal_factor(i, j) * δ
    )
    
    return normalize(FRAME_ij)
```

## Validation Metrics

### 1. Prediction Accuracy
- Can FRAME predict which papers will cite each other?
- Does high FRAME(i→j) correlate with actual citations?

### 2. Information Flow
- Do chunks propagate along high-FRAME paths?
- Is transformation inversely related to FRAME?

### 3. Bridge Detection
- Papers with high bidirectional FRAME should be influential
- These should correspond to known seminal works

## Implementation with arXiv Data

### Phase 1: Citation Extraction
```python
# Extract from arXiv metadata
citations = extract_citations_from_references(pdf_text)
# Build bidirectional graph
build_citation_network(citations)
```

### Phase 2: Chunk Analysis
```python
# Using late chunking results from experiment_2
chunks = load_experiment2_chunks()
# Compute semantic embeddings
chunk_embeddings = compute_embeddings(chunks)
# Find propagation patterns
propagation_patterns = trace_chunk_evolution(chunk_embeddings, citation_graph)
```

### Phase 3: FRAME Computation
```python
# For all paper pairs with citations
for source, target in citation_pairs:
    frame_forward = compute_frame(source, target)
    frame_backward = compute_frame(target, source)
    
    # Store asymmetric relationship
    store_frame_edge(source, target, frame_forward, frame_backward)
```

## Expected Discoveries

### 1. Natural Information Gradients
- Survey papers: High outgoing FRAME, low incoming
- Technical breakthroughs: High incoming FRAME from future
- Tutorial papers: Bidirectional high FRAME

### 2. Temporal Patterns
- FRAME decay over time (concepts become less accessible)
- Sudden FRAME increases (paradigm shifts making old work relevant)

### 3. Domain Boundaries
- Low FRAME between unrelated fields
- "Bridge" papers with cross-domain high FRAME

## Integration with Core Theory

```
Information(i→j|S-O) = WHERE × WHAT × CONVEYANCE × TIME × FRAME(i→j)
```

FRAME(i→j) acts as the directional gate controlling information flow from source i to receiver j.

## Next Steps

1. **Implement citation extraction** from arXiv references
2. **Build citation network** from full arXiv corpus
3. **Compute FRAME values** for citation pairs
4. **Validate predictions** against actual citation patterns
5. **Discover emergent patterns** in FRAME landscape

This empirical approach transforms FRAME from a philosophical concept to a measurable, predictive component of information transfer.