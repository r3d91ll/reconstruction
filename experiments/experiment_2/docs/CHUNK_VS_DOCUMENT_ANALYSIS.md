# Chunk vs Document Level Analysis

## Experiment Design

### Research Questions

1. **Granularity vs Noise Tradeoff**
   - Do chunk-level connections reveal hidden patterns?
   - Or do they introduce spurious connections?

2. **Semantic Precision**
   - Can we connect specific methodologies to implementations?
   - Do figure-to-figure connections make sense?
   - Are code chunks semantically meaningful?

3. **Theory-Practice Bridge Resolution**
   - Document level: "Paper A relates to Paper B"
   - Chunk level: "Method in A.3 implements theory from B.2"

## Comparison Framework

### 1. Connection Quality Metrics

```python
# Document-level connection
Doc_A <--0.75--> Doc_B

# Chunk-level connections  
Doc_A.intro <--0.82--> Doc_B.intro      # Higher similarity
Doc_A.method <--0.91--> Doc_B.results   # Specific bridge
Doc_A.fig2 <--0.88--> Doc_B.fig5       # Visual similarity
Doc_A.code <--0.95--> Doc_B.algorithm  # Implementation match
```

### 2. Signal vs Noise Analysis

**Signal Indicators:**
- High-similarity chunk pairs have semantic coherence
- Chunk types align (method→method, figure→figure)
- Connections form meaningful patterns

**Noise Indicators:**
- References sections dominate connections
- Boilerplate text creates false matches
- Random distribution of chunk types in connections

### 3. Computational Comparison

| Metric | Document-Level | Chunk-Level |
|--------|---------------|-------------|
| Nodes | 1,000 | ~20,000 |
| Edges | ~500K | ~10M |
| Embedding Storage | 8GB | 160GB |
| Query Time | <1s | ~10s |
| Semantic Resolution | Paper | Section/Figure/Code |

## Implementation Strategy

### Phase 1: Chunk Extraction (No GPU needed)
- Use existing Docling PDF content
- Extract semantic sections
- Separate figures, tables, code blocks
- ~1-2 hours for 1000 papers

### Phase 2: Chunk Embeddings (GPU intensive)
- Generate embeddings per chunk
- Add positional encoding (chunk index)
- Preserve paper-chunk relationships
- ~8-10 hours with dual GPUs

### Phase 3: Graph Construction (GPU for similarity)
- Higher threshold (0.6-0.7) for chunks
- Filter by chunk type compatibility
- Build hierarchical graph structure
- ~2-3 hours

### Phase 4: Comparative Analysis
- Map document connections to chunk connections
- Identify new connections only visible at chunk level
- Measure noise ratio
- Generate comparison visualizations

## Expected Patterns

### If Chunking Adds Value:

1. **Method-Implementation Bridges**
   - "Attention mechanism" (paper A methodology)
   - Connects to "class MultiHeadAttention" (paper B code)

2. **Figure Evolution Tracking**
   - Architecture diagrams across papers
   - Progressive refinement visible

3. **Theory-Proof Connections**
   - Theorem statement chunks
   - Connect to proof chunks in other papers

### If Chunking Adds Noise:

1. **Reference Pollution**
   - All papers citing similar work falsely connected
   - References dominate connection graph

2. **Boilerplate Clusters**
   - "Related work" sections all similar
   - Acknowledgments create false connections

3. **Context Loss**
   - Important document-level relationships lost
   - Chunks lack sufficient context

## Visualization Plans

1. **Dual-Level Graph**
   - Document nodes (large)
   - Chunk nodes (small, colored by type)
   - Show both levels simultaneously

2. **Connection Heatmap**
   - X-axis: Chunk types in source
   - Y-axis: Chunk types in target
   - Color: Connection frequency

3. **Precision-Recall Curves**
   - Compare document vs chunk retrieval
   - Measure at different thresholds

## Success Metrics

1. **Precision Gain**: Find 20%+ more theory-practice bridges
2. **Noise Tolerance**: <30% connections are spurious
3. **Query Performance**: <10 second response time
4. **Interpretability**: Chunk connections explainable

## Risk Mitigation

1. **Memory Management**
   - Stream processing for embeddings
   - Batch similarity computation
   - Use sparse matrices where possible

2. **Quality Control**
   - Filter short chunks (<100 tokens)
   - Remove boilerplate sections
   - Validate chunk boundaries

3. **Computational Efficiency**
   - Use both GPUs for embedding generation
   - Optimize similarity computation
   - Consider approximate nearest neighbors

## Timeline

- Hour 0-2: Chunk extraction from existing PDFs
- Hour 2-10: Generate chunk embeddings (parallel GPUs)
- Hour 10-13: Build chunk similarity graph
- Hour 13-16: Comparative analysis
- Hour 16-20: Generate visualizations and report

Total: ~20 hours (vs 66 hours for full PDF extraction)