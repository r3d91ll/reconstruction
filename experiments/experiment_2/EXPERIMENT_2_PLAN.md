# Experiment 2: Late Chunking Analysis Plan

**Target**: 2000 documents  
**Key Innovation**: Semantic late chunking with Jina V4  
**Database**: `information_reconstructionism_exp2` (separate from exp1)

## Objectives

1. **Address Experiment 1 Limitations**:
   - Restore granularity lost with whole-document embeddings
   - Preserve semantic boundaries within documents
   - Enable chunk-level similarity analysis

2. **New Capabilities**:
   - Multi-scale information representation (chunk → document → corpus)
   - Intra-document semantic structure analysis
   - Fine-grained theory-practice bridge discovery

3. **Empirical α Measurement**:
   - Compare chunk-level similarities to document-level
   - Attempt to measure "natural" context amplification
   - Test if different chunk types need different α values

## Technical Approach

### 1. Late Chunking Configuration
- **Method**: Jina V4 with `late_chunking=True`
- **Chunk size**: ~512 tokens (semantic boundaries)
- **Overlap**: Minimal, respecting semantic units
- **Expected chunks per doc**: 50-200 (depending on length)

### 2. Database Design
```
Database: information_reconstructionism_exp2
Collections:
- papers_exp2: Document metadata
- chunks_exp2: Individual chunks with embeddings
- chunk_similarity_exp2: Chunk-to-chunk edges
- doc_similarity_exp2: Aggregated document similarities
- chunk_hierarchy_exp2: Chunk-to-document relationships
```

### 3. Pipeline Steps

#### Step 1: Extract with Late Chunking
- Use existing `generate_late_chunked_embeddings.py`
- Process 2000 documents from papers directory
- Store both chunk and document embeddings

#### Step 2: Load Hierarchical Data
- Load papers → chunks → embeddings
- Maintain chunk-document relationships
- Calculate chunk statistics per document

#### Step 3: Multi-Scale Similarity
- Chunk-to-chunk similarities (fine-grained)
- Document-to-document (aggregated from chunks)
- Cross-scale analysis

#### Step 4: Context Amplification Analysis
- Apply different α values to different similarity scales
- Measure clustering at chunk vs document level
- Identify optimal α for different content types

## Expected Outcomes

1. **Granularity**: 100K-400K chunks from 2000 documents
2. **Similarity edges**: ~10M chunk-level connections
3. **Performance**: GPU acceleration critical for scale
4. **Insights**: 
   - Chunk-level semantic patterns
   - Document structure understanding
   - Multi-scale information dynamics

## Success Criteria

1. Successfully process 2000 documents with late chunking
2. Demonstrate improved semantic granularity vs experiment_1
3. Identify natural α values from chunk-document relationships
4. Show theory-practice bridges at chunk level
5. Maintain separate database for comparison

## Timeline

1. **Setup** (30 min): Database creation, pipeline configuration
2. **Processing** (4-6 hours): Late chunking extraction for 2000 docs
3. **Computation** (1-2 hours): Similarity calculations with GPU
4. **Analysis** (1 hour): Context amplification and validation
5. **Report** (30 min): Generate findings and comparisons

---

Ready to begin experiment_2 with improved semantic analysis through late chunking.