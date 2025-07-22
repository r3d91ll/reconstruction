# Step-by-Step Proof Plan for Information Reconstructionism

## Goal: Prove ONE thing at a time with ArangoDB

### Step 1: Generate Embeddings (WHAT dimension) ✅

**Prove**: We can create semantic representations

```bash
# Install sentence-transformers
pip install sentence-transformers

# Generate embeddings for 1000 papers
python generate_embeddings_simple.py --limit 1000
```

**Output**: Papers with embeddings in JSON

---

### Step 2: Load into ArangoDB ✅

**Prove**: Basic graph structure works

```python
# Create database and collections
# Load papers as nodes
# No edges yet - just nodes
```

**Validation**: Can query papers by year, category

---

### Step 3: Compute Semantic Similarity ✅

**Prove**: Context scores are meaningful

```python
# For each paper pair:
# Context(i,j) = cosine_similarity(embedding_i, embedding_j)
# Store as edges with weight = Context
```

**Validation**: Similar papers have higher Context scores

---

### Step 4: Apply Context^1.5 Amplification ✅

**Prove**: Context^α amplification works as predicted

```python
# For each edge:
# amplified_weight = Context^1.5
# Update edge weights
```

**Validation**: Distribution changes as theory predicts

---

### Step 5: Multiply by Physical Grounding ✅

**Prove**: Full CONVEYANCE calculation

```python
# For each edge:
# conveyance = physical_grounding * Context^1.5
# Final edge weight = conveyance
```

**Validation**: High-grounding papers have stronger connections

---

### Step 6: Find ONE Gravity Well ✅

**Prove**: Semantic clustering emerges naturally

```aql
# Find paper with most high-weight connections
# This should be a foundational paper
```

**Validation**: It's actually an important paper (e.g., "Attention Is All You Need")

---

### Step 7: Trace ONE Concept Evolution ✅

**Prove**: Information flows through network over time

```aql
# Pick "attention mechanism"
# Find papers mentioning it
# Follow high-weight edges
# Show temporal progression
```

**Validation**: Concept evolves from 2014 → 2024

---

## Success Criteria

Each step must work before moving to next. No fancy visualizations yet. Just:

- Print statements showing numbers
- Simple CSV outputs
- Basic AQL queries returning results

## What This Proves

1. **Zero Propagation**: Missing embedding = no edges
2. **Multiplicative Model**: All dimensions required
3. **Context Amplification**: α=1.5 creates meaningful clusters
4. **Physical Grounding**: Distinguishes theory from practice
5. **Semantic Gravity**: Important papers naturally become hubs
6. **Information Flow**: Concepts evolve through network

## Next: Build the Code

One file per step. Keep it simple. Prove the math works.
