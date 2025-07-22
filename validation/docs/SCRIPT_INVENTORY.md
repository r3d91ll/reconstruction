# Validation Script Inventory

## Core Proof Sequence (`/validation/core/`)

These scripts run in order to prove the framework:

1. **`step1_generate_embeddings_jina.py`**
   - Generates 1024-dimensional embeddings using local Jina V4 model
   - Supports up to 16K context window
   - Outputs: Papers with WHAT dimension embeddings

2. **`step2_load_arangodb.py`**
   - Loads embedded papers into ArangoDB as nodes
   - Creates `papers` collection
   - No edges yet - just validates graph structure

3. **`step3_compute_similarity.py`**
   - Computes pairwise semantic similarity (Context scores)
   - Creates `semantic_similarity` edges
   - Filters by threshold (default 0.5)

4. **`step4_context_amplification.py`**
   - Applies Context^1.5 amplification to edge weights
   - Demonstrates clustering effect
   - Validates α=1.5 theoretical prediction

5. **`run_proof_sequence.py`**
   - Runs steps 1-4 in sequence
   - Checks prerequisites
   - Stops on any failure

## Advanced Options (`/validation/core/`)

- **`step1_jina_advanced.py`**
  - Advanced Jina V4 with late chunking
  - PDF text extraction support
  - Multimodal ready (for future image support)

## Analysis Scripts (`/validation/analysis/`)

- **`compute_physical_grounding.py`**
  - Computes Physical Grounding Factor from text markers
  - Validates theory vs practice distinction

- **`arangodb_setup.py`**
  - Advanced ArangoDB queries
  - Gravity well detection
  - Concept evolution tracing

## Master Runner (`/validation/`)

- **`run_validation.py`**
  - Creates timestamped output directories
  - Runs full validation sequence
  - Saves logs and results
  - Creates 'latest' symlink

## Output Structure

Each run creates:
```
/validation/results/run_YYYYMMDD_HHMMSS/
├── data/                  # Embedded papers, computed scores
├── logs/                  # Execution logs
├── analysis/              # Analysis results
└── run_info.txt          # Run metadata
```

## To Execute

```bash
cd /home/todd/reconstructionism/validation
python run_validation.py
```

This will:
1. Create timestamped directory
2. Run all validation steps
3. Save results and logs
4. Update 'latest' symlink

## Next Steps After Validation

Once core validation passes, add:
- Step 5: Multiply by Physical Grounding Factor
- Step 6: Find gravity wells
- Step 7: Trace concept evolution
- Population genetics analysis