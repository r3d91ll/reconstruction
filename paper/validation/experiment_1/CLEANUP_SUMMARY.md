# Experiment 1 Cleanup Summary

## Actions to Take Before Pushing Upstream

### 1. Run Cleanup Script
```bash
cd /home/todd/reconstructionism/validation/experiment_1
./cleanup_for_repo.sh
```

This will:
- Archive 18 older/alternate versions to `archive/`
- Delete 11 one-time check and debug scripts
- Clean up __pycache__ directories
- Keep only essential pipeline and analysis scripts

### 2. Replace README
```bash
mv README_updated.md README.md
```

The updated README includes:
- Latest empirical results from 997-paper run
- Consolidated findings and validation
- Clean directory structure
- Comparison with Experiment 2

### 3. Files to Keep

#### Pipeline (7 scripts)
- `extract_pdfs_docling.py` - PDF extraction
- `step1_generate_embeddings_jina_full.py` - Embeddings with full content
- `step2_load_arangodb.py` - Database loading
- `step3_compute_similarity.py` - Similarity computation
- `step4_context_amplification_batch.py` - Batch amplification
- `run_pipeline_with_docling.py` - Main runner
- `arangodb_setup.py` - Database utilities
- `logging_utils.py` - Logging helpers

#### Analysis (3 scripts)
- `document_semantic_landscape.py`
- `empirical_alpha_measurement.py`
- `semantic_primitives_3d.py`

#### Wolfram (3 scripts)
- `extract_for_wolfram_validation.py`
- `wolfram_validation.py`
- `generate_wolfram_report.py`

#### Results to Include
- `results/docling_run_1000_20250724_003905/FINAL_REPORT.md`
- `results/analysis/amplification_distribution.png`

### 4. Key Results Summary

**Empirical Validation Complete**:
- 997 papers analyzed with full PDF content
- Multiplicative model confirmed (zero propagation)
- Context amplification α = 0.824 (lower than theoretical 1.5)
- Domain-specific α values measured
- Theory-practice bridges identified

### 5. Repository Structure After Cleanup

```
experiment_1/
├── README.md (updated with results)
├── pipeline/ (7 core scripts)
├── analysis/ (3 analysis tools)
├── wolfram/ (3 validation scripts)
├── results/
│   └── docling_run_1000_20250724_003905/
└── archive/ (18 archived scripts)
```

### 6. Git Commands
```bash
# After cleanup
git add -A
git commit -m "Clean experiment_1 for upstream: empirical validation complete with 997 papers"
git push
```

## Important Notes

1. The cleanup preserves all essential functionality
2. Archived scripts remain available in `archive/` if needed
3. Results demonstrate successful validation of core theory
4. Ready for comparison with experiment_2 chunk-level analysis