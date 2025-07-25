# Experiments

This directory contains experiments that bridge the theoretical framework and infrastructure implementation to validate Information Reconstructionism claims.

## Why Shared Experiments?

These experiments:
- Use the infrastructure to process real data
- Validate theoretical predictions
- Will move with infrastructure when repositories split
- Provide reproducible validation results

## Available Experiments

### Experiment 1: Multiplicative Model Validation
Tests the core claim that information requires ALL dimensions > 0.

```bash
cd experiment_1/
python run_experiment.py
```

Key validations:
- Zero propagation (any dimension = 0 → information = 0)
- Context amplification measurement
- WHERE × WHAT × CONVEYANCE interactions

### Experiment 2: Large-Scale GPU Processing
Validates the theory at scale using dual A6000 GPUs.

```bash
cd experiment_2/
./pipeline/launch_gpu_pipeline.sh
```

Key validations:
- Processing 10,000+ documents
- Multi-scale similarity analysis
- Theory-practice bridge discovery

## Adding New Experiments

New experiments should:
1. Import from `infrastructure.irec_infrastructure`
2. Test specific theoretical claims
3. Use pre-computed embeddings when possible
4. Generate reproducible results

## Results

Experiment results are stored in:
- `experiment_*/results/` - Raw data
- `experiment_*/analysis/` - Processed findings
- `experiment_*/visualizations/` - Graphs and plots

## Dependencies

All experiments depend on:
- Infrastructure being properly set up
- Database populated with documents
- GPU access for embedding generation

See `infrastructure/INFRA_CLAUDE.md` for setup instructions.