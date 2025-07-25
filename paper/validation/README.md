# Information Reconstructionism Validation

## Overview

This directory contains the validation framework and experiments for Information Reconstructionism theory.

## Directory Structure

```
validation/
├── README.md              # This file
├── experiment_1/          # WHERE × WHAT × Context mapping
│   ├── pipeline/         # Core processing scripts
│   ├── wolfram/          # Mathematical validation
│   ├── data/             # Generated datasets
│   └── docs/             # Experiment documentation
├── docs/                  # General documentation
│   ├── PROOF_PLAN.md
│   ├── SCRIPT_INVENTORY.md
│   └── context_alpha_validation_spec.md
├── setup/                 # Setup and utility scripts
│   ├── install_dependencies.sh
│   ├── run_validation.py
│   └── verify_results.py
└── results/              # Test results (gitignored)
```

## Theory

Information Reconstructionism posits that information exists as a multiplicative function:

```
Information(i→j|S-O) = WHERE × WHAT × CONVEYANCE × TIME × FRAME
```

## Experiments

### Experiment 1: Semantic Context Mapping
- Maps WHERE × WHAT × Context(top-k)
- Validates zero propagation principle
- Tests Context^α amplification
- **Status**: Complete ✓

### Experiment 2: Conveyance Mapping (Planned)
- Will map WHERE × WHAT × Conveyance
- Measure transformation potential
- Test Context-Conveyance correlation

## Quick Start

```bash
# Install dependencies
cd setup && ./install_dependencies.sh

# Run test pipeline
cd ../experiment_1/pipeline
python test_pipeline_10.py

# Run full validation
python run_batch.py 1000
```

## Required Environment Variables

```bash
export ARANGO_HOST="http://192.168.1.69:8529"
export ARANGO_USERNAME="root"
export ARANGO_PASSWORD="your_password"
export WOLFRAM_APP_ID="your_app_id"
```

## Key Findings

1. **Zero Propagation**: Mathematically validated ✓
2. **Context Amplification**: Context^1.5 creates natural clustering ✓
3. **Graph Structure**: Semantic similarity forms dense connections ✓

## Next Steps

- [ ] Complete conveyance measurement implementation
- [ ] Validate Context-Conveyance correlation
- [ ] Test on larger datasets (10M+ papers)
- [ ] Build practical applications