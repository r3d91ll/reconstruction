# Information Reconstructionism

A new theory of observer-dependent information existence with supporting infrastructure for validation.

## Project Structure

This repository contains two main components that work together:

### 📚 Paper (`paper/`)
The theoretical framework, mathematical proofs, and academic materials for Information Reconstructionism.

- **Core Theory**: Mathematical framework proving information as multiplicative function
- **Validation**: Wolfram scripts and Python demonstrations
- **Evidence**: Empirical validation of zero propagation and context amplification
- See [`paper/PAPER_CLAUDE.md`](paper/PAPER_CLAUDE.md) for theory-specific guidance

### 🏗️ Infrastructure (`infrastructure/`)
Scalable document processing system for validating the theory on large academic corpora.

- **3-Collection Architecture**: Atomic storage of metadata, documents, and chunks
- **GPU Processing**: Dual A6000 GPUs with Jina v4 embeddings
- **Graph Database**: ArangoDB for similarity and relationship queries
- See [`infrastructure/INFRA_CLAUDE.md`](infrastructure/INFRA_CLAUDE.md) for implementation details

### 🧪 Experiments (`experiments/`)
Shared experiments that use the infrastructure to validate theoretical claims.

- **Experiment 1**: Multiplicative model validation
- **Experiment 2**: Large-scale GPU processing and analysis

## Core Equation

```
Information(i→j|S-O) = WHERE × WHAT × CONVEYANCE × TIME × FRAME
```

If ANY dimension = 0, then Information = 0

## Quick Start

### Theory Work
```bash
cd paper/
# Review theoretical framework
cat theory/main_theory.md
# Run Wolfram validation
mathematica validation/wolfram/zero_propagation.nb
```

### Infrastructure Setup
```bash
cd infrastructure/
./setup_venv.sh
source venv/bin/activate
pip install -r setup/requirements.txt

# Process documents
python setup/process_documents_atomic.py --count 10 --clean-start
```

### Running Experiments
```bash
cd experiments/
# Test multiplicative model
python experiment_1/run_experiment.py
# Large-scale processing
./experiment_2/pipeline/launch_gpu_pipeline.sh
```

## Key Innovation

Information Reconstructionism differs from traditional information theory by:

1. **Observer Dependency**: Information exists relative to observers
2. **Multiplicative Model**: All dimensions must be > 0
3. **Context Amplification**: Context acts as exponential amplifier (Context^α)
4. **Theory-Practice Bridges**: High-conveyance connections between abstract and concrete

## Current Status

- ✅ Core theoretical framework documented
- ✅ Mathematical foundations laid
- ✅ Infrastructure implemented with atomic transactions
- 🔄 Validation experiments in progress
- 📝 Paper preparation for academic publication

## Future Separation

The infrastructure is designed to be easily separated into its own repository when the time comes. Simply move the `infrastructure/` directory and update import paths in experiments.

## Contributing

This is an active research project. For questions or contributions:
- Theory/Paper: Focus on mathematical rigor and empirical validation
- Infrastructure: Maintain atomic transaction integrity and GPU optimization

## License

Apache 2.0 - See LICENSE file for details