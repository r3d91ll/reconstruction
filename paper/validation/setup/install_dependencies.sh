#!/bin/bash
# Install all dependencies for Information Reconstructionism validation

echo "Installing dependencies for Jina V4 embeddings..."

# Core dependencies
pip install transformers torch python-arango

# Jina V4 specific requirements
pip install peft  # Parameter-Efficient Fine-Tuning

# Optional but useful
pip install numpy pandas matplotlib tqdm

echo "Dependencies installed!"
echo ""
echo "Checking installations..."
python3 -c "import transformers; print(f'✓ transformers {transformers.__version__}')"
python3 -c "import torch; print(f'✓ torch {torch.__version__}')"
python3 -c "import arango; print(f'✓ python-arango installed')"
python3 -c "import peft; print(f'✓ peft {peft.__version__}')"

echo ""
echo "Ready to run validation pipeline!"